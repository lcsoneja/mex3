"""Owner voiceprint, multi-master wake lock & historical speaker clustering (CAM++).

Architecture (designed for noisy rooms & 4GB Raspberry Pi 4):
1. Multi-Master Pool (up to 4 active masters): any person who says "Nova" is
   added/refreshed as an active wake master with a 5-min idle timeout — multiple
   people can wake & talk without evicting each other.
2. Stay quiet until woken: when no master is active (initial idle or after 5m
   timeout), room chatter without "Nova" is held cleanly until someone says "Nova".
3. Adaptive Noise-Floor SNR Gate: tracks ambient room RMS while silent and
   requires speech >= max(0.024, 2.2 * noise_floor) + >=0.65s voiced audio before
   registering a new speaker (prevents 1 person splitting into 5 in noisy rooms).
4. Historical Speaker Persistence (`out/speaker_history.json`) + Multi-Prototype
   Profiles (up to 4 vectors/speaker) + automatic drift consolidation.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


class VoiceprintError(RuntimeError):
    """Raised when the speaker model is missing or embedding fails."""


def _speech_metrics(
    samples: np.ndarray, rate: int = 16000, frame_rms_min: float = 0.016
) -> tuple[float, float]:
    """Measure voiced duration (s) & RMS without altering the waveform fed to CAM++."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    win = max(160, int(rate * 0.03))
    voiced_samples = 0
    voiced_sq = 0.0
    for i in range(0, arr.size, win):
        chunk = arr[i : i + win]
        if chunk.size < win // 2:
            continue
        sq = float(np.mean(np.square(chunk)) or 0.0)
        if np.sqrt(sq) >= frame_rms_min:
            voiced_samples += int(chunk.size)
            voiced_sq += sq * int(chunk.size)
    if voiced_samples == 0:
        return 0.0, float(np.sqrt(np.mean(np.square(arr))) or 0.0)
    return voiced_samples / float(rate), float(np.sqrt(voiced_sq / voiced_samples))


class Voiceprint:
    """Multi-master wake lock, adaptive SNR gating & persistent 7-speaker roster."""

    def __init__(
        self,
        model_path: Path | str,
        store_path: Path | str,
        threshold: float = 0.35,
        match_threshold: float = 0.34,
        wake_idle_timeout_s: float = 300.0,
        max_speakers: int = 7,
        history_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.store_path = Path(store_path)
        self.history_path = (
            Path(history_path)
            if history_path is not None
            else self.store_path.with_name("speaker_history.json")
        )
        self.threshold = float(threshold)
        self.match_threshold = float(match_threshold)
        self.wake_idle_timeout_s = float(wake_idle_timeout_s)
        self.max_speakers = int(max_speakers)
        self._lock = threading.Lock()
        self._extractor = None
        self._embedding: np.ndarray | None = None
        if self.store_path.exists():
            try:
                self._embedding = np.load(self.store_path)
            except (OSError, ValueError):
                self._embedding = None

        self._speakers: dict[str, dict[str, Any]] = {}
        self._next_spk_num = 1
        # Multi-master pool: {spk_id: last_active_epoch}, max 4 simultaneous masters
        self._masters: dict[str, float] = {}
        self._short_wake_candidates: dict[str, float] = {}
        self._last_speaker_id: str = ""
        self._last_speaker_at: float = 0.0
        self._last_embed_ms: float = 0.0
        self._avg_embed_ms: float = 0.0
        self._embed_count: int = 0
        self._model_rss_mb: float = 0.0
        self._events: deque[dict[str, Any]] = deque(maxlen=30)
        self._bg_last_at: float = 0.0
        # Adaptive ambient room noise floor (updated by _listen_loop during silence)
        self._noise_floor: float = 0.008
        self._last_snr: float = 0.0

        self._load_history()
        if self._embedding is not None and "owner" not in self._speakers:
            self._speakers["owner"] = self._new_profile(
                "owner", "Owner", self._embedding
            )

    @staticmethod
    def _new_profile(spk_id: str, label: str, emb: np.ndarray) -> dict[str, Any]:
        return {
            "id": spk_id,
            "label": label,
            "emb": emb,
            "protos": [emb],
            "count": 0,
            "last_seen": 0.0,
            "last_rms": 0.0,
            "peak_rms": 0.0,
            "last_sim": 1.0,
            "ignored": False,
            "volumes": deque(maxlen=36),
        }

    def _load_history(self) -> None:
        if not self.history_path.is_file():
            return
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in (raw.get("speakers") or [])[: self.max_speakers]:
            sid = str(row.get("id") or "")
            protos_raw = row.get("protos") or []
            if not sid or not protos_raw:
                continue
            protos = [np.asarray(p, dtype=np.float32) for p in protos_raw[:4]]
            prof = self._new_profile(sid, str(row.get("label") or sid), protos[0])
            prof["protos"] = protos
            prof["count"] = int(row.get("count") or 0)
            prof["ignored"] = bool(row.get("ignored"))
            prof["last_seen"] = float(row.get("last_seen") or 0.0)
            self._speakers[sid] = prof
            if sid.startswith("spk_"):
                try:
                    self._next_spk_num = max(
                        self._next_spk_num, int(sid.split("_", 1)[1]) + 1
                    )
                except ValueError:
                    pass

    def _save_history_locked(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for sid, prof in list(self._speakers.items())[: self.max_speakers]:
                rows.append(
                    {
                        "id": sid,
                        "label": prof["label"],
                        "count": int(prof["count"]),
                        "ignored": bool(prof.get("ignored")),
                        "last_seen": float(prof["last_seen"]),
                        "protos": [
                            np.asarray(p, dtype=float).tolist()
                            for p in (prof.get("protos") or [])[:4]
                        ],
                    }
                )
            self.history_path.write_text(
                json.dumps({"speakers": rows}, indent=1), encoding="utf-8"
            )
        except OSError:
            pass

    def update_noise_floor(self, rms: float) -> None:
        """Track ambient noise RMS during non-speech blocks for adaptive SNR gating."""
        val = float(rms)
        if 0.0 < val < 0.035:
            self._noise_floor = 0.94 * self._noise_floor + 0.06 * val

    def _min_speech_rms(self) -> float:
        return max(0.024, self._noise_floor * 2.2)

    def _log_decision_locked(
        self, label: str, decision: str, sim: float, rms: float, detail: str = ""
    ) -> None:
        self._events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "label": label,
                "decision": decision,
                "sim": round(float(sim), 2),
                "rms": round(float(rms), 3),
                "detail": detail,
            }
        )

    @property
    def model_present(self) -> bool:
        return self.model_path.is_file()

    @property
    def enrolled(self) -> bool:
        return self._embedding is not None

    def warm(self) -> None:
        if self.model_present:
            with self._lock:
                self._get_extractor()

    def _get_extractor(self):
        if self._extractor is None:
            if not self.model_present:
                raise VoiceprintError(f"speaker model not found: {self.model_path}")
            import sherpa_onnx

            rss_before = self._proc_rss_mb()
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path), num_threads=2, provider="cpu"
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            delta = self._proc_rss_mb() - rss_before
            file_mb = self.model_path.stat().st_size / (1024 * 1024)
            self._model_rss_mb = round(max(delta, file_mb), 1)
        return self._extractor

    @staticmethod
    def _proc_rss_mb() -> float:
        try:
            import os

            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return 0.0

    def _embed(self, samples: np.ndarray, rate: int = 16000) -> np.ndarray:
        t0 = time.perf_counter()
        extractor = self._get_extractor()
        stream = extractor.create_stream()
        stream.accept_waveform(
            int(rate), np.asarray(samples, dtype=np.float32).reshape(-1)
        )
        vec = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        out = vec / norm if norm > 0 else vec
        ms = (time.perf_counter() - t0) * 1000.0
        self._last_embed_ms = round(ms, 1)
        self._embed_count += 1
        self._avg_embed_ms = round(
            self._avg_embed_ms + (ms - self._avg_embed_ms) / min(self._embed_count, 20),
            1,
        )
        return out

    @staticmethod
    def _profile_sim(prof: dict[str, Any], emb: np.ndarray) -> float:
        protos = prof.get("protos") or [prof["emb"]]
        return max(float(np.dot(p, emb)) for p in protos)

    def enroll(self, samples: np.ndarray, rate: int = 16000) -> bool:
        with self._lock:
            emb = self._embed(samples, rate)
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.store_path, emb)
            self._embedding = emb
            self._speakers["owner"] = self._new_profile("owner", "Owner", emb)
            self._save_history_locked()
        return True

    def score(self, samples: np.ndarray, rate: int = 16000) -> float:
        with self._lock:
            if self._embedding is None:
                return 0.0
            emb = self._embed(samples, rate)
            owner = self._speakers.get("owner")
            if owner is not None:
                return self._profile_sim(owner, emb)
            return float(np.dot(self._embedding, emb))

    def accepts(self, samples: np.ndarray, rate: int = 16000) -> bool:
        return self.score(samples, rate) >= self.threshold

    def clear(self) -> None:
        with self._lock:
            self._embedding = None
            self.store_path.unlink(missing_ok=True)
            self._speakers.pop("owner", None)
            self._masters.pop("owner", None)
            self._save_history_locked()

    # ------------------------------------------------- multi-master & historical clustering
    def _expire_masters_locked(self, now: float) -> None:
        expired = [
            sid
            for sid, ts in self._masters.items()
            if ts > 0.0 and (now - ts) > self.wake_idle_timeout_s
        ]
        for sid in expired:
            self._masters.pop(sid, None)
            self._short_wake_candidates.pop(sid, None)
            lbl = self._speakers.get(sid, {}).get("label", sid)
            self._log_decision_locked(str(lbl), "EXPIRED", 0.0, 0.0, "5m idle timeout")

    def _add_master_locked(self, sid: str, now: float, short_wake: bool) -> str:
        already = sid in self._masters
        if not already and len(self._masters) >= 4:
            oldest = min(self._masters, key=lambda k: self._masters[k])
            self._masters.pop(oldest, None)
        self._masters[sid] = now
        if short_wake:
            self._short_wake_candidates[sid] = now
        else:
            self._short_wake_candidates.pop(sid, None)
        return "MASTER REFRESH" if already else "MASTER ADDED"

    def _consolidate_locked(self) -> None:
        """Historical self-cleanup: merge two dynamic clusters whose prototypes drifted close."""
        ids = [k for k in self._speakers if k != "owner"]
        for i, a in enumerate(ids):
            if a not in self._speakers:
                continue
            for b in ids[i + 1 :]:
                if b not in self._speakers:
                    continue
                pa, pb = self._speakers[a], self._speakers[b]
                sim = max(
                    float(np.dot(va, vb))
                    for va in (pa.get("protos") or [pa["emb"]])
                    for vb in (pb.get("protos") or [pb["emb"]])
                )
                if sim >= 0.42:
                    # Keep the higher-count / active-master profile
                    keep, drop = (
                        (a, b)
                        if (a in self._masters or pa["count"] >= pb["count"])
                        else (b, a)
                    )
                    self._merge_locked(drop, keep, reason="auto-consolidated")

    def _match_or_register_locked(
        self, emb: np.ndarray, now: float, rms: float, can_spawn: bool
    ) -> tuple[str, str, float]:
        best_id = ""
        best_sim = -1.0
        for sid, prof in self._speakers.items():
            sim = self._profile_sim(prof, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        gate = self.match_threshold
        if best_id == "owner":
            gate = max(self.match_threshold, self.threshold - 0.05)
        elif best_id == self._last_speaker_id and (now - self._last_speaker_at) < 3.5:
            gate = max(0.30, self.match_threshold - 0.04)

        if best_id and best_sim >= gate:
            prof = self._speakers[best_id]
            protos: list[np.ndarray] = prof.setdefault("protos", [prof["emb"]])
            if best_sim < 0.65 and can_spawn:
                if len(protos) < 4:
                    protos.append(emb)
                else:
                    idx = max(
                        range(len(protos)), key=lambda i: float(np.dot(protos[i], emb))
                    )
                    b = 0.8 * protos[idx] + 0.2 * emb
                    n = float(np.linalg.norm(b))
                    if n > 0:
                        protos[idx] = b / n
            prof["count"] += 1
            prof["last_seen"] = now
            prof["last_rms"] = round(rms, 4)
            prof["peak_rms"] = round(max(float(prof["peak_rms"]), rms), 4)
            prof["last_sim"] = round(best_sim, 3)
            prof["volumes"].append(round(rms, 4))
            self._last_speaker_id = best_id
            self._last_speaker_at = now
            if prof["count"] % 5 == 0:
                self._consolidate_locked()
                self._save_history_locked()
            return best_id, str(prof["label"]), best_sim

        if not can_spawn:
            if best_id:
                return (
                    best_id,
                    str(self._speakers[best_id]["label"]),
                    max(0.0, best_sim),
                )
            return "", "Quiet/Short", 0.0

        if len(self._speakers) >= self.max_speakers:
            victims = [
                k for k in self._speakers if k != "owner" and k not in self._masters
            ]
            victims.sort(key=lambda k: self._speakers[k]["last_seen"])
            if victims:
                self._speakers.pop(victims[0], None)

        sid = f"spk_{self._next_spk_num}"
        label = f"Speaker {self._next_spk_num}"
        self._next_spk_num += 1
        prof = self._new_profile(sid, label, emb)
        prof["count"] = 1
        prof["last_seen"] = now
        prof["last_rms"] = round(rms, 4)
        prof["peak_rms"] = round(rms, 4)
        prof["last_sim"] = 1.0
        prof["volumes"].append(round(rms, 4))
        self._speakers[sid] = prof
        self._last_speaker_id = sid
        self._last_speaker_at = now
        self._save_history_locked()
        return sid, label, 1.0

    def observe_background(
        self, samples: np.ndarray, rate: int = 16000
    ) -> dict[str, Any]:
        """Lightweight CAM++ scan of clear voiced audio (SNR-gated against room noise)."""
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        voiced_s, vrms = _speech_metrics(arr, rate)
        now = time.time()
        with self._lock:
            self._expire_masters_locked(now)
            self._last_snr = round(vrms / max(0.002, self._noise_floor), 1)
            if (
                not self.model_present
                or voiced_s < 0.75
                or vrms < self._min_speech_rms()
            ):
                return {}
            emb = self._embed(arr, rate)
            sid, label, sim = self._match_or_register_locked(
                emb, now, vrms, can_spawn=True
            )
            if not sid:
                return {}
            self._bg_last_at = now
            self._log_decision_locked(
                label, "BG TRACKED", sim, vrms, f"snr {self._last_snr}x"
            )
            return {
                "id": sid,
                "label": label,
                "similarity": round(sim, 3),
                "rms": round(vrms, 4),
            }

    def classify_and_track(
        self,
        samples: np.ndarray,
        rate: int = 16000,
        is_wake: bool = False,
        wake_lock_enabled: bool = False,
        require_wake_when_idle: bool = True,
    ) -> dict[str, Any]:
        """Multi-master wake lock, SNR noise gating & stay-quiet-until-Nova policy."""
        target = np.asarray(samples, dtype=np.float32).reshape(-1)
        voiced_s, rms = _speech_metrics(target, rate)
        now = time.time()
        with self._lock:
            self._expire_masters_locked(now)
            self._last_snr = round(rms / max(0.002, self._noise_floor), 1)
            can_spawn = bool(voiced_s >= 0.65 and rms >= self._min_speech_rms())
            if not self.model_present or target.size < int(rate * 0.25):
                allowed = (
                    (not wake_lock_enabled)
                    or (not require_wake_when_idle)
                    or is_wake
                    or bool(self._masters)
                )
                return {
                    "id": "",
                    "label": "Voice",
                    "similarity": 0.0,
                    "rms": round(rms, 4),
                    "is_master": False,
                    "allowed": allowed,
                    "master_label": self._masters_summary_locked(),
                }

            emb = self._embed(target, rate)
            sid, label, sim = "", "", 0.0
            calibrated = False
            # Wake-calibration grace window for any recently-woken short-"Nova" master
            if (
                not is_wake
                and wake_lock_enabled
                and self._short_wake_candidates
                and voiced_s >= 0.55
            ):
                best_m, best_msim = "", -1.0
                for m_id, w_ts in list(self._short_wake_candidates.items()):
                    if (now - w_ts) > 8.0:
                        self._short_wake_candidates.pop(m_id, None)
                        continue
                    mp = self._speakers.get(m_id)
                    if mp is not None and not mp.get("ignored"):
                        ms = self._profile_sim(mp, emb)
                        if ms > best_msim:
                            best_msim, best_m = ms, m_id
                if best_m and best_msim >= 0.18:
                    mp = self._speakers[best_m]
                    protos = mp.setdefault("protos", [mp["emb"]])
                    if len(protos) < 4:
                        protos.append(emb)
                    mp["count"] += 1
                    mp["last_seen"] = now
                    mp["last_rms"] = round(rms, 4)
                    mp["peak_rms"] = round(max(float(mp["peak_rms"]), rms), 4)
                    mp["last_sim"] = round(best_msim, 3)
                    mp["volumes"].append(round(rms, 4))
                    self._last_speaker_id = best_m
                    self._last_speaker_at = now
                    self._short_wake_candidates.pop(best_m, None)
                    self._save_history_locked()
                    sid, label, sim = best_m, str(mp["label"]), best_msim
                    calibrated = True

            if not calibrated:
                sid, label, sim = self._match_or_register_locked(
                    emb, now, rms, can_spawn=can_spawn or is_wake
                )

            prof = self._speakers.get(sid, {})
            if prof.get("ignored"):
                self._log_decision_locked(
                    label, "BLOCKED", sim, rms, "speaker ignored/muted"
                )
                return {
                    "id": sid,
                    "label": label,
                    "similarity": round(sim, 3),
                    "rms": round(rms, 4),
                    "is_master": False,
                    "allowed": False,
                    "blocked": True,
                    "master_label": self._masters_summary_locked(),
                }

            if is_wake and wake_lock_enabled and sid:
                kind = self._add_master_locked(
                    sid, now, short_wake=bool(voiced_s < 1.1)
                )
                self._log_decision_locked(
                    label, kind, sim, rms, f"wake word ({len(self._masters)} active)"
                )

            allowed = True
            is_master = bool(sid and sid in self._masters)
            if wake_lock_enabled and not is_wake:
                if not self._masters:
                    if require_wake_when_idle:
                        allowed = False
                        self._log_decision_locked(
                            label or "Voice",
                            "AWAITING WAKE",
                            sim,
                            rms,
                            "no active master — say Nova",
                        )
                else:
                    # Match against ANY active master in the multi-master pool
                    best_m, best_msim = "", -1.0
                    for m_id in self._masters:
                        mp = self._speakers.get(m_id)
                        if mp is not None:
                            ms = self._profile_sim(mp, emb)
                            if ms > best_msim:
                                best_msim, best_m = ms, m_id
                    master_gate = max(0.24, self.match_threshold - 0.03)
                    if (sid in self._masters) or (best_m and best_msim >= master_gate):
                        target_m = sid if sid in self._masters else best_m
                        self._masters[target_m] = now
                        is_master = True
                        allowed = True
                        self._log_decision_locked(
                            self._speakers[target_m]["label"],
                            "ACCEPTED",
                            max(sim, best_msim),
                            rms,
                            "active master",
                        )
                    else:
                        allowed = False
                        sim = max(0.0, best_msim)
                        self._log_decision_locked(
                            label or "Voice",
                            "IGNORED",
                            sim,
                            rms,
                            f"not in masters ({self._masters_summary_locked()})",
                        )
            elif not is_wake and sid:
                self._log_decision_locked(label, "ACCEPTED", sim, rms, "unlocked")

            return {
                "id": sid,
                "label": label,
                "similarity": round(sim, 3),
                "rms": round(rms, 4),
                "is_master": is_master,
                "allowed": allowed,
                "master_label": self._masters_summary_locked(),
            }

    def _masters_summary_locked(self) -> str:
        return ", ".join(
            str(self._speakers.get(sid, {}).get("label", sid)) for sid in self._masters
        )

    # ------------------------------------------------- per-speaker management actions
    def rename_speaker(self, spk_id: str, new_label: str) -> dict[str, Any]:
        clean = str(new_label or "").strip()[:32]
        if not clean:
            return {"ok": False, "error": "label cannot be empty"}
        with self._lock:
            prof = self._speakers.get(spk_id)
            if prof is None:
                return {"ok": False, "error": f"unknown speaker: {spk_id}"}
            old = prof["label"]
            prof["label"] = clean
            self._save_history_locked()
            self._log_decision_locked(clean, "RENAMED", 1.0, 0.0, f"from {old}")
            return {"ok": True, "id": spk_id, "label": clean}

    def _merge_locked(
        self, source_id: str, target_id: str, reason: str = "manual"
    ) -> dict[str, Any]:
        src = self._speakers.get(source_id)
        dst = self._speakers.get(target_id)
        if src is None or dst is None:
            return {"ok": False, "error": "speaker not found"}
        protos: list[np.ndarray] = dst.setdefault("protos", [dst["emb"]])
        for p in src.get("protos") or [src["emb"]]:
            if len(protos) < 6:
                protos.append(p)
        dst["count"] = int(dst["count"]) + int(src["count"])
        dst["peak_rms"] = max(float(dst["peak_rms"]), float(src["peak_rms"]))
        if source_id in self._masters:
            ts = self._masters.pop(source_id)
            self._masters[target_id] = max(self._masters.get(target_id, 0.0), ts)
        if self._last_speaker_id == source_id:
            self._last_speaker_id = target_id
        src_lbl = src["label"]
        self._speakers.pop(source_id, None)
        self._save_history_locked()
        self._log_decision_locked(
            str(dst["label"]), "MERGED", 1.0, 0.0, f"{reason}: {src_lbl}"
        )
        return {"ok": True, "into": dst["label"], "protos": len(protos)}

    def merge_speakers(self, source_id: str, target_id: str) -> dict[str, Any]:
        if not source_id or not target_id or source_id == target_id:
            return {"ok": False, "error": "pick two different speakers"}
        with self._lock:
            return self._merge_locked(source_id, target_id, reason="absorbed")

    def set_match_threshold(self, value: float) -> dict[str, Any]:
        val = max(0.18, min(0.60, float(value)))
        with self._lock:
            self.match_threshold = round(val, 2)
            return {"ok": True, "match_threshold": self.match_threshold}

    def set_ignored(self, spk_id: str, ignored: bool) -> dict[str, Any]:
        with self._lock:
            prof = self._speakers.get(spk_id)
            if prof is None:
                return {"ok": False, "error": f"unknown speaker: {spk_id}"}
            prof["ignored"] = bool(ignored)
            if ignored:
                self._masters.pop(spk_id, None)
            self._save_history_locked()
            self._log_decision_locked(
                str(prof["label"]),
                "MUTED" if ignored else "UNMUTED",
                1.0,
                0.0,
                "manual",
            )
            return {"ok": True, "id": spk_id, "ignored": prof["ignored"]}

    def make_master(self, spk_id: str) -> dict[str, Any]:
        with self._lock:
            prof = self._speakers.get(spk_id)
            if prof is None:
                return {"ok": False, "error": f"unknown speaker: {spk_id}"}
            prof["ignored"] = False
            self._add_master_locked(spk_id, time.time(), short_wake=False)
            self._log_decision_locked(
                str(prof["label"]), "MASTER ADDED", 1.0, 0.0, "manual pin"
            )
            return {
                "ok": True,
                "master_id": spk_id,
                "master_label": self._masters_summary_locked(),
            }

    def promote_owner(self, spk_id: str) -> dict[str, Any]:
        with self._lock:
            prof = self._speakers.get(spk_id)
            if prof is None:
                return {"ok": False, "error": f"unknown speaker: {spk_id}"}
            emb = np.asarray(prof["emb"], dtype=np.float32)
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.store_path, emb)
            self._embedding = emb
            self._speakers["owner"] = self._new_profile("owner", "Owner", emb)
            self._speakers["owner"]["protos"] = list(prof.get("protos") or [emb])
            self._save_history_locked()
            self._log_decision_locked(
                str(prof["label"]), "SET OWNER", 1.0, 0.0, "saved voiceprint"
            )
            return {"ok": True, "enrolled": True}

    def reset_roster(self) -> dict[str, Any]:
        with self._lock:
            owner = self._speakers.get("owner")
            self._speakers = {"owner": owner} if owner is not None else {}
            self._next_spk_num = 1
            self._masters.clear()
            self._short_wake_candidates.clear()
            self._last_speaker_id = ""
            self._last_speaker_at = 0.0
            self._events.clear()
            self.history_path.unlink(missing_ok=True)
            return {"ok": True}

    def unlock_master(self) -> None:
        with self._lock:
            self._masters.clear()
            self._short_wake_candidates.clear()

    def dashboard(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._expire_masters_locked(now)
            masters_list = [
                {
                    "id": sid,
                    "label": str(self._speakers.get(sid, {}).get("label", sid)),
                    "ttl_s": max(0, int(self.wake_idle_timeout_s - (now - ts))),
                }
                for sid, ts in self._masters.items()
                if sid in self._speakers
            ]
            max_ttl = max((m["ttl_s"] for m in masters_list), default=0)
            ordered = sorted(
                self._speakers.values(),
                key=lambda p: (
                    1 if p["id"] in self._masters else 0,
                    float(p["last_seen"]),
                    int(p["count"]),
                ),
                reverse=True,
            )[: self.max_speakers]
            speakers = [
                {
                    "id": p["id"],
                    "label": p["label"],
                    "count": int(p["count"]),
                    "protos": len(p.get("protos") or [p["emb"]]),
                    "last_rms": float(p["last_rms"]),
                    "peak_rms": float(p["peak_rms"]),
                    "last_sim": float(p["last_sim"]),
                    "ignored": bool(p.get("ignored")),
                    "volumes": list(p["volumes"]),
                    "is_master": p["id"] in self._masters,
                    "is_talking": (p["id"] == self._last_speaker_id)
                    and (now - float(p["last_seen"]) < 3.0),
                    "ago_s": round(max(0.0, now - float(p["last_seen"])), 1)
                    if p["last_seen"]
                    else None,
                }
                for p in ordered
            ]
            model_mb = self._model_rss_mb or (
                round(self.model_path.stat().st_size / (1024 * 1024), 1)
                if self.model_present
                else 0.0
            )
            bg_active = (now - self._bg_last_at) < 4.0 if self._bg_last_at else False
            return {
                "model_present": self.model_present,
                "enrolled": self.enrolled,
                "threshold": self.threshold,
                "match_threshold": self.match_threshold,
                "max_speakers": self.max_speakers,
                "master_id": masters_list[0]["id"] if masters_list else "",
                "master_label": self._masters_summary_locked(),
                "master_ttl_s": max_ttl,
                "masters": masters_list,
                "noise_floor": round(self._noise_floor, 4),
                "snr_ratio": self._last_snr,
                "bg_scanning_now": bg_active,
                "speakers": speakers,
                "top5": speakers[:5],
                "events": list(reversed(self._events)),
                "telemetry": {
                    "mem_mb": model_mb,
                    "last_embed_ms": self._last_embed_ms,
                    "avg_embed_ms": self._avg_embed_ms,
                    "cpu_pct_est": round(min(100.0, self._avg_embed_ms / 10.0), 1),
                    "net_kbps": 0.0,
                },
            }
