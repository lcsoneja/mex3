"""SONJ4 non-LLM Decision Classifier (<100 ms budget, +10% Front bias).

Sits between the user utterance/message and Front. Takes a structured
DecisionRequest (text + compact session context) and returns a fixed-choice
Decision (`front` | `back` | `doctor`) with zero LLM prose or question
generation.

Architecture:
    Stage 0: text normalization (<0.2 ms)
    Stage 1: deterministic rule/keyword feature scorer (<0.5 ms)
    Stage 2: optional local ONNX/linear scorer plug-in under models/classifier/,
             bounded by the remaining <90 ms deadline (aborts cleanly to Stage 1
             or Front fallback if exceeded)
    Bias   : +0.10 added to score_front in the Front-vs-Back decision; Back must
             beat Front by >0.10 to dispatch a pool worker. Low-confidence ties
             abstain to `front`.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

RouteChoice = Literal["front", "back", "doctor"]
CHOICES: tuple[RouteChoice, ...] = ("front", "back", "doctor")

FRONT_BIAS = 0.10
DEADLINE_MS = 90.0
ABSTAIN_FLOOR = 0.50
DOCTOR_THRESHOLD = 0.65

BASE = Path(__file__).resolve().parent
ONNX_DIR = BASE / "models" / "classifier"

# Stage 2 callable signature: req -> (score_front, score_back, score_doctor, reason_code)
Stage2Fn = Callable[["DecisionRequest"], tuple[float, float, float, str]]


@dataclass(frozen=True)
class DecisionRequest:
    """Typed classifier input (text -> typed question -> structured decision)."""

    text: str
    session_state: str = "idle"
    active_backs: int = 0
    recent_route: str = ""
    choices: tuple[str, ...] = CHOICES


@dataclass(frozen=True)
class Decision:
    """Structured classification output — never free text."""

    route: RouteChoice
    confidence: float
    reason_code: str
    latency_ms: float
    score_front: float
    score_back: float
    score_doctor: float
    bias_applied: bool = True
    fallback: bool = False
    suggested_mode: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Signal dictionaries (Stage 1 rules-first feature scorer) ---

DOCTOR_STRONG = (
    "doctor",
    "health check",
    "system health",
    "incident log",
    "incidents",
    "diagnose",
    "diagnostic",
    "diagnostics",
    "app error",
    "app errors",
    "errors in the",
    "check the errors",
    "check errors",
    "check the logs",
    "app log",
    "wake_gate",
    "config.json",
    "inspect architecture",
    "app architecture",
    "app settings",
    "change setting",
    "propose setting",
    "tasks failed",
    "task failed",
    "troubleshoot",
)

DOCTOR_WEAK = (
    "error",
    "errors",
    "log",
    "logs",
    "setting",
    "settings",
    "health",
    "architecture",
    "restart",
    "status of the app",
    "voiceprint",
    "barge",
)

FRONT_DEVICE = (
    "what time",
    "time is it",
    "turn on",
    "turn off",
    "light",
    "lights",
    "lamp",
    "dim ",
    "brighten",
    "set a timer",
    "set timer",
    "timer for",
    "alarm",
    "remind me",
    "list reminders",
    "volume",
    "pause music",
    "play music",
    "next track",
    "previous track",
    "remember that",
    "favorite color",
    "my name is",
    "calculate ",
    "tell me a joke",
    "short joke",
    "hello",
    "hi ",
    "hey ",
    "how are you",
    "good morning",
    "good night",
    "thank you",
    "thanks",
)

BACK_BUILD = (
    "create ",
    "build ",
    "make a ",
    "make me ",
    "write a python",
    "write a script",
    "python script",
    "snake game",
    "game",
    "website",
    "webpage",
    "html",
    "app ",
    "bio link",
    "linktree",
    "code ",
    "automate",
    "in parallel",
    "two small scripts",
    "sandbox",
)

BACK_RESEARCH = (
    "research ",
    "latest news",
    "headline",
    "headlines",
    "official gazette",
    "philippine gazette",
    "download ",
    "fetch ",
    "look up",
    "search the web",
    "search more",
    "find out",
    "source link",
    "give me a source",
    "current price",
    "weather trends",
)

BACK_VERIFY = (
    "verify ",
    "audit ",
    "fact-check",
    "fact check",
    "is this claim",
    "fake or real",
    "real or fake",
)

BACK_SUMMARIZE = (
    "summarize",
    "summarise",
    "condense",
    "tl;dr",
    "tldr",
    "top 3 takeaways",
)

_MATH_RE = re.compile(r"\b\d+\s*[*+\-/x]\s*\d+\b")


def apply_front_bias(
    score_front: float,
    score_back: float,
    score_doctor: float,
    *,
    reason_code: str = "scored",
    suggested_mode: str = "general",
    latency_ms: float = 0.0,
    fallback: bool = False,
) -> Decision:
    """Apply the +10% Front bias to Front-vs-Back and enforce Doctor threshold."""
    sf = max(0.0, min(1.0, float(score_front)))
    sb = max(0.0, min(1.0, float(score_back)))
    sd = max(0.0, min(1.0, float(score_doctor)))

    if sd >= DOCTOR_THRESHOLD and sd >= max(sf, sb):
        return Decision(
            route="doctor",
            confidence=round(sd, 4),
            reason_code=reason_code,
            latency_ms=round(latency_ms, 3),
            score_front=round(sf, 4),
            score_back=round(sb, 4),
            score_doctor=round(sd, 4),
            bias_applied=False,
            fallback=fallback,
            suggested_mode="diagnose",
        )

    biased_front = min(1.0, sf + FRONT_BIAS)
    peak = max(biased_front, sb)
    if peak < ABSTAIN_FLOOR:
        return Decision(
            route="front",
            confidence=round(biased_front, 4),
            reason_code="abstain_front",
            latency_ms=round(latency_ms, 3),
            score_front=round(biased_front, 4),
            score_back=round(sb, 4),
            score_doctor=round(sd, 4),
            bias_applied=True,
            fallback=fallback,
            suggested_mode="chat",
        )

    if biased_front >= sb:
        return Decision(
            route="front",
            confidence=round(biased_front, 4),
            reason_code=reason_code,
            latency_ms=round(latency_ms, 3),
            score_front=round(biased_front, 4),
            score_back=round(sb, 4),
            score_doctor=round(sd, 4),
            bias_applied=True,
            fallback=fallback,
            suggested_mode=suggested_mode
            if suggested_mode in {"device", "chat"}
            else "chat",
        )

    return Decision(
        route="back",
        confidence=round(sb, 4),
        reason_code=reason_code,
        latency_ms=round(latency_ms, 3),
        score_front=round(biased_front, 4),
        score_back=round(sb, 4),
        score_doctor=round(sd, 4),
        bias_applied=True,
        fallback=fallback,
        suggested_mode=suggested_mode
        if suggested_mode not in {"device", "chat", "diagnose"}
        else "general",
    )


def _stage1_scores(text: str) -> tuple[float, float, float, str, str]:
    """Deterministic sub-millisecond rule/keyword scorer."""
    low = " ".join(str(text or "").lower().split())
    if not low:
        return (0.90, 0.0, 0.0, "chat", "empty_front")

    doc_hits = sum(1 for k in DOCTOR_STRONG if k in low)
    doc_weak = sum(1 for k in DOCTOR_WEAK if k in low)
    if doc_hits >= 1:
        return (
            0.10,
            0.15,
            min(0.95, 0.78 + 0.08 * doc_hits),
            "diagnose",
            "doctor_strong",
        )

    back_build_hits = sum(1 for k in BACK_BUILD if k in low)
    back_res_hits = sum(1 for k in BACK_RESEARCH if k in low)
    back_ver_hits = sum(1 for k in BACK_VERIFY if k in low)
    back_sum_hits = sum(1 for k in BACK_SUMMARIZE if k in low)
    back_total = back_build_hits + back_res_hits + back_ver_hits + back_sum_hits

    front_hits = sum(1 for k in FRONT_DEVICE if k in low)
    if _MATH_RE.search(low) and back_total == 0:
        front_hits += 1

    # Check Doctor weak signals when neither build/research nor device dominates
    if doc_weak >= 2 and back_build_hits == 0 and back_res_hits == 0:
        return (0.15, 0.25, 0.74, "diagnose", "doctor_weak_pair")

    if back_total > 0 and front_hits == 0:
        if back_ver_hits >= 1 and back_build_hits == 0:
            mode = "verify"
        elif back_res_hits >= 1 and back_build_hits == 0:
            mode = "research"
        elif back_sum_hits >= 1 and back_build_hits == 0 and back_res_hits == 0:
            mode = "summarize"
        else:
            mode = "build"
        sb = min(0.95, 0.78 + 0.07 * back_total)
        return (0.12, sb, 0.05 + 0.15 * doc_weak, mode, f"back_{mode}")

    if front_hits > 0 and back_total == 0:
        mode = (
            "chat"
            if any(
                k in low
                for k in (
                    "joke",
                    "hello",
                    "hi ",
                    "hey ",
                    "how are you",
                    "thanks",
                    "thank you",
                )
            )
            else "device"
        )
        sf = min(0.95, 0.80 + 0.07 * front_hits)
        return (sf, 0.08, 0.02, mode, f"front_{mode}")

    # Mixed cues: score both sides explicitly so +10% Front bias arbitrates cleanly
    if back_total > 0 and front_hits > 0:
        sf = min(0.82, 0.55 + 0.10 * front_hits)
        sb = min(0.90, 0.62 + 0.10 * back_total)
        mode = (
            "build" if back_build_hits else ("research" if back_res_hits else "general")
        )
        return (sf, sb, 0.05, mode, "mixed_bias_check")

    # Short general conversation / recall questions stay on Front unless complex length
    words = len(low.split())
    if words <= 14:
        return (0.62, 0.25, 0.05 + 0.10 * doc_weak, "chat", "short_chat_front")
    return (0.42, 0.48, 0.05 + 0.10 * doc_weak, "general", "uncertain_borderline")


class DecisionClassifier:
    """Non-LLM <100 ms routing classifier sitting between user and Front."""

    def __init__(
        self,
        deadline_ms: float = DEADLINE_MS,
        stage2_fn: Stage2Fn | None = None,
        onnx_dir: Path | None = None,
    ) -> None:
        self.deadline_ms = max(10.0, float(deadline_ms))
        self.stage2_fn = stage2_fn
        self.onnx_dir = onnx_dir or ONNX_DIR
        self._onnx_session: Any = None
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sonj4-clf"
        )
        self._latencies: deque[float] = deque(maxlen=60)
        self._total_latency_ms: float = 0.0
        self.stats: dict[str, Any] = {
            "calls": 0,
            "routes": {"front": 0, "back": 0, "doctor": 0},
            "fallbacks": 0,
            "last_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "recent_latencies": [],
            "deadline_ms": self.deadline_ms,
            "front_bias_pct": round(FRONT_BIAS * 100),
            "onnx_ready": self.has_onnx(),
            "stage": "rules-v1",
        }

    def has_onnx(self) -> bool:
        return (self.onnx_dir / "intent.onnx").is_file()

    def classify(self, req: DecisionRequest | str) -> Decision:
        request = (
            req if isinstance(req, DecisionRequest) else DecisionRequest(text=str(req))
        )
        t0 = time.perf_counter()
        sf, sb, sd, mode, reason = _stage1_scores(request.text)

        # Only invoke optional Stage 2 (ONNX / plug-in) when Stage 1 is not already decisive
        # OR when an explicit stage2_fn was provided, strictly within deadline_ms.
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if self.stage2_fn is not None and reason in {
            "uncertain_borderline",
            "short_chat_front",
        }:
            remaining_s = max(0.001, (self.deadline_ms - elapsed_ms) / 1000.0)
            fut = self._pool.submit(self.stage2_fn, request)
            try:
                sf2, sb2, sd2, r2 = fut.result(timeout=remaining_s)
                sf, sb, sd, reason = float(sf2), float(sb2), float(sd2), str(r2)
            except concurrent.futures.TimeoutError:
                ms = (time.perf_counter() - t0) * 1000.0
                dec = Decision(
                    route="front",
                    confidence=0.50,
                    reason_code="deadline_fallback",
                    latency_ms=round(ms, 3),
                    score_front=round(min(1.0, sf + FRONT_BIAS), 4),
                    score_back=round(sb, 4),
                    score_doctor=round(sd, 4),
                    bias_applied=True,
                    fallback=True,
                    suggested_mode="chat",
                )
                self._record(dec)
                return dec
            except Exception as exc:  # noqa: BLE001 - classifier never crashes a turn
                print(
                    f"[classifier] stage2 failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        ms = (time.perf_counter() - t0) * 1000.0
        if ms > self.deadline_ms:
            dec = Decision(
                route="front",
                confidence=0.50,
                reason_code="deadline_fallback",
                latency_ms=round(ms, 3),
                score_front=round(min(1.0, sf + FRONT_BIAS), 4),
                score_back=round(sb, 4),
                score_doctor=round(sd, 4),
                bias_applied=True,
                fallback=True,
                suggested_mode="chat",
            )
            self._record(dec)
            return dec

        dec = apply_front_bias(
            sf,
            sb,
            sd,
            reason_code=reason,
            suggested_mode=mode,
            latency_ms=ms,
            fallback=False,
        )
        self._record(dec)
        return dec

    def _record(self, dec: Decision) -> None:
        self.stats["calls"] += 1
        self.stats["routes"][dec.route] = self.stats["routes"].get(dec.route, 0) + 1
        if dec.fallback:
            self.stats["fallbacks"] += 1
        self._total_latency_ms += float(dec.latency_ms)
        self._latencies.append(float(dec.latency_ms))
        vals = sorted(self._latencies)
        n = len(vals)
        self.stats["last_latency_ms"] = round(float(dec.latency_ms), 3)
        self.stats["avg_latency_ms"] = round(
            self._total_latency_ms / max(1, int(self.stats["calls"])), 3
        )
        self.stats["p50_latency_ms"] = round(vals[n // 2], 3) if n else 0.0
        self.stats["p95_latency_ms"] = (
            round(vals[min(n - 1, int(n * 0.95))], 3) if n else 0.0
        )
        self.stats["max_latency_ms"] = round(
            max(float(self.stats["max_latency_ms"]), float(dec.latency_ms)), 3
        )
        self.stats["recent_latencies"] = [
            round(v, 3) for v in list(self._latencies)[-40:]
        ]

    def to_turn_kind(self, text: str) -> str:
        """Map classifier output to legacy filler/router label ('command'|'chat'|'complex')."""
        dec = self.classify(DecisionRequest(text=text))
        if dec.route in {"back", "doctor"}:
            return "complex"
        return "command" if dec.suggested_mode == "device" else "chat"
