"""Text-to-speech backends for the ME3 laptop voice app.

Windows SAPI is used here because it needs no download and no API key, which
keeps the first end-to-end test about the pipeline rather than about a model.
The class is the laptop stand-in for the TTS MCP server that will run on the
Raspberry Pi (same method names: say / stop / to_wav).
"""

from __future__ import annotations

import contextlib
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import aec
from audio import PA_LOCK


class TtsError(RuntimeError):
    """Raised when a text-to-speech backend cannot be created or used."""


def log(message: str) -> None:
    """Diagnostics go to stderr.

    An MCP server's stdout carries JSON-RPC and nothing else, so a stray print
    there breaks the protocol - which is exactly what this helper prevents.
    """
    print(message, file=sys.stderr)


def _ensure_com() -> None:
    """SAPI is COM, so every thread that touches it needs COM initialised."""
    try:
        import pythoncom
    except ImportError:
        return
    try:
        pythoncom.CoInitialize()
    except pythoncom.com_error as exc:
        # RPC_E_CHANGED_MODE means COM is already up in another apartment.
        log(f"[tts] CoInitialize reported: {exc}")


def _import_sapi():
    """Import win32com lazily so non-Windows hosts can still import this file."""
    try:
        import win32com.client
    except ImportError as exc:
        raise TtsError(
            "pywin32 is required for Windows SAPI (pip install pywin32)"
        ) from exc
    return win32com.client


class SapiTts:
    """Windows SAPI5 voice. speak() is asynchronous so it can be cut off."""

    backend = "sapi"
    _SPF_ASYNC = 1
    _SPF_PURGE_BEFORE_SPEAK = 2
    _SSFM_CREATE_FOR_WRITE = 3

    def __init__(
        self, rate: int = 0, volume: int = 100, voice: str | None = None
    ) -> None:
        _ensure_com()
        client = _import_sapi()
        self._voice = client.Dispatch("SAPI.SpVoice")
        self._default_output = self._voice.AudioOutputStream
        self._voice.Rate = int(rate)
        self._voice.Volume = int(volume)
        if voice:
            self.set_voice(voice)

    @property
    def voice_name(self) -> str:
        return str(self._voice.Voice.GetDescription())

    @property
    def speaking(self) -> bool:
        """SAPI run states: 1 = done, 2 = speaking."""
        status = self._voice.Status
        return int(status.RunningState) == 2

    def list_voices(self) -> list[str]:
        return [str(v.GetDescription()) for v in self._voice.GetVoices()]

    def set_voice(self, name: str) -> None:
        for token in self._voice.GetVoices():
            if name.lower() in str(token.GetDescription()).lower():
                self._voice.Voice = token
                return
        raise TtsError(f"no SAPI voice matching '{name}'")

    def set_rate(self, rate: int) -> None:
        self._voice.Rate = int(rate)

    def say(self, text: str) -> None:
        """Start speaking without blocking, replacing anything already queued."""
        flags = self._SPF_ASYNC | self._SPF_PURGE_BEFORE_SPEAK
        self._voice.Speak(text, flags)

    def stop(self) -> None:
        """Cut speech off immediately (the barge-in path)."""
        self._voice.Speak("", self._SPF_ASYNC | self._SPF_PURGE_BEFORE_SPEAK)

    def wait(self, timeout_s: float = 30.0) -> None:
        self._voice.WaitUntilDone(int(timeout_s * 1000))

    def to_wav(self, text: str, path: Path | str) -> Path:
        """Render text to a WAV file synchronously (default SAPI format)."""
        client = _import_sapi()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stream = client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(out), self._SSFM_CREATE_FOR_WRITE, False)
        self._voice.AudioOutputStream = stream
        try:
            self._voice.Speak(text, 0)
        finally:
            stream.Close()
            self._voice.AudioOutputStream = self._default_output
        # SpFileStream flushes on close, but the file may land a moment later.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if out.exists() and out.stat().st_size > 44:
                break
            time.sleep(0.05)
        return out


NEURAL_PREFIX = "Neural: "

# Kokoro packs 54 voices into one voices.bin, selected by generate(sid=...). These are
# the ones synthesised and heard on this machine (2026-11-08): warm female for Nova,
# natural males for Piattos. Extend after listening, not from the published table alone.
KOKORO_VOICE_IDS = {
    "af_heart": 0,
    "af_bella": 3,
    "am_adam": 11,
    "am_fenrir": 14,
    "am_michael": 17,
}
KOKORO_PERSONA_PREFIX = NEURAL_PREFIX + "kokoro:"
NEURAL_THREADS = (
    2 if platform.machine().lower() in {"aarch64", "arm64", "armv7l"} else 4
)

# Pre-rendered acknowledgements. Short on purpose: they cover the thinking gap, they are
# not the answer. router.FILLER_FOR chooses which key suits the turn.
FILLER_PHRASES = {
    "okay": "Understood.",
    "sure": "Copy.",
    "hmm": "Got it.",
    "one_moment": "On it.",
    "working": "Sure thing.",
}


# --- single-owner playback -------------------------------------------------
# Cross-thread sd.stop()/stream teardown segfaults Windows PortAudio (observed
# 0xC0000005). Rule: only the thread that called sd.play may stop the stream.
# Stops are signaled by bumping a global generation + a per-player event; the
# playback thread notices within 20 ms and halts its OWN stream. A global lock
# serializes playback so two engines never touch the stream at once.
_PLAY_LOCK = threading.Lock()
_PLAY_GEN_LOCK = threading.Lock()
_PLAY_GEN = 0


def _bump_play_generation() -> int:
    global _PLAY_GEN
    with _PLAY_GEN_LOCK:
        _PLAY_GEN += 1
        return _PLAY_GEN


def _current_play_generation() -> int:
    with _PLAY_GEN_LOCK:
        return _PLAY_GEN


def _play_owned(samples, rate: int, generation: int, stop_event) -> None:
    """Play one chunk; the calling thread owns the stream for its whole life."""
    import sounddevice as sd

    with _PLAY_LOCK:
        if generation != _current_play_generation() or stop_event.is_set():
            return
        with PA_LOCK:
            sd.play(samples, rate)
        try:
            while not stop_event.is_set() and generation == _current_play_generation():
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break
                time.sleep(0.02)
        finally:
            with PA_LOCK, contextlib.suppress(Exception):
                sd.stop()


def _speed_from_rate(rate: int) -> float:
    """Map SAPI's -10..10 slider onto a speaking rate, clamped to sane bounds."""
    try:
        steps = int(rate)
    except (TypeError, ValueError):
        steps = 0
    return float(min(2.0, max(0.5, 1.0 + 0.06 * steps)))


def _sentences(text: str) -> list[str]:
    """Split a reply so speech can start before the whole answer is rendered."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _neural_candidates(models_dir: Path | None = None) -> list[dict]:
    """Find runnable offline neural voices under models/.

    Two families work with sherpa-onnx: piper VITS voices (small and fast) and
    Kokoro (larger, slower, closer to commercial quality). Weights that are not
    int8 are preferred on purpose - on this CPU the int8 builds measured about
    six times slower than fp32 for the same voice.
    """
    root = (
        Path(models_dir) if models_dir else Path(__file__).resolve().parent / "models"
    )
    found: list[dict] = []

    for folder in sorted(root.glob("vits-piper*")):
        sibling = root / folder.name.replace("-int8", "")
        if "int8" in folder.name and sibling.is_dir():
            # An fp32 sibling exists, and the int8 build measured ~6x slower here.
            continue
        waves = sorted(folder.glob("*.onnx"))
        tokens = folder / "tokens.txt"
        data_dir = folder / "espeak-ng-data"
        if not waves or not tokens.exists() or not data_dir.exists():
            continue
        weights = next((w for w in waves if "int8" not in w.name.lower()), waves[0])
        found.append(
            {
                "name": folder.name.replace("vits-piper-", "").replace("vits-", ""),
                "kind": "vits",
                "model": weights,
                "tokens": tokens,
                "data_dir": data_dir,
            }
        )

    for folder in sorted(root.glob("kokoro-*")):
        waves = sorted(folder.glob("*.onnx"))
        tokens = folder / "tokens.txt"
        data_dir = folder / "espeak-ng-data"
        voices = folder / "voices.bin"
        if not waves or not tokens.exists() or not voices.exists():
            continue
        weights = next((w for w in waves if "int8" not in w.name.lower()), waves[0])
        found.append(
            {
                "name": folder.name,
                "kind": "kokoro",
                "model": weights,
                "tokens": tokens,
                "data_dir": data_dir,
                "voices": voices,
                "dict_dir": folder / "dict",
                "lang": "en-us",
            }
        )

    return found


def list_neural_voices(models_dir: Path | None = None) -> list[str]:
    """Display names for every offline neural voice on this machine.

    Delegates to NeuralTts.list_voices() when the engine is available, so the
    persona voices ("kokoro:af_heart", ...) come from the same single source the
    engine uses; falls back to plain model names when kokoro cannot instantiate.
    """
    try:
        return NeuralTts(models_dir=models_dir).list_voices()
    except TtsError:
        return [NEURAL_PREFIX + cand["name"] for cand in _neural_candidates(models_dir)]


def list_all_voices(models_dir: Path | None = None) -> list[str]:
    """Every selectable voice: SAPI first, then the offline neural ones."""
    voices: list[str] = []
    try:
        voices.extend(SapiTts().list_voices())
    except TtsError as exc:
        log(f"[tts] SAPI voices unavailable: {exc}")
    voices.extend(list_neural_voices(models_dir))
    return voices


class NeuralTts:
    """Offline neural voice, streamed one sentence at a time.

    Streaming is what keeps the conversation natural: a 2.8 s sentence renders in
    about 0.8 s with fp32 piper on four threads, and the first sentence starts
    playing while the next one is still being synthesised. stop() cuts playback
    immediately, which is what barge-in needs.
    """

    backend = "neural"

    def __init__(
        self,
        voice: str | None = None,
        rate: int = 0,
        threads: int = NEURAL_THREADS,
        models_dir: Path | None = None,
    ) -> None:
        # Stamped when the device is asked to play the first chunk of a reply, so a
        # turn can report when the answer actually became audible.
        self.first_audio_at = 0.0
        self._candidates = _neural_candidates(models_dir)
        if not self._candidates:
            raise TtsError("no offline neural voice found under models/")
        try:
            self._threads = max(1, int(threads))
        except (TypeError, ValueError):
            self._threads = NEURAL_THREADS
        self._current = self._pick(voice)
        self._sid = self._parse_sid(voice)
        self._speed = _speed_from_rate(rate)
        # Built by _load(); typed loosely because sherpa-onnx has no stubs.
        self._engine: Any = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._speaking = False
        self._lock = threading.Lock()
        self._use_espeak_frontend = False
        self._load()
        try:
            import sherpa_onnx

            if (
                str(getattr(sherpa_onnx, "__version__", "")).startswith("2.")
                and (shutil.which("espeak-ng") or shutil.which("espeak"))
                and hasattr(sherpa_onnx, "GenerationConfig")
            ):
                self._use_espeak_frontend = True
        except (ImportError, AttributeError) as exc:
            log(f"[tts] espeak frontend check skipped: {exc}")

    @staticmethod
    def _parse_sid(voice: str | None) -> int:
        """'Neural: kokoro:af_heart' -> 0; anything else keeps the first voice."""
        wanted = (voice or "").removeprefix(NEURAL_PREFIX)
        if wanted.startswith("kokoro:"):
            return KOKORO_VOICE_IDS.get(wanted.split(":", 1)[1].strip().lower(), 0)
        return 0

    def _pick(self, voice: str | None) -> dict:
        """Match a display name loosely, so 'lessac' finds the lessac voice."""
        if voice:
            wanted = voice.removeprefix(NEURAL_PREFIX).lower()
            # A persona name ("kokoro:am_michael") never matches a model's own name
            # as a substring, so it must select the kokoro engine explicitly.
            if wanted.startswith("kokoro:"):
                for cand in self._candidates:
                    if cand["kind"] == "kokoro":
                        return cand
                return self._candidates[0]
            for cand in self._candidates:
                if wanted and wanted in cand["name"].lower():
                    return cand
        return self._candidates[0]

    def _load(self) -> None:
        import sherpa_onnx

        cand = self._current
        if cand["kind"] == "kokoro":
            kokoro_kwargs: dict[str, Any] = {
                "model": str(cand["model"]),
                "voices": str(cand["voices"]),
                "tokens": str(cand["tokens"]),
                "data_dir": str(cand["data_dir"]),
                "dict_dir": str(cand["dict_dir"]) if cand.get("dict_dir") else "",
                "lang": str(cand.get("lang") or "en-us"),
            }
            if not (shutil.which("espeak-ng") or shutil.which("espeak")):
                lexs = [
                    str(cand["model"].parent / fn)
                    for fn in ("lexicon-us-en.txt", "lexicon-zh.txt")
                    if (cand["model"].parent / fn).is_file()
                ]
                if lexs:
                    kokoro_kwargs["lexicon"] = ",".join(lexs)
            model = sherpa_onnx.OfflineTtsKokoroModelConfig(**kokoro_kwargs)
            engine_kwargs = {"kokoro": model}
        else:
            model = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(cand["model"]),
                tokens=str(cand["tokens"]),
                data_dir=str(cand["data_dir"]),
            )
            engine_kwargs = {"vits": model}

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                **engine_kwargs, num_threads=self._threads, provider="cpu"
            ),
            max_num_sentences=1,
        )
        self._engine = sherpa_onnx.OfflineTts(config)
        log(
            f"[tts] neural voice ready: {self._current['name']} ({self._threads} threads)"
        )

    @property
    def voice_name(self) -> str:
        return NEURAL_PREFIX + self._current["name"]

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def list_voices(self) -> list[str]:
        names = [NEURAL_PREFIX + cand["name"] for cand in self._candidates]
        if any(cand["kind"] == "kokoro" for cand in self._candidates):
            names += [KOKORO_PERSONA_PREFIX + key for key in KOKORO_VOICE_IDS]
        return names

    def set_voice(self, name: str) -> None:
        picked = self._pick(name)
        # The speaker id lives on the engine, not the candidate: switching between two
        # kokoro personas picks the SAME candidate, so the early return below left the
        # previous speaker id in place and Piattos spoke with Nova's id. Audit 7853a104.
        self._sid = self._parse_sid(name)
        if picked is self._current:
            return
        self.stop()
        self._current = picked
        self._load()

    def set_rate(self, rate: int) -> None:
        self._speed = _speed_from_rate(rate)

    def say(self, text: str) -> None:
        """Start speaking without blocking, replacing anything already playing."""
        self.stop()
        self.first_audio_at = 0.0
        if not text.strip():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._stream, args=(text,), daemon=True)
        self._worker.start()

    @staticmethod
    def _espeak_codepoints(sentence: str) -> list[list[int]] | None:
        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak_bin:
            return None
        try:
            ipa = subprocess.check_output(
                [espeak_bin, "-q", "-v", "en-us", "--ipa", sentence],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        clean = " ".join(
            line.strip() for line in ipa.splitlines() if line.strip()
        ).replace("\u200d", "")
        if not clean:
            return None
        return [[ord(ch) for ch in clean]]

    def _render_with_espeak(self, sentence: str):
        import sherpa_onnx

        gc_cls = getattr(sherpa_onnx, "GenerationConfig", None)
        if gc_cls is None:
            return None
        codepoints = self._espeak_codepoints(sentence)
        if not codepoints:
            return None
        gc = gc_cls()
        gc.sid = self._sid
        gc.speed = self._speed
        gc.phoneme_codepoints = codepoints
        return self._engine.generate(sentence, config=gc)

    def _render(self, sentence: str | None):
        """Synthesise one sentence into (samples, sample_rate)."""
        if not sentence:
            return None
        if self._use_espeak_frontend:
            result = self._render_with_espeak(sentence)
            if result is None:
                return None
            samples = np.asarray(result.samples, dtype=np.float32)
            return (samples, int(result.sample_rate)) if samples.size > 0 else None
        result = self._engine.generate(sentence, sid=self._sid, speed=self._speed)
        samples = np.asarray(result.samples, dtype=np.float32)
        if samples.size == 0:
            # sherpa-onnx 2.0+ omits bundled GPL espeak-ng; use system espeak-ng IPA.
            fb = self._render_with_espeak(sentence)
            if fb is not None:
                fb_samples = np.asarray(fb.samples, dtype=np.float32)
                if fb_samples.size > 0:
                    self._use_espeak_frontend = True
                    return fb_samples, int(fb.sample_rate)
            return None
        return samples, int(result.sample_rate)

    def _stream(self, text: str) -> None:
        """C: play sentence N while a producer renders N+1 — no gap between sentences.

        _play_owned blocks until the chunk finishes, so rendering the next sentence
        after it left a 280-600 ms silence at every sentence boundary. A bounded
        producer queue renders ahead while the current chunk plays.
        """
        with self._lock:
            self._speaking = True
        try:
            sentences = _sentences(text) or [text]
            rendered: queue.Queue = queue.Queue(maxsize=2)

            def _produce() -> None:
                for sentence in sentences:
                    if self._stop.is_set():
                        break
                    rendered.put(self._render(sentence))
                rendered.put(None)

            producer = threading.Thread(target=_produce, daemon=True, name="tts-render")
            producer.start()
            index = 0
            while not self._stop.is_set():
                try:
                    audio = rendered.get(timeout=0.05)
                except queue.Empty:
                    continue
                if audio is None:
                    break
                aec.feed_reference(audio[0], audio[1])
                if index == 0:
                    self.first_audio_at = time.perf_counter()
                _play_owned(audio[0], audio[1], _current_play_generation(), self._stop)
                index += 1
        except Exception as exc:  # noqa: BLE001 - a broken voice must not kill the turn
            log(f"[tts] neural playback failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        """Cut speech off immediately (the barge-in path).

        Never calls sd.stop here: the playback thread halts its own stream when
        it sees the bumped generation (cross-thread sd.stop segfaults PortAudio).
        """
        self._stop.set()
        _bump_play_generation()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=0.2)
        with self._lock:
            self._speaking = False

    def wait(self, timeout_s: float = 30.0) -> None:
        # ident is only assigned once a thread actually starts; joining an
        # un-started worker raised RuntimeError and killed the whole engine
        # mid-conversation (seen live after TTS had been disabled).
        worker = self._worker
        if worker and worker.ident is not None:
            worker.join(timeout=timeout_s)

    def to_wav(self, text: str, path: Path | str) -> Path:
        """Render text to a WAV file synchronously."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        chunks = []
        rate = 22050
        for sentence in _sentences(text) or [text]:
            rendered = self._render(sentence)
            if rendered:
                chunks.append(rendered[0])
                rate = rendered[1]
        samples = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        sf.write(str(out), samples, rate)
        return out


class FillerBank:
    """Pre-rendered acknowledgements that play almost instantly.

    Synthesising a sentence costs 280-600 ms; playing a cached clip costs about 20-40 ms.
    Saying "Okay." immediately is how human conversation covers thinking time, and it is
    what makes the assistant feel responsive while the real answer is still being made.

    Clips are rendered with whatever voice is active, so they match the assistant's voice;
    warm() is safe to call again after a voice change.
    """

    def __init__(
        self,
        voice,
        phrases: dict[str, str] | None = None,
        cache_dir: Path | None = None,
        use_stream: bool | None = None,
    ) -> None:
        self.voice = voice
        self.phrases = dict(phrases or FILLER_PHRASES)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else (Path(__file__).resolve().parent / "out" / "fillers")
        )
        self._clips: dict[str, tuple[np.ndarray, int]] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._playing = False
        self._fill_stop = threading.Event()
        # One output stream is kept open: opening a WASAPI stream per clip costs
        # 190-600 ms, which dominated the acknowledgement's whole latency budget.
        # The persistent-stream fast path is OPT-IN. It saves ~300 ms of device setup, but
        # Windows MME wedges a stream that is cut off mid-clip, and closing such a stream from
        # another thread while its writer is active segfaulted the process. Until that is
        # solved properly, the default is the boring, stable per-call playback.
        if use_stream is None:
            use_stream = os.environ.get("ME3_FILLER_STREAM", "") == "1"
        self.use_stream = bool(use_stream)
        self._stream = None
        self._rate = 0
        self._needs_restart = False
        self._stream_failures = 0
        # started_at is stamped inside the playback thread, immediately before the
        # device is asked to play. Returning from play() only proves a thread started;
        # an earlier version measured that and over-reported the latency win.
        self.started_at = 0.0
        self.started_ms = 0.0
        self.finished_at = 0.0

    def clip_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.wav"

    def warm(self) -> int:
        """Render any missing clip. Returns how many were newly rendered."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        rendered = 0
        for key, phrase in self.phrases.items():
            path = self.clip_path(key)
            if path.exists() and path.stat().st_size > 1024:
                continue
            try:
                self.voice.to_wav(phrase, path)
                rendered += 1
            except Exception as exc:  # noqa: BLE001 - a missing clip is not fatal
                log(
                    f"[tts] filler '{key}' could not be rendered: {type(exc).__name__}: {exc}"
                )
        return rendered

    def load(self) -> int:
        """Read every clip into memory once, so playback never touches the disk."""
        loaded = 0
        with self._lock:
            for key in self.phrases:
                if key in self._clips:
                    continue
                path = self.clip_path(key)
                if not path.exists():
                    continue
                try:
                    samples, rate = sf.read(str(path), dtype="float32")
                except Exception as exc:  # noqa: BLE001
                    log(f"[tts] filler '{key}' unreadable: {type(exc).__name__}: {exc}")
                    continue
                self._clips[key] = (np.asarray(samples, dtype=np.float32), int(rate))
                loaded += 1
        return loaded

    def ready(self, key: str) -> bool:
        with self._lock:
            return key in self._clips

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    def _ensure_stream(self, rate: int):
        """Return a started stream, or None to use per-call playback.

        Discipline that matters: the stream may only be started and written from the playback
        thread, and it is NEVER closed from elsewhere - a close racing an in-flight write is
        what crashed this process. A stream that had to be aborted is restarted on the next
        play, from that play's own thread.
        """
        if not self.use_stream or self._stream_failures >= 2:
            return None
        """Return an open output stream, opening it once and keeping it.

        sounddevice.play() opens and closes a stream per call: measured 189-595 ms of
        setup on this machine, against ~0 ms when the stream already exists. Returns
        None when no stream can be opened, so playback falls back to sounddevice.play.
        """
        with self._lock:
            stream = self._stream
            needs_restart = self._needs_restart
            if stream is not None and self._rate == rate:
                if not needs_restart:
                    return stream
                try:
                    stream.start()
                    self._needs_restart = False
                    return stream
                except Exception as exc:  # noqa: BLE001 - restart can fail on MME
                    self._stream_failures += 1
                    log(
                        f"[tts] filler stream could not restart ({type(exc).__name__}); "
                        f"using per-call playback"
                    )
                    return None
            old = self._stream
            self._stream = None
        if old is not None:
            self._stop_stream(old)
        try:
            import sounddevice as sd

            stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
            stream.start()
        except Exception as exc:  # noqa: BLE001 - fall back, never break a turn
            # Count open failures too: review found the breaker ignored them, so a host that
            # cannot open the stream retried (and logged) on every single clip.
            self._stream_failures += 1
            log(
                f"[tts] filler stream unavailable ({type(exc).__name__}: {exc}); "
                f"using per-call playback (failure {self._stream_failures})"
            )
            return None
        with self._lock:
            self._stream = stream
            self._rate = rate
        return stream

    def play(self, key: str, on_done: Callable[[], None] | None = None) -> bool:
        """Start a clip without blocking; replaces anything already playing.

        on_done runs after the clip has finished sounding, which is what lets the caller
        open the echo window for the acknowledgement exactly as it does for a reply.
        """
        if not self.ready(key):
            self.load()
        with self._lock:
            clip = self._clips.get(key)
        if clip is None:
            return False
        samples, rate = clip
        requested = time.perf_counter()
        stream = self._ensure_stream(rate)

        def run() -> None:
            import sounddevice as sd

            aec.feed_reference(samples, rate)
            with self._lock:
                self._playing = True
            try:
                if stream is not None:
                    self.started_at = time.perf_counter()
                    self.started_ms = (self.started_at - requested) * 1000.0
                    try:
                        stream.write(
                            np.asarray(samples, dtype=np.float32).reshape(-1, 1)
                        )
                    except Exception as exc:  # noqa: BLE001 - the stream can wedge
                        # Windows MME refuses to restart a stream while data is still playing, so
                        # a stream can wedge after a cut-off. Drop it and play this clip simply.
                        self._stream_failures += 1
                        log(
                            f"[tts] filler stream unusable ({type(exc).__name__}); "
                            f"falling back for this clip (failure {self._stream_failures})"
                        )
                        self._mark_stream_unusable()
                        sd.play(samples, rate)
                else:
                    self.started_at = time.perf_counter()
                    self.started_ms = (self.started_at - requested) * 1000.0
                    _play_owned(
                        samples, rate, _current_play_generation(), self._fill_stop
                    )
            except Exception as exc:  # noqa: BLE001 - audio must never break a turn
                log(f"[tts] filler playback failed: {type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._playing = False
                self.finished_at = time.perf_counter()
                if on_done is not None:
                    try:
                        on_done()
                    except Exception as exc:  # noqa: BLE001
                        log(f"[tts] filler on_done failed: {type(exc).__name__}: {exc}")

        self._fill_stop.clear()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self.last_play_ms = (time.perf_counter() - requested) * 1000.0
        return True

    @staticmethod
    def _stop_stream(stream) -> None:
        """Stop a stream without closing it: closing can race an in-flight write."""
        with contextlib.suppress(Exception):
            stream.stop()

    def _mark_stream_unusable(self) -> None:
        """Give up on the fast path for this session, without touching the handle."""
        with self._lock:
            self._stream = None
            self._rate = 0
            self._needs_restart = False

    def stop(self) -> None:
        """Cut the filler short - the real answer is about to start."""
        try:
            with self._lock:
                stream = self._stream
                playing = self._playing
            if stream is not None and playing:
                stream.abort()
                # Do NOT restart here: that happens on the next play, from the playback thread.
                with self._lock:
                    self._needs_restart = True
            elif self.playing:
                self._fill_stop.set()
                _bump_play_generation()
        except Exception as exc:  # noqa: BLE001 - stopping must never raise
            # Observed on Windows MME: aborting while data plays fails with -9999.
            self._stream_failures += 1
            log(
                f"[tts] filler stop failed ({type(exc).__name__}); next play restarts the stream "
                f"(failure {self._stream_failures})"
            )
            with self._lock:
                self._needs_restart = True
        worker = self._thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.3)
        with self._lock:
            self._playing = False


# Session character usage per cloud provider, for credit analytics.
CLOUD_CHARS: dict[str, int] = {"elevenlabs": 0, "openai": 0}


def cloud_chars() -> dict[str, int]:
    return dict(CLOUD_CHARS)


class CloudTts:
    """Optional cloud TTS (OpenAI / ElevenLabs flash), key-gated, local default.

    Same interface as the local engines; playback goes through _play_owned so
    the single-owner stream rule holds for cloud audio too.
    """

    backend = "cloud"

    def __init__(self, provider: str, api_key: str, voice: str = "") -> None:
        self.provider = str(provider)
        self.api_key = str(api_key)
        self.voice = voice or (
            "alloy" if provider == "openai" else "21m00Tcm4TlvDq8ikWAM"
        )
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._speaking = False
        self._lock = threading.Lock()
        self.first_audio_at = 0.0
        self._speed = 1.0

    @property
    def voice_name(self) -> str:
        return f"cloud:{self.provider}:{self.voice}"

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def list_voices(self) -> list[str]:
        return [self.voice_name]

    def set_voice(self, name: str) -> None:
        if name:
            self.voice = name.split(":")[-1]

    def set_rate(self, rate: int) -> None:
        self._speed = _speed_from_rate(rate)

    def _fetch(self, text: str) -> tuple[np.ndarray, int]:
        import io

        import requests

        CLOUD_CHARS[self.provider] = CLOUD_CHARS.get(self.provider, 0) + len(text)

        if self.provider == "openai":
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4o-mini-tts",
                    "voice": self.voice,
                    "input": text,
                    "response_format": "wav",
                    "speed": self._speed,
                },
                timeout=30,
            )
            resp.raise_for_status()
            samples, rate = sf.read(io.BytesIO(resp.content), dtype="float32")
            return np.asarray(samples, dtype=np.float32).reshape(-1), int(rate)
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}",
            params={"output_format": "pcm_16000"},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_flash_v2_5"},
            timeout=30,
        )
        resp.raise_for_status()
        pcm = np.frombuffer(resp.content, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, 16000

    def say(self, text: str) -> None:
        self.stop()
        self.first_audio_at = 0.0
        if not text.strip():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._stream, args=(text,), daemon=True)
        self._worker.start()

    def _stream(self, text: str) -> None:
        import requests

        with self._lock:
            self._speaking = True
        try:
            for sentence in _sentences(text) or [text]:
                if self._stop.is_set():
                    break
                try:
                    samples, rate = self._fetch(sentence)
                except (requests.RequestException, ValueError, OSError) as exc:
                    log(f"[tts] cloud fetch failed: {type(exc).__name__}: {exc}")
                    break
                if samples.size == 0:
                    continue
                aec.feed_reference(samples, rate)
                if self.first_audio_at <= 0.0:
                    self.first_audio_at = time.perf_counter()
                _play_owned(samples, rate, _current_play_generation(), self._stop)
        finally:
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        self._stop.set()
        _bump_play_generation()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=0.2)
        with self._lock:
            self._speaking = False

    def wait(self, timeout_s: float = 30.0) -> None:
        worker = self._worker
        if worker and worker.ident is not None:
            worker.join(timeout=timeout_s)

    def to_wav(self, text: str, path: Path | str) -> Path:
        samples, rate = self._fetch(text)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), samples, rate)
        return out


class SimulatedTts:
    """Speaks by waiting instead of playing audio.

    Duration is estimated from the text at a normal speaking rate, and stop()
    cuts the wait short. That is what makes barge-in measurable on a machine with
    no speaker attached (a headless Pi, or CI) while keeping the timing of the
    conversation realistic.
    """

    backend = "simulate"
    WORDS_PER_SECOND = 2.6
    STOP_GRACE_S = 0.15

    def __init__(self, rate: int = 0) -> None:
        self._stop = threading.Event()
        self._deadline = 0.0
        self._lock = threading.Lock()
        self._rate = int(rate)

    @property
    def voice_name(self) -> str:
        return "(simulated voice - no audio device needed)"

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._deadline > time.monotonic() and not self._stop.is_set()

    def list_voices(self) -> list[str]:
        return ["(simulated voice)"]

    def set_voice(self, name: str) -> None:
        del name

    def set_rate(self, rate: int) -> None:
        self._rate = int(rate)

    def duration_of(self, text: str) -> float:
        """Estimated spoken duration, so callers can size a silent WAV."""
        words = max(1, len(text.split()))
        speed = self.WORDS_PER_SECOND * (1 + self._rate * 0.06)
        return max(0.4, words / max(0.5, speed)) + self.STOP_GRACE_S

    def say(self, text: str) -> None:
        """Start 'speaking': record when it would finish."""
        with self._lock:
            self._stop.clear()
            self._deadline = time.monotonic() + self.duration_of(text)

    def stop(self) -> None:
        """Barge-in: end the wait immediately."""
        self._stop.set()
        with self._lock:
            self._deadline = 0.0

    def wait(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._stop.is_set() and time.monotonic() < self._deadline:
            if time.monotonic() > deadline:
                return
            time.sleep(0.02)

    def to_wav(self, text: str, path: Path | str) -> Path:
        """Write silence of the right length, so file-based paths still work."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(out),
            np.zeros(int(16000 * self.duration_of(text)), dtype="float32"),
            16000,
        )
        return out


class NullTts:
    """Used when TTS is disabled; keeps the pipeline shape identical."""

    backend = "none"

    @property
    def voice_name(self) -> str:
        return "(tts off)"

    @property
    def speaking(self) -> bool:
        return False

    def list_voices(self) -> list[str]:
        return []

    def set_voice(self, name: str) -> None:
        raise TtsError("TTS is disabled")

    def set_rate(self, rate: int) -> None:
        return None

    def say(self, text: str) -> None:
        print(f"[tts off] {text}")

    def stop(self) -> None:
        return None

    def wait(self, timeout_s: float = 30.0) -> None:
        return None

    def to_wav(self, text: str, path: Path | str) -> Path:
        raise TtsError("TTS is disabled")
