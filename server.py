"""Standalone laptop app for the ME3 full-duplex voice agent.

One command starts a local web UI that shows the conversation, every tool call,
and the per-stage latency. Audio runs on the laptop itself (mic and speakers);
the browser is only a monitor and control panel, which is the same split the
Raspberry Pi deployment will use.

Run:  python me3_laptop/server.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import difflib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from aec import EchoSuppressor, NoiseSuppressor
from agent import Agent, ToolBoxLike
from asr import AsrError, CloudAsr, create_asr, discover_models
from audio import (
    AudioError,
    MicStream,
    Player,
    ScriptedStream,
    can_open_input,
    default_devices,
    input_devices,
    output_devices,
    probe_input_blocks,
)
from classifier import Decision, DecisionClassifier, DecisionRequest
from doctor import AGENTS as DOCTOR_AGENTS
from doctor import CATALOGUE as DOCTOR_REPAIRS
from doctor import SCOPES as DOCTOR_SCOPES
from doctor import Doctor, engine_health, engine_processes
from doctor import escalate as doctor_escalate
from errors import BUS
from llm import (
    LlmError,
    create_llm,
    find_elevenlabs_key,
    find_openrouter_key,
    list_ollama_models,
    list_openrouter_models,
    loaded_ollama_models,
    unload_ollama_model,
)
from mcp_client import McpClient
from pi_rpc import PiPane, PiRpcError, list_pi_models
from router import DEFAULT_ROUTER_MODEL, Router
from syshealth import (
    CONFIRM_PHRASE,
    DoctorGate,
    app_introspection,
    feature_distribution,
    inspect_architecture,
    is_asking,
    performance_review,
    settings_catalog,
    system_health,
)
from tasks import TaskManager
from toolbox import ToolBox
from tts import (
    NEURAL_PREFIX,
    CloudTts,
    FillerBank,
    NeuralTts,
    NullTts,
    SapiTts,
    SimulatedTts,
    TtsError,
    list_all_voices,
)
from vad import SileroVad, VadError
from voiceprint import Voiceprint

BASE = Path(__file__).resolve().parent
BACK_POOL: tuple[str, ...] = ("back-1", "back-2", "back-3", "back-4", "back-5")
BACK_ALIASES = {"back", "researcher", "critic", "summarizer"}
MODELS_DIR = BASE / "models"
STATIC_DIR = BASE / "static"
CONFIG_PATH = BASE / "config.json"
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
OUT_DIR = BASE / "out"
# S0: one engine, one voice. The pid/lock pair is what stops a second engine
# from opening a second audio device and fighting the first for the speakers.
ENGINE_PID = OUT_DIR / "engine.pid"
ENGINE_LOCK = OUT_DIR / "engine.lock"

COPILOT_MODELS = [
    "github-copilot/gpt-6-luna",
    "github-copilot/gpt-6-astra",
    "github-copilot/gpt-6-sol",
    "github-copilot/claude-sonnet-5",
]

OPENROUTER_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "openai/gpt-6-luna",
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-27b",
]

# Tool calling quality matters: 3B models frequently ignore tool results, so the
# default is the smallest local model that was verified to honour them.
DEFAULT_MODEL = "rafw007/qwen35-claude-coder:9b"
# Free, tool-verified OpenRouter model: costs nothing against a small credit cap.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"
MODELS_CACHE_S = 120.0

# Barge-in tuning. Without these guards the assistant hears its own speaker and
# cuts itself off mid-sentence, which looks like the app is broken.
MIN_BARGE_RMS = 0.02
# After the assistant stops speaking its own tail is still in the mic buffer.
ECHO_TAIL_S = 0.8
ECHO_SIMILARITY = 0.6
# A robotic local voice can be badly mis-heard (the Pi's espeak turned "bedroom"
# into "memory"), so a fuzzier match is allowed - but only for audio that arrives
# right after the agent stopped speaking.
# The echo arrives while the agent speaks, but its transcript only exists once
# the echo ends, which can be a whole sentence later - so the window has to cover
# a played-back sentence, not a moment.
ECHO_WINDOW_S = 8.0
FUZZY_ECHO_RATIO = 0.45
INTERRUPT_GRACE_MS = 500.0
DEFAULT_INTERRUPT_MIN_MS = 350.0
LOG_PATH = BASE / "out" / "app.log"
# Set once teardown has finished, so a second exit path does not re-run it.
_SHUTDOWN_DONE = threading.Event()


def process_memory_mb() -> float | None:
    """Resident memory of this process, so the UI can show what the app holds."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Without explicit argtypes ctypes passes the handle as a C int and
            # the call fails silently - which is how this read "unavailable".
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return counters.WorkingSetSize / (1024 * 1024)
            log_event(
                f"GetProcessMemoryInfo failed (win error {ctypes.get_last_error()})"
            )
            return None
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, AttributeError) as exc:
        log_event(f"could not read process memory ({exc})")
        return None


AudioSource = MicStream | ScriptedStream


def log_event(message: str) -> None:
    """Append a timestamped line to out/app.log and to the console."""
    line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # A cp1252 console cannot print emoji; never let logging break a turn.
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print(f"could not write {LOG_PATH.name}: {exc}", flush=True)


def clamp_barge_rms(value: object, default: float = 0.06) -> float:
    """Keep the interrupt threshold above the floor where it self-triggers.

    A config file can hold anything; below this floor the assistant hears its own
    speaker and cuts itself off, so the value is clamped on load.
    """
    try:
        return max(MIN_BARGE_RMS, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


ASR_MAX_AUDIO_S = 25.0


# W1: only deliberate STOP words cut a reply. Wake words (nova/piattos) are NOT
# cut triggers — saying "Nova" to continue must not slice the previous answer.
BARGE_TRIGGER_WORDS = (
    "stop",
    "cancel",
    "quiet",
    "enough",
    "shush",
    "wait",
    "pause",
    "hold on",
)

# Re-check window: the content check needs ASR, so one continuous run is judged
# at most once per this many seconds.
BARGE_REJUDGE_S = 0.7

# ASR input cap. Whisper handles at most 30 s, and sherpa's circular buffer aborts the
# process on oversized pushes (seen in server_err.txt: "circular-buffer.cc:Push:110
# Overflow" and "Only waves less than 30 seconds are supported"). 25 s leaves headroom.
ASR_MAX_AUDIO_S = 25.0


def _heard_trigger(heard: str, words: tuple[str, ...] = ()) -> str | None:
    """Which stop word (if any) the transcript contains, word-boundary exact."""
    for word in words or BARGE_TRIGGER_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", heard, flags=re.IGNORECASE):
            return word
    return None


def is_echo(
    transcript: str,
    spoken: str,
    threshold: float = ECHO_SIMILARITY,
    recent: bool = False,
) -> bool:
    """True when a transcript looks like the assistant's own recent speech.

    Without this the app answers itself: it speaks, the microphone hears it, the
    text returns as user input, and the conversation loops forever.
    """
    words = {
        word for word in re.findall(r"[a-z0-9']+", transcript.lower()) if len(word) > 2
    }
    if len(words) < 2:
        return False
    spoken_words = set(re.findall(r"[a-z0-9']+", spoken.lower()))
    if (len(words & spoken_words) / len(words)) >= threshold:
        return True
    if not recent:
        return False
    # A robotic local voice can be badly mis-heard (the Pi's espeak turned
    # "bedroom" into "memory"), so a fuzzy match counts - but only for audio
    # that arrives right after the agent finished speaking.
    ratio = difflib.SequenceMatcher(None, transcript.lower(), spoken.lower()).ratio()
    return ratio >= FUZZY_ECHO_RATIO


def repair_provider_model(settings: Settings) -> Settings:
    """Fix a stored model name that belongs to the other provider.

    A provider switch used to pair the new provider with the old provider's model
    name. Ollama answers an unknown id with HTTP 400, and because that error used
    to be spoken aloud the microphone echoed it back as input - an endless loop.
    """
    if settings.provider != "ollama":
        return settings
    try:
        available = list_ollama_models()
    except LlmError as exc:
        log_event(f"could not verify the configured model ({exc})")
        return settings
    if settings.model in available:
        return settings
    fallback = (
        DEFAULT_MODEL
        if DEFAULT_MODEL in available
        else (available[0] if available else "")
    )
    if fallback:
        log_event(
            f"model '{settings.model}' is not installed in Ollama - using '{fallback}'"
        )
        settings.model = fallback
    return settings


def should_interrupt(
    rms: float,
    segment_ms: float,
    playback_ms: float,
    threshold: float,
    min_segment_ms: float = DEFAULT_INTERRUPT_MIN_MS,
    grace_ms: float = INTERRUPT_GRACE_MS,
) -> bool:
    """Is this audio the user barging in, or the speaker bleeding into the mic?

    Three guards, in order: a grace window after playback begins (echo is not
    meaningful instantly), a minimum speech length (coughs and clicks are not
    interruptions), then the loudness threshold.
    """
    if playback_ms < grace_ms:
        return False
    if segment_ms < min_segment_ms:
        return False
    return rms >= threshold


AGENTS_PATH = BASE / "agents.json"
ROOT_AGENTS_PATH = BASE.parent / "agents.json"


class _SentenceSplitter:
    """Groups streamed model pieces into complete sentences (M2.2).

    A sentence boundary is where speech can safely start: the rest of the reply
    keeps generating while this sentence is already being spoken.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        self._buffer += str(piece)
        ready: list[str] = []
        while True:
            match = re.search(r"([.!?…])(\s|$)", self._buffer)
            if match is None:
                # M2.6: a long sentence takes seconds to close, and speech cannot
                # start before a boundary. Clause punctuation (, ; :) is a safe
                # earlier boundary once the leading part is long enough to be
                # worth saying on its own.
                head = re.match(r"^(.{36,}?[,;:])(\s)", self._buffer)
                if head is not None and len(head.group(1).split()) >= 8:
                    chunk = self._buffer[: head.end(1)].strip()
                    self._buffer = self._buffer[head.end(0) :]
                    if len(chunk) > 1:
                        ready.append(chunk)
                    continue
                break
            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]
            if len(sentence) > 1:
                ready.append(sentence)
        return ready

    def flush(self) -> str:
        rest, self._buffer = self._buffer.strip(), ""
        return rest


def sanitize_for_speech(text: str) -> str:
    """Strip markdown and URLs so TTS never reads formatting or raw links aloud."""
    s = str(text or "")
    # Markdown links [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", s)
    # Bare URLs -> omitted so TTS stays natural
    s = re.sub(r"\bhttps?://[^\s)\]>\"']+", "", s)
    s = re.sub(r"\bwww\.[^\s)\]>\"']+", "", s)
    # Code fences / inline backticks / headings / bold / italic / bullets
    s = re.sub(r"```.*?```", " ", s, flags=re.DOTALL)
    s = re.sub(r"[`*_~]+", "", s)
    s = re.sub(r"(^|\s)#{1,6}\s+", r"\1", s)
    s = re.sub(r"(^|\n)\s*[-*•]\s+", r"\1", s)
    return " ".join(s.split()).strip()


def _clip_speech(text: str, cap: int) -> str:
    """At most `cap` characters, cut on a sentence boundary (0 = no cap)."""
    text = " ".join(str(text).split())
    if not cap or len(text) <= cap:
        return text
    cut = text[:cap]
    boundary = max(
        cut.rfind(". "),
        cut.rfind("! "),
        cut.rfind("? "),
        cut.rfind("; "),
        cut.rfind(", "),
    )
    if boundary >= 40:
        cut = cut[: boundary + 1]
    return cut.strip()


def _wants_long_form(text: str) -> bool:
    """True when the owner explicitly asks for long-form spoken content (a story,
    poem, essay, ...). The short-reply speech cap must not truncate these."""
    low = str(text).lower()
    forms = (
        "story",
        "tale",
        "poem",
        "poetry",
        "essay",
        "song",
        "lyrics",
        "fable",
        "legend",
        "bedtime",
        "narrate",
        "compose",
        "paragraph",
        "sentences",
        "in detail",
        "full version",
        "long version",
        "longer version",
        "write me a letter",
        "tell me about it in",
    )
    return any(k in low for k in forms)


class SpeechBudget:
    """R12: keep the spoken part short; the rest stays on screen.

    The transcript always records the full reply - only the audio is trimmed, so
    nothing is lost, it just is not read aloud. Without this a long answer takes
    a minute to say and reads as a freeze.
    """

    def __init__(self, cap_chars: int = 0) -> None:
        self.cap = max(0, int(cap_chars or 0))
        self.spoken = 0
        self.dropped = 0

    def take(self, text: str) -> str:
        text = " ".join(str(text).split())
        if not self.cap or not text:
            return text
        room = self.cap - self.spoken
        if room <= 0:
            self.dropped += len(text)
            return ""
        piece = _clip_speech(text, room)
        self.spoken += len(piece)
        self.dropped += max(0, len(text) - len(piece))
        return piece


class SpeechPump:
    """Speaks streamed sentences on its own thread (M2.2, D-2).

    push() never blocks the model-stream reader, so generation and speech
    overlap - the first sentence is audible while the rest is still being
    written. The pump records what was actually spoken (for the honest
    transcript after a barge-in), per-sentence timing (M2.3) and the
    first-audio stamp (the streamed TTFA).
    """

    def __init__(
        self,
        voice,
        on_first: Callable[[], None] | None,
        started: float,
        on_spoken: Callable[[str], None] | None = None,
    ) -> None:
        self._voice = voice
        self._on_first = on_first
        self._on_spoken = on_spoken
        self._started = started
        self._queue: queue.Queue = queue.Queue()
        self._stopped = threading.Event()
        self._failed = False
        self._lock = threading.Lock()
        self._played: list[str] = []
        self._marks: list[dict[str, Any]] = []
        self._spoken_ms = 0.0
        self._first_say_at = 0.0
        self._first_audio_at = 0.0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ producer side
    def push(self, sentence: str) -> None:
        text = sanitize_for_speech(sentence)
        if text and not self._stopped.is_set():
            self._queue.put(text)

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut the voice instantly."""
        self._stopped.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with contextlib.suppress(TtsError):
            self._voice.stop()

    def finish(self, timeout_s: float = 120.0) -> None:
        """Wait for the spoken queue to drain (bounded)."""
        deadline = time.perf_counter() + timeout_s
        while not self._stopped.is_set() and time.perf_counter() < deadline:
            if self._queue.empty():
                break
            time.sleep(0.05)
        if not self._stopped.is_set():
            with contextlib.suppress(TtsError):
                self._voice.wait(timeout_s=60)

    # -------------------------------------------------------------- reader side
    def played_text(self) -> str:
        with self._lock:
            return " ".join(self._played).strip()

    def failed(self) -> bool:
        return self._failed

    def spoken_ms(self) -> float:
        with self._lock:
            return self._spoken_ms

    def first_audio_ms(self) -> float:
        stamp = self._first_audio_at or self._first_say_at
        if stamp <= 0.0 or stamp < self._started:
            return 0.0
        return (stamp - self._started) * 1000.0

    def sentence_marks(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._marks)

    def word_marks(self) -> list[dict[str, Any]]:
        """Approximate word offsets (M2.3): words are placed linearly inside their
        sentence's measured span, since the engines report sentence timing, not
        per-word boundaries."""
        words: list[dict[str, Any]] = []
        for mark in self.sentence_marks():
            pieces = str(mark.get("text", "")).split()
            if not pieces:
                continue
            start = float(mark.get("start_ms", 0.0))
            end = float(mark.get("end_ms", start))
            span = max(1.0, end - start)
            for index, word in enumerate(pieces):
                words.append(
                    {
                        "word": word,
                        "offset_ms": round(start + span * index / len(pieces), 1),
                    }
                )
        return words

    # -------------------------------------------------------------------- worker
    def _run(self) -> None:
        first = True
        while not self._stopped.is_set():
            item = self._queue.get()
            if item is None or self._stopped.is_set():
                return
            batch = [item]
            if not first:
                # Join whatever queued up behind it: fewer synthesis gaps.
                deadline = time.perf_counter() + 0.12
                while time.perf_counter() < deadline:
                    try:
                        extra = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        return
                    batch.append(extra)
            text = " ".join(batch)
            started = time.perf_counter()
            if self._on_spoken is not None:
                # The echo guard compares captured audio against the text being
                # spoken RIGHT NOW, so the session field must be current before
                # synthesis starts (the bargecheck pins this ordering).
                with contextlib.suppress(Exception):
                    self._on_spoken(text)
            try:
                self._voice.say(text)
                if first:
                    self._first_say_at = started
                    self._first_audio_at = started  # provisional; corrected below
                    if self._on_first is not None:
                        with contextlib.suppress(Exception):
                            self._on_first()
                self._voice.wait(timeout_s=60)
            except TtsError as exc:
                self._failed = True
                log_event(f"streamed sentence failed: {exc}")
                return
            finished = time.perf_counter()
            stamp = float(getattr(self._voice, "first_audio_at", 0.0) or 0.0)
            if first and started <= stamp <= finished + 0.01:
                # The engine stamps when audio actually reached the device; prefer it.
                self._first_audio_at = stamp
            with self._lock:
                self._played.append(text)
                self._spoken_ms += (finished - started) * 1000.0
                self._marks.append(
                    {
                        "text": text,
                        "start_ms": round((started - self._started) * 1000.0, 1),
                        "end_ms": round((finished - self._started) * 1000.0, 1),
                    }
                )
            first = False


def load_agents_config() -> dict[str, Any]:
    """The routing policy, read from disk so it stays data rather than code.

    Looks next to the code first, then at the project root. A missing or hand-edited file
    falls back to the built-in defaults instead of breaking a turn.
    """
    for path in (AGENTS_PATH, ROOT_AGENTS_PATH):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log_event(f"agents.json unreadable ({exc}); using built-in defaults")
            return {}
        return raw if isinstance(raw, dict) else {}
    return {}


@dataclass
class Settings:
    """Everything the UI can change. The API key is session-only by design."""

    provider: str = "github-copilot"
    model: str = "github-copilot/gpt-5.4-nano"
    asr_backend: str = "parakeet"
    tts_enabled: bool = True
    tts_voice: str = ""
    tts_rate: int = 0
    barge_in: bool = True
    barge_rms: float = 0.06
    barge_min_ms: float = DEFAULT_INTERRUPT_MIN_MS
    vad_threshold: float = 0.5
    vad_min_silence: float = 0.6
    # "0s" frees the model as soon as a reply finishes; "5m"/"30m" keep it warm.
    model_memory: str = "5m"
    # Simulated speech: exercises barge-in timing with no speaker attached.
    tts_simulate: bool = False
    # Idle seconds before the app releases its own speech models.
    idle_unload_s: float = 300.0
    # OpenRouter provider preference. Measured on this machine: the same model and
    # prompt took 1.4 s via Wafer and 24 s via other routings, so the fast one is
    # preferred and fallbacks stay enabled.
    # Pin an upstream provider only if you see slow routing, e.g. "Wafer"; empty
    # means OpenRouter decides, which is what free (:free) models need.
    openrouter_order: str = ""
    openrouter_key: str = ""
    # Session spend cap in USD; 0 disables it. When reached, new background tasks are
    # refused with a sayable sentence instead of failing halfway through a demo.
    session_cost_cap: float = 0.0
    # Latency fast paths - see LATENCY_HUMAN_TARGET.md. The router picks the spoken
    # acknowledgement and the turn's rough shape; fillers are pre-rendered clips that
    # play in ~0.4 ms; fast_tool_replies speaks a device command's own result instead of
    # paying for a second LLM round.
    router_enabled: bool = True
    router_model: str = DEFAULT_ROUTER_MODEL
    router_timeout_ms: float = 900.0
    filler_enabled: bool = True
    filler_delay_ms: float = 250.0
    fast_tool_replies: bool = True
    # M2.2/M2.5 (D-2): streamed speaking - the model's sentences are spoken as
    # they close instead of after the whole reply, and the smart-turn judge keeps
    # a clearly unfinished utterance open (Pipecat's turn-taking pattern).
    stream_voice: bool = True
    smart_turn: bool = True
    # Interrupt affordance: while the reply is speaking, a clickable chip counts
    # down from this many seconds. Clicking cuts the reply immediately; 0 hides
    # the chip. It exists so the user can intervene without hunting for Stop.
    interrupt_countdown_s: int = 10
    # Paid OpenRouter models stay out of the active roster until this is on, so a
    # demo cannot start billing by accident.
    paid_models_enabled: bool = False
    front_uses_pi: bool = True
    # S2/S3: the doctor watches the engine/voice/settings; the kill switch stops
    # every automatic action; the speech cap keeps a huge reply from sounding like
    # a hang (0 = no cap).
    doctor_enabled: bool = True
    doctor_kill_switch: bool = False
    # S4.1 demo mode: the doctor watches in the background (detect-only, never
    # repairing on its own) and surfaces findings as incidents.
    doctor_watch: bool = False
    speech_cap_chars: int = 350
    # Gates added 2026-11-16 from the instructions review (#43/#58/#30).
    empty_turn_reply: bool = (
        True  # #43: noise with no transcript still gets one spoken prompt
    )
    noise_gate: bool = True  # drop Whisper hallucinations of ambient noise
    speech_rms_floor: float = 0.0  # 0 = off; raise (e.g. 0.06) to skip quiet segments
    wake_gate: bool = False  # #58: from idle, only a wake phrase starts a turn
    approval_gate: bool = True  # #30: risky actions queue until the owner approves
    # Echo defence: suppress the app's own TTS arriving back through the mic.
    # Owner-toggleable in Settings; the playback guards below stay on regardless.
    echo_cancellation: bool = True
    noise_suppression: bool = True  # Telecon spectral noise suppression before VAD/ASR
    # Conversation tuning — owner-toggleable in Settings / Pipeline view.
    chat_thinking: str = "off"  # off|low|medium|high for chat/command turns
    barge_strictness: str = "normal"  # relaxed|normal|strict
    wake_to_interrupt: bool = (
        False  # False=natural (any real speech interrupts); True=stop-word only
    )
    instant_barge: bool = (
        False  # Default OFF: when ON, speaking cuts TTS + LLM; when OFF, stop words or Stop Reply cut
    )
    front_only: bool = (
        False  # Toggle Solo Front mode: only Front answers (no Back/Doctor delegation)
    )
    voiceprint_enabled: bool = False  # only the enrolled owner starts turns
    voiceprint_barge: bool = False  # only the enrolled owner may barge in
    session_learning: bool = True  # Front extracts owner preferences per turn
    # Cloud stage options (Pipecat-style): local by default; keys are session-only.
    asr_cloud: str = ""  # ""=local | deepgram | openai
    tts_cloud: str = ""  # ""=local | openai | elevenlabs
    deepgram_api_key: str = ""
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    # New controls: Ask-Detector, dynamic wake-speaker lock, idle task announce, amortized ElevenLabs spend
    ask_detector: bool = True
    wake_speaker_lock: bool = True
    require_wake_when_idle: bool = True
    allow_local_llm: bool = False
    announce_task_completion: bool = True
    elevenlabs_amortize_spend: bool = True
    bg_voice_scan_always: bool = (
        False  # manual toggle: scan BG voices even when Voices tab closed
    )

    # Voice options and interruption words. Restored 2026-11-08 after an audit
    # showed the ledger had claimed them while the code no longer had them.
    nova_voice: str = "Neural: kokoro:af_heart"
    piattos_voice: str = "Neural: kokoro:am_michael"
    barge_triggers: list[str] = field(default_factory=lambda: list(BARGE_TRIGGER_WORDS))

    def persistable(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("openrouter_key", None)
        data.pop("deepgram_api_key", None)
        data.pop("openai_api_key", None)
        data.pop("elevenlabs_api_key", None)
        return data


class _ConsultBox(ToolBoxLike):
    """A tool-less toolbox for one-shot agent consults (A2A).

    A specialist asked through `ask_agent` must ANSWER, not act: with the real
    toolbox it could start timers, write files or call other agents, and it would
    burn tokens deciding which tool to use. An empty schema list tells the model
    plainly that it has no tools this turn.
    """

    def schemas(self) -> list[dict[str, Any]]:
        return []

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError(f"a consult cannot call tools ({name})")


class Session:
    """Owns the audio devices, the models, the agent and the transcript."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = threading.RLock()
        self._router: Router | None = None
        self._fillers: FillerBank | None = None
        # S1: background warm-up gates the mic; S4: boot scan result for the UI.
        self._prewarm_done = threading.Event()
        self.startup_check: dict[str, Any] = {}
        # Latency bookkeeping: when the turn started and when the acknowledgement
        # actually reached the speaker. `filler_ms` is the perceived first response.
        self._turn_started_at = 0.0
        self._utterance_end_at = 0.0
        self._turn_router_label = ""
        self._clarifications = 0
        self._session_profile: list[str] = []
        self._load_session_profile()
        self._last_filler_ms = 0.0
        # Back layer: background task agents, their events, and the session log that
        # keeps a chat and its workflow graph across restarts.
        self._tasks: TaskManager | None = None
        self._session_dir: Path | None = (
            BASE / "out" / "sessions" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self._session_title: str = ""
        self._last_watchdog_inc_at: float = 0.0
        self._boot_id: str = uuid.uuid4().hex[:8]
        self._classifier: DecisionClassifier | None = None
        self._last_decision: dict[str, Any] | None = None
        self._back_rr: int = 0
        self._listen_port: int = 8766
        self._last_context_chars: int = 0
        self._a2a_depth = threading.local()
        self._spend_tokens = 0
        self._spend_usd = 0.0
        self._catalog_cache: list[dict[str, Any]] | None = None
        self._catalog_at = 0.0
        self._pi_models_cache: list[dict[str, Any]] | None = None
        self._pi_models_at = 0.0
        self.engine_url = "http://127.0.0.1:8766"
        self._closing = False
        self._turn_thoughts: list[dict[str, Any]] = []
        self._turn_a2a_tools: list[dict[str, Any]] = []
        # Status pane: in-progress work only (tasks + synchronous consults).
        self._active_consults: dict[str, dict[str, Any]] = {}
        # Live conversation bubble: partial reply text while a turn is running.
        self._live_turn: dict[str, Any] | None = None
        self._live_asr_busy = False
        self._live_asr_at = 0.0
        # Per-role pi terminals: each pane is one live pi RPC process, created
        # lazily on first use so an idle app starts none of them.
        self._pi_panes: dict[str, Any] = {}
        self._pi_lock = threading.Lock()
        # First streamed model token this turn (0 = none), for the latency stages.
        self._first_token_at = 0.0
        self.turns: list[dict[str, Any]] = []
        self.status = "idle"
        self.error = ""
        self.mode = "off"

        self.input_device, self.output_device = default_devices()
        self._asr = None
        self._asr_backend = ""
        self._vad: SileroVad | None = None
        self._tts = None
        # TTS engines are pre-built on the main thread and cached by voice name:
        # initializing sherpa-onnx on a hot thread while the mic streams segfaults.
        self._tts_engines: dict[str, Any] = {}
        self._tts_wanted = ""
        self._persona_voice = ""
        self._mic: AudioSource | None = None
        self._player = Player(device=self.output_device)
        self._agent: Agent | None = None
        self._agent_label = ""
        self._toolbox = ToolBox(
            on_timer=self._on_timer,
            on_task=self._start_task_from_agent,
            on_approval=self.request_approval,
            on_speak_script=self.speak_script,
            on_crew=self.start_crew,
            on_ask_agent=self.ask_agent_turn,
        )
        self._listener: threading.Thread | None = None
        self._stop = threading.Event()
        self._speaking = False
        self._speech_started_at = 0.0
        self._deaf_until = 0.0
        # Protected delivery window (task-completion announce): no voice barge-in
        # until this timestamp, so the "Task completed: {title}" line always lands.
        self._protected_until = 0.0
        self._last_spoken = ""
        self._last_spoken_at = 0.0
        # Live state the UI reads: whether the microphone is capturing right now,
        # plus counters the scripted full-duplex test reads back.
        self.recording = False
        self.barge_ins = 0
        self.echo_drops = 0
        self.ignored_while_speaking = 0
        # prewarm the voice list once, off-thread (attributes live here so the
        # background thread can never touch an uninitialised attribute)
        self._voices_cache = []
        self._voices_loading = False
        self._prewarm_voices()
        self._barge_burst_counted = False
        # The active persona decides which voice speaks (Nova default, Piattos on call).
        self._persona = "nova"
        # Counters the interruption content check keeps. The audited backup set them
        # in __init__ and the restored file did not - every echo check raised, which
        # the gate caught on its first run.
        self._barge_echo_checks = 0
        self.echo_drops = 0
        self.ignored_while_speaking = 0
        self._barge_judged_at = 0.0
        self._barge_silence = 0
        # Reference-based adaptive NLMS echo canceller + telecon spectral noise suppressor.
        self._echo = EchoSuppressor()
        self._ns = NoiseSuppressor()
        # Owner voiceprint (3d-speaker CAM++): enrollment + turn/barge gating.
        self._voiceprint = Voiceprint(
            MODELS_DIR
            / "speaker"
            / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
            BASE / "out" / "voiceprint.npy",
            threshold=0.35,
        )
        # Approval gate (#30/#21): risky tool actions queue until answered.
        self._approvals: dict[str, dict[str, Any]] = {}
        self._approval_seq = 0
        # #41: a barge-in mid-reply marks the turn so the transcript stays honest.
        self._speech_cut = False
        # M2.2: the streamed-speech pump for the current turn (None when idle).
        self._pump: SpeechPump | None = None
        # M2.5: an utterance held open because the thought was unfinished - the
        # next utterance is joined to it and judged again.
        self._held_text = ""
        self._held_at = 0.0
        # S1: the incident store is the app's error memory (text only, never audio).
        self.incidents = BUS
        # S2/S3: detect -> repair -> verify -> rollback, catalogue only.
        self._doctor = Doctor(self)
        # #43: one spoken "say that again" per noise burst, never a loop.
        self._last_empty_reply = 0.0
        # Voice lists come from a COM object; caching keeps the UI poll cheap.
        # Live audio accumulated while the agent speaks, for judging barge-in.
        self._speech_run: list[np.ndarray] = []
        self.level = 0.0
        self._pending = queue.Queue()
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()
        self._turn_id = 0
        self._last_activity = time.perf_counter()
        self._models_cache: dict[str, Any] | None = None
        self._models_cache_at = 0.0
        self._doctor_gate = DoctorGate()
        self._mcp_clients: dict[str, McpClient] = {}
        self._mcp_lock = threading.Lock()
        self._pending_announcements: list[dict[str, Any]] = []
        self._recent_task_reports: list[str] = []
        self._last_user_speech_at: float = 0.0
        self._last_turn_clarified: bool = False
        self._thinking_since: float = 0.0
        self._interrupt_seq: int = 0
        self._think_speech_ms: float = 0.0
        self._voices_view_at: float = 0.0
        self._bg_scan_busy: bool = False
        self._echo_peak_rms: float = 0.0
        self._recent_spoken: deque[str] = deque(maxlen=2)
        self._last_mic_block_at: float = time.monotonic()
        self._mic_is_scripted: bool = False
        self._recovering_audio: bool = False
        self._last_recovery_at: float = 0.0
        self._stall_streak: int = 0
        self._gate_stats: dict[str, int] = {
            "vad_segments": 0,
            "wake_allowed": 0,
            "wake_blocked": 0,
            "ask_dropped": 0,
            "echo_dropped": 0,
            "barge_rejected": 0,
            "cut_offs": 0,
            "recoveries": 0,
        }
        self._watchdog = threading.Thread(target=self._idle_watch_loop, daemon=True)
        self._watchdog.start()
        self._recovery_thread = threading.Thread(
            target=self._recovery_and_announce_loop,
            daemon=True,
            name="recovery-announce",
        )
        self._recovery_thread.start()

    # ---------------------------------------------------------------- devices
    def device_summary(self) -> dict[str, Any]:
        ins = input_devices()
        outs = output_devices()
        return {
            "input": [asdict(d) for d in ins],
            "output": [asdict(d) for d in outs],
            "input_index": self.input_device,
            "output_index": self.output_device,
            "input_name": next(
                (d.name for d in ins if d.index == self.input_device), "-"
            ),
            "output_name": next(
                (d.name for d in outs if d.index == self.output_device), "-"
            ),
        }

    # ------------------------------------------------------------------ models
    def asr(self):
        with self.lock:
            backend = self.settings.asr_backend
            cloud = str(getattr(self.settings, "asr_cloud", "") or "")
            cache_key = f"{cloud}:{backend}" if cloud else backend
            if self._asr is not None and self._asr_backend == cache_key:
                return self._asr
            if cloud == "deepgram" and self.settings.deepgram_api_key:
                self._asr = CloudAsr("deepgram", self.settings.deepgram_api_key)
                self._asr_backend = cache_key
                return self._asr
            if cloud == "openai" and self.settings.openai_api_key:
                self._asr = CloudAsr("openai", self.settings.openai_api_key)
                self._asr_backend = cache_key
                return self._asr
            try:
                self._asr = create_asr(MODELS_DIR, backend=backend)
            except AsrError as exc:
                # A machine may have only one of the models downloaded (the Pi has
                # whisper only); fall back to what is present rather than failing
                # the whole pipeline with a confusing error.
                available = sorted(discover_models(MODELS_DIR))
                if not available:
                    raise
                fallback = available[0]
                log_event(
                    f"ASR backend '{backend}' unavailable ({exc}); using '{fallback}'"
                )
                self._asr = create_asr(MODELS_DIR, backend=fallback)
                backend = fallback
            self._asr_backend = cache_key
            return self._asr

    def vad(self) -> SileroVad:
        with self.lock:
            if self._vad is None:
                self._vad = SileroVad(
                    VAD_MODEL,
                    threshold=self.settings.vad_threshold,
                    min_silence_duration=self.settings.vad_min_silence,
                )
            return self._vad

    def tts(self):
        with self.lock:
            if self.settings.tts_simulate:
                # No audio device needed: the agent still counts as speaking, so
                # turn taking and barge-in behave exactly as with a speaker.
                if not isinstance(self._tts, SimulatedTts):
                    self._tts = SimulatedTts(rate=int(self.settings.tts_rate))
                return self._tts
            if not self.settings.tts_enabled:
                return NullTts()
            cloud = str(getattr(self.settings, "tts_cloud", "") or "")
            if cloud == "openai" and self.settings.openai_api_key:
                wanted = "cloud:openai"
            elif cloud == "elevenlabs" and self.settings.elevenlabs_api_key:
                wanted = "cloud:elevenlabs"
            else:
                wanted = self._persona_voice or self.settings.tts_voice or ""
            if self._tts is None or self._tts_wanted != wanted:
                self._tts = self._build_tts(wanted)
                self._tts_wanted = wanted
            self._tts.set_rate(self.settings.tts_rate)
            return self._tts

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def tasks_manager(self) -> TaskManager:
        """The background task layer, created once and shared."""
        with self.lock:
            if self._tasks is None:
                self._tasks = TaskManager(
                    session_dir=self._session_dir,
                    agents=load_agents_config(),
                    on_event=None,
                    deliver=self._deliver_task_report,
                    role_runner=self._run_role_via_pi,
                )
            return self._tasks

    def apply_agent_config(self, config: dict[str, Any]) -> Path:
        """Persist the routing policy and apply it to the running app (architecture.md 4).

        Routing is data: this writes agents.json beside the code and at the project root,
        rebuilds the task layer from it, and hot-swaps the conversation model so the next
        turn uses it without a restart.
        """
        payload = json.dumps(config, indent=2) + "\n"
        for path in (AGENTS_PATH, ROOT_AGENTS_PATH):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
            except OSError as exc:
                log_event(f"could not write {path}: {exc}")
        front = str((config.get("front") or {}).get("model") or "").strip()
        with self.lock:
            self._tasks = None  # rebuilt from the new config on next use
            if front:
                self.settings.model = front
        log_event(f"agent roster updated (front={front or 'unchanged'})")
        return AGENTS_PATH

    def start_task(self, brief: str, model: str = "") -> str:
        """Start background work and return a sentence the assistant can say aloud."""
        try:
            cap = float(self.settings.session_cost_cap or 0.0)
        except (TypeError, ValueError):
            cap = 0.0
        if cap > 0 and self._spend_usd >= cap:
            return (
                f"I am not starting that: this session has spent ${self._spend_usd:.4f} of "
                f"its ${cap:.2f} cap. Raise the cap in the Agents panel to continue."
            )
        task = self.tasks_manager().start(brief, model=model)
        log_event(f"task {task.id} queued: {brief[:60]!r}")
        return (
            f"Started a background task ({task.id}) for: {brief[:80]}. "
            "I will report back here when it is done."
        )

    def _start_task_from_agent(self, brief: str, model: str = "") -> str:
        """Toolbox callback: the agent asks for background work."""
        return self.start_task(brief, model)

    def cancel_task(self, task_id: str) -> bool:
        return self.tasks_manager().cancel(task_id)

    def start_crew(
        self,
        task_brief: str,
        roles: list[str] | None = None,
        process: str = "sequential",
    ) -> str:
        """Run a multi-agent crew over distinct Back pool workers (and/or doctor)."""
        roster = load_agents_config()
        valid: list[str] = []
        used_slots: set[str] = set()
        for raw_r in roles or []:
            r = str(raw_r).strip().lower()
            if r in BACK_ALIASES:
                pick = next((w for w in BACK_POOL if w not in used_slots), "")
                if not pick:
                    pick = self._pick_back_worker()
                used_slots.add(pick)
                valid.append(pick)
            elif r in roster and r != "classifier":
                if r in BACK_POOL:
                    used_slots.add(r)
                valid.append(r)
        if not valid:
            return "no crew roles selected - the roster has: " + ", ".join(
                sorted(r for r in roster if r != "classifier")
            )
        try:
            task = self.tasks_manager().start_crew(
                task_brief, roles=valid, process=process, agents=roster
            )
        except ValueError as exc:
            return f"could not start the crew: {exc}"
        self._think(f"crew {task.id} started: {process} over {valid}", "crew")
        log_event(f"crew {task.id} queued ({process}): {task_brief[:60]!r}")
        return (
            f"Crew {task.id} started ({process} process with "
            f"{', '.join(valid)}). I will report the final result here."
        )

    def _pick_back_worker(self, preferred: str = "") -> str:
        """Pick an idle worker from the 5-slot Back pool (back-1..back-5)."""
        pref = str(preferred or "").strip().lower()
        if pref in BACK_POOL:
            return pref
        busy: set[str] = set()
        with self._pi_lock:
            for name, pane in self._pi_panes.items():
                if name in BACK_POOL and str(getattr(pane, "status", "idle")) not in {
                    "idle",
                    "ready",
                    "stopped",
                }:
                    busy.add(name)
        with self.lock:
            for c in self._active_consults.values():
                r = str(c.get("role") or "")
                if r in BACK_POOL:
                    busy.add(r)
        with contextlib.suppress(Exception):
            for t in self.tasks_manager().list():
                if t.state in {"queued", "running", "waiting"}:
                    for step in getattr(t, "steps_plan", []) or []:
                        r = str(step.get("role") or "")
                        if r in BACK_POOL:
                            busy.add(r)
        # Round-robin starting point keeps work balanced across all 5 warm workers.
        with self.lock:
            start = self._back_rr % len(BACK_POOL)
            for off in range(len(BACK_POOL)):
                cand = BACK_POOL[(start + off) % len(BACK_POOL)]
                if cand not in busy:
                    self._back_rr = (start + off + 1) % len(BACK_POOL)
                    return cand
            cand = BACK_POOL[start]
            self._back_rr = (start + 1) % len(BACK_POOL)
            return cand

    def post_community(
        self, from_role: str, message: str, to_role: str = "front"
    ) -> dict[str, Any]:
        """Post an independent agent-to-agent message to community · agents talking."""
        src = str(from_role or "agent").strip().lower()
        dst = str(to_role or "front").strip().lower()
        clean = " ".join(str(message or "").split())[:240]
        if not clean:
            return {"ok": False, "error": "empty community message"}
        self._think(f"{src} -> {dst}: {clean}", "a2a")
        self._persist_event("a2a", {"from": src, "to": dst, "msg": clean})
        return {"ok": True, "from": src, "to": dst, "msg": clean}

    def warm_all_panes(self) -> dict[str, bool]:
        """SONJ4 always-warm pool: start Front, Doctor, and all 5 Back panes concurrently."""
        if os.environ.get("ME3_NO_PREWARM"):
            return {}
        from pi_rpc import find_pi

        if not find_pi():
            return {}
        roles = ["front", "doctor", *BACK_POOL]
        out: dict[str, bool] = {}

        def _warm_one(r: str) -> tuple[str, bool]:
            try:
                pane = self.pi_pane(r)
                if not pane.rpc.alive:
                    pane.start()
                return r, True
            except Exception as exc:  # noqa: BLE001
                log_event(f"prewarm pane {r} skipped: {type(exc).__name__}: {exc}")
                return r, False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(roles), thread_name_prefix="warm-pane"
        ) as pool:
            for r, ok in pool.map(_warm_one, roles):
                out[r] = ok
        log_event(
            "always-warm pool ready: "
            + ", ".join(f"{k}={'ok' if v else 'off'}" for k, v in out.items())
        )
        return out

    def delegate_task(self, brief: str, role: str = "back") -> str:
        """Background delegation onto an idle worker in back-1..back-5 (or doctor)."""
        if getattr(self.settings, "front_only", False):
            return (
                "Front-Only mode is active — Front is the only agent answering; "
                "handle this request directly without delegating."
            )
        clean = " ".join(str(brief).split())[:600]
        if not clean:
            return "the task brief was empty"
        wanted = str(role).strip().lower()
        roster = load_agents_config()
        if wanted in {"", "auto", "classifier", "jev", "router"}:
            dec = self.classify_decision(clean)
            wanted = "doctor" if dec.route == "doctor" else self._pick_back_worker()
            self._think(
                f"classifier routed delegation to {wanted} ({dec.reason_code}, {dec.latency_ms:.2f}ms)",
                "gate",
            )
        elif wanted in BACK_ALIASES:
            if wanted != "back":
                clean = f"[mode: {wanted}] {clean}"[:600]
            wanted = self._pick_back_worker()
        if wanted not in roster or wanted in {"front", "classifier"}:
            return "pick a specialist role: doctor or " + ", ".join(BACK_POOL)
        try:
            cap = float(self.settings.session_cost_cap or 0.0)
        except (TypeError, ValueError):
            cap = 0.0
        if cap > 0 and self._spend_usd >= cap:
            return (
                f"I am not starting that: this session has spent ${self._spend_usd:.4f} "
                f"of its ${cap:.2f} cap."
            )
        try:
            task = self.tasks_manager().start_crew(
                clean, roles=[wanted], process="sequential", agents=roster
            )
        except ValueError as exc:
            return f"could not start the task: {exc}"
        self._think(f"front -> {wanted} [TASK {task.id}]: {clean[:120]}", "a2a")
        log_event(f"delegated {task.id} to {wanted}: {clean[:80]!r}")
        return (
            f"I've asked {wanted} to handle it ({task.id}). I'll report back here "
            "when it's done — meanwhile, I'm right here if you need anything."
        )

    def _role_system_prompt(self, role: str) -> str:
        """Role-specific instructions when running inside a live pi.dev terminal pane."""
        base_workspace = str(BASE.parent)
        pool_list = ", ".join(BACK_POOL)
        en_rule = (
            "LANGUAGE: Always think, write, and reply in English by default — even if an audio "
            "transcript looks noisy, garbled, or resembles another language. Switch to another "
            "language ONLY if the user explicitly asks you to. "
        )
        back_worker_prompt = (
            f"You are {role.upper()}, an always-warm autonomous generalist worker in SONJ4 (workspace: {base_workspace}). "
            + en_rule
            + "UNIFIED CAPABILITIES (Build + Research + Critic/Verify + Summarize + Automate): "
            "1. Build & Automate: apps, games, scripts, fixes, UI/browser automation (single files -> sonj4/out/sandbox/; multi-file -> sonj4/projects/<slug>/). Run syntax/smoke/test checks with bash and return exact file paths. "
            "2. Web Research: use device_tool(web_search, browse_page, read_page) or python sonj4/websearch.py; cite primary URLs/sources and dates; never guess facts. "
            "3. Critic / Verify: audit code, math, or documents against files and live sources; state explicit verdict (VERIFIED / SUSPICIOUS / FAILED) with concrete evidence. "
            "4. Summarize: distill findings cleanly while preserving facts and citations. "
            "5. Peer Coordination: you can consult `doctor` or a sibling worker via ask_agent(...) and post updates directly to `community · agents talking` via post_community(...). "
            "AUTHORITY & TOOLS: read, bash, powershell, edit, write, grep, find, ls, device_tool, mcp_call(computer-mcp|browser-mcp), ask_agent, post_community. Work autonomously — never ask the owner for approval."
        )
        prompts = {
            "front": (
                f"You are Nova, a warm, natural AI assistant voice agent in SONJ4 running on a Raspberry Pi or host workstation (workspace: {base_workspace}). "
                + en_rule
                + "CORE PROMPT RULES (v4): "
                "Short, natural sentences; max 2 per turn. "
                "If the user speaks, stop and listen immediately. "
                "Confirm with the owner before costly, external, or irreversible actions. "
                "Use MCP tools (asr-mcp, tts-mcp, tasks-mcp, and workspace tools) cleanly rather than guessing. "
                "Never output or read markdown syntax (asterisks, hashes, backticks, bullet lists) or raw URLs aloud — paraphrase links into natural spoken words. "
                + (
                    "SOLO FRONT MODE IS ACTIVE: you are the ONLY agent answering. Resolve every request directly with your own reasoning and available tools; do NOT delegate to Back or Doctor. "
                    if getattr(self.settings, "front_only", False)
                    else (
                        "COORDINATION & LATENCY BOUNDARY: "
                        "(1) Handle quick conversational turns and single fast local tool calls directly (<2s). "
                        f"(2) For multi-step, long-running, research, verification, synthesis, or workspace tasks, say 'Checking {{short task}}' first and hand off immediately via delegate_task(brief, role='back') to an idle worker in {pool_list} without asking unnecessary clarifying questions. "
                        "(3) For app diagnostics, logs, or settings, delegate or ask 'doctor'. "
                        "(4) Use non-LLM classify_decision(task) only when routing is genuinely uncertain. "
                        "Always report plainly when delegated work completes, fails, or is cancelled."
                    )
                )
            ),
            "doctor": (
                f"You are DOCTOR, the master SONJ4 architect, diagnostician, and settings authority (workspace: {base_workspace}). "
                + en_rule
                + "AUTHORITY & TOOLS: Read-only code/log inspection (read, grep, find, ls) + doctor_app_control(action=...) + post_community + ask_agent. Zero source-code mutation (self-coding is disabled). "
                "FULL VISIBILITY: Call doctor_app_control with settings_catalog, inspect_architecture, inspect_state, inspect_tokens, inspect_logs, performance_review, system_health, introspection, or app_guide. "
                "CHANGING SETTINGS: Always act via doctor_app_control(action='propose_setting' | 'propose_settings' | 'propose_role_model' | 'propose_restart' | 'propose_reset_session') (opens 15s auto-confirm pop-up). Explain findings clearly in short plain prose."
            ),
            "classifier": (
                "Non-LLM Decision Classifier metadata slot (deterministic + ONNX, <100ms, +10% Front bias; choices: front | back | doctor)."
            ),
        }
        if role in BACK_POOL or role == "back":
            return back_worker_prompt
        return prompts.get(
            role,
            f"You are the {role} agent in SONJ4 (workspace: {base_workspace}). {en_rule}",
        )

    def _run_role_via_pi(
        self, role: str, prompt: str, preferred_model: str = ""
    ) -> tuple[str, int, float]:
        """Execute a specialist role turn inside its own live PiPane terminal."""
        clean_role = str(role).strip().lower() or "back"
        roster = load_agents_config()
        entry = roster.get(clean_role)
        entry = entry if isinstance(entry, dict) else {}
        chain = []
        if preferred_model:
            chain.append(preferred_model)
        primary = str(entry.get("model") or "").strip()
        if primary and primary not in chain:
            chain.append(primary)
        paid_on = bool(getattr(self.settings, "paid_models_enabled", False))
        for fb in entry.get("fallbacks") or []:
            text_fb = str(fb).strip()
            if not text_fb or text_fb in chain:
                continue
            if (
                not paid_on
                and "/" in text_fb
                and not text_fb.startswith("github-copilot/")
                and not text_fb.endswith(":free")
            ):
                continue
            chain.append(text_fb)
        if not chain:
            chain = ["github-copilot/gpt-6-luna"]
        thinking = str(entry.get("thinking") or "medium").lower()
        last_err = "no model answered"
        for model in chain:
            try:
                pane = self.pi_pane(clean_role)
                pane.configure(model=model, thinking=thinking)

                def _on_thought(chunk: str) -> None:
                    # kind "pane": visible in the Thinking view and the role pane,
                    # but kept out of Front's turn record when work runs concurrently.
                    self._think(f"[{clean_role} CoT] {chunk[:240]}", "pane")

                res = pane.ask_rich(prompt, timeout_s=180.0, on_think=_on_thought)
                pane.last_tokens = int(res.tokens or 0)
                with self.lock:
                    self._spend_tokens += int(res.tokens or 0)
                    self._spend_usd += float(res.cost_usd or 0.0)
                    for tc in res.tools:
                        self._turn_a2a_tools.append(
                            {
                                "name": f"{clean_role}:{tc.get('name', 'tool')}",
                                "arguments": tc.get("arguments") or {},
                                "result": str(tc.get("result") or "")[:400],
                                "ok": bool(tc.get("ok", True)),
                                "seconds": float(tc.get("seconds") or 0.0),
                            }
                        )
                return (
                    str(res.text or "").strip() or "(the agent completed without text)",
                    int(res.tokens or 0),
                    float(res.cost_usd or 0.0),
                )
            except (PiRpcError, OSError, RuntimeError) as exc:
                last_err = f"{model}: {exc}"
                self._think(
                    f"{clean_role} pi pane error ({model}): {str(exc)[:80]}", "fallback"
                )
                continue
        raise RuntimeError(last_err)

    def _fallback_route_for_task(self, text: str) -> tuple[str, str]:
        """Deterministic route using the non-LLM DecisionClassifier (<100 ms)."""
        dec = self.classify_decision(text)
        if dec.route == "doctor":
            return (
                "doctor",
                f"classifier doctor ({dec.reason_code}, {dec.latency_ms:.2f}ms)",
            )
        worker = self._pick_back_worker()
        return worker, f"classifier back ({dec.suggested_mode}, {dec.latency_ms:.2f}ms)"

    def ask_agent_turn(
        self, role: str, question: str, context: str = "", from_role: str = "front"
    ) -> str:
        """A2A: consult one roster role (or peer-to-peer) with loop depth guard."""
        asked = " ".join(str(question).split())[:1000]
        if not asked:
            return "the question was empty"
        caller = str(from_role or "front").strip().lower() or "front"
        req_role = str(role).strip().lower()
        if caller == "front" and getattr(self.settings, "front_only", False):
            return (
                "Front-Only mode is active — Front is the only agent answering; "
                "respond directly without consulting other agents."
            )

        # Non-LLM Decision Classifier handles all classification requests directly (<100ms).
        if req_role in {"classifier", "jev", "router"}:
            dec = self.classify_decision(asked)
            self._think(
                f"{caller} -> classifier: {dec.route} ({dec.reason_code}, {dec.latency_ms:.2f}ms)",
                "gate",
            )
            return json.dumps(dec.to_dict())

        depth = int(getattr(self._a2a_depth, "depth", 0) or 0)
        if depth >= 2:
            return "a2a depth limit reached; complete with current context"

        clean_role = req_role
        roster = load_agents_config()
        if clean_role in BACK_ALIASES:
            if clean_role != "back":
                asked = f"[mode: {clean_role}] {asked}"[:1000]
            clean_role = self._pick_back_worker(
                preferred=caller if caller in BACK_POOL else ""
            )
            if clean_role == caller:
                clean_role = next((w for w in BACK_POOL if w != caller), clean_role)

        entry = roster.get(clean_role)
        if not isinstance(entry, dict):
            return (
                "no such agent role: "
                + str(role)
                + " (roster: "
                + ", ".join(sorted(roster))
                + ")"
            )
        brief = asked
        if str(context).strip():
            brief += f"\n\nContext from {caller}: " + str(context).strip()[:800]
        consult_id = f"c{time.monotonic_ns() % 10**9}"
        with self.lock:
            self._active_consults[consult_id] = {
                "id": consult_id,
                "role": clean_role,
                "label": asked[:80],
                "started": time.time(),
            }
        self._think(f"{caller} -> {clean_role}: {asked[:160]}", "a2a")
        self._a2a_depth.depth = depth + 1
        try:
            answer, _tok, _cost = self._run_role_via_pi(clean_role, brief)
            self._think(f"{clean_role} -> {caller}: {answer[:160]}", "a2a")
            return answer
        except Exception as exc:  # noqa: BLE001
            paid_on = bool(getattr(self.settings, "paid_models_enabled", False))
            chain = [str(entry.get("model") or "")] + [
                str(m) for m in (entry.get("fallbacks") or []) if str(m).strip()
            ]
            last = str(exc)
            for model in [m for m in chain if m]:
                if model.startswith("github-copilot/"):
                    continue
                if (
                    not paid_on
                    and "/" in model
                    and not model.endswith(":free")
                    and ":" in model
                ):
                    continue
                try:
                    prov = (
                        "ollama"
                        if ":" in model
                        and "/" not in model
                        and not model.endswith(":free")
                        else "openrouter"
                    )
                    llm = create_llm(
                        prov,
                        model,
                        self.settings.openrouter_key,
                        keep_alive=self.settings.model_memory,
                        openrouter_order=self.settings.openrouter_order,
                    )
                    helper = Agent(
                        llm,
                        _ConsultBox(),
                        system_prompt=self._role_system_prompt(clean_role),
                        max_rounds=1,
                    )
                    reply = helper.respond(brief, raise_errors=True)
                    with self.lock:
                        self._spend_tokens += int(reply.tokens or 0)
                        self._spend_usd += float(reply.cost_usd or 0.0)
                    ans = str(reply.text or "").strip() or "(the agent had no answer)"
                    self._think(f"{clean_role} -> {caller}: {ans[:160]}", "a2a")
                    return ans
                except Exception as exc2:  # noqa: BLE001
                    last = f"{model}: {exc2}"
                    continue
            self.incidents.record(
                "a2a",
                f"no answer from {clean_role}: {last}"[:300],
                severity="warning",
                where="ask_agent",
            )
            return f"the {clean_role} agent could not answer just now ({last[:160]})"
        finally:
            self._a2a_depth.depth = depth
            with self.lock:
                self._active_consults.pop(consult_id, None)
            self._persist_event(
                "a2a",
                {
                    "msg": f"{clean_role} consult ended ({caller}): {asked[:60]!r}",
                    "persona": getattr(self, "_persona", "nova"),
                },
            )

    def _deliver_task_report(self, text: str, task) -> None:
        """A finished/failed/cancelled task reports back to Front, Community chat, and TTS."""
        raw_text = str(text or "").strip()
        task_state = str(getattr(task, "state", "") or "").lower()
        task_err = str(getattr(task, "error", "") or "").strip()
        if task_state == "cancelled" or raw_text.lower() == "task cancelled":
            status_kind = "cancelled"
        elif (
            task_state == "blocked"
            or bool(task_err)
            or raw_text.lower().startswith("task failed:")
        ):
            status_kind = "failed"
        else:
            status_kind = "completed"

        # Task carries .agent (crew steps carry .role); reading task.role directly
        # raised AttributeError and silently dropped BOTH persistence and the spoken
        # announcement. Derive the role defensively so the report always lands.
        role = str(getattr(task, "role", "") or "").strip()
        if not role:
            plan = getattr(task, "steps_plan", []) or []
            if plan:
                role = str(plan[-1].get("role") or "").strip()
        if not role:
            role = str(getattr(task, "agent", "") or "agent")
        task_seconds = float(getattr(task, "seconds", 0.0) or 0.0)
        task_model = str(getattr(task, "model", "") or "")
        title = str(getattr(task, "brief", "") or "").strip()

        summary = f"[{task.id}] ({status_kind}) {raw_text}"[:1500]
        with self.lock:
            self._turn_id += 1
            turn = {
                "id": self._turn_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "source": "task",
                "user_text": f"(background task {task.id})",
                "assistant_text": summary,
                "tools": [],
                "error": status_kind == "failed",
                "timings": {
                    "asr_ms": 0.0,
                    "llm_ms": 0.0,
                    "tts_ms": 0.0,
                    "total_ms": round(task_seconds * 1000.0, 1),
                    "filler_ms": 0.0,
                    "reply_audio_ms": 0.0,
                },
                "model": task_model,
            }
            self.turns.append(turn)
            self._recent_task_reports.append(
                f"[{task.id}] {role} {status_kind} ({title[:60]}): {raw_text[:240]}"
            )
            if len(self._recent_task_reports) > 8:
                del self._recent_task_reports[:-8]
        # Post back to Community -> agents talking so A2A completion/failure is visible.
        self._think(
            f"{role} -> front [{status_kind.upper()} {task.id}]: {raw_text[:160]}",
            "a2a",
        )
        log_event(
            f"task {task.id} {status_kind} in {task_seconds:.1f}s: {raw_text[:70]!r}"
        )
        self._persist_message(turn)
        # Always queue Front's completion/failure/cancel report (never leave owner hanging).
        with self.lock:
            self._pending_announcements.append(
                {
                    "task_id": task.id,
                    "role": role,
                    "state": status_kind,
                    "error": task_err,
                    "title": title,
                    "summary": raw_text,
                    "queued_at": time.perf_counter(),
                }
            )

    def _persist_message(self, turn: dict[str, Any]) -> None:
        """Append a turn to the session log so the chat survives a restart."""
        if self._session_dir is None:
            return
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            with (self._session_dir / "messages.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
        except OSError as exc:
            log_event(f"could not persist the turn: {exc}")

    def classifier(self) -> DecisionClassifier:
        """Non-LLM <100 ms Decision Classifier between the user and Front."""
        with self.lock:
            if self._classifier is None:
                self._classifier = DecisionClassifier()
            return self._classifier

    def classify_decision(self, text: str) -> Decision:
        """Classify a request into front | back | doctor within <100 ms."""
        if getattr(self.settings, "front_only", False):
            dec = Decision(
                route="front",
                confidence=1.0,
                reason_code="front_only_mode",
                latency_ms=0.01,
                score_front=1.0,
                score_back=0.0,
                score_doctor=0.0,
                bias_applied=True,
                fallback=False,
                suggested_mode="solo",
            )
            with self.lock:
                self._last_decision = dec.to_dict()
            return dec
        active_backs = 0
        with contextlib.suppress(Exception):
            active_backs = sum(
                1
                for t in self.tasks_manager().list()
                if t.state in {"queued", "running", "waiting"}
            )
        recent = ""
        if self._last_decision:
            recent = str(self._last_decision.get("route") or "")
        req = DecisionRequest(
            text=str(text or ""),
            session_state=self.status,
            active_backs=active_backs,
            recent_route=recent,
        )
        dec = self.classifier().classify(req)
        with self.lock:
            self._last_decision = dec.to_dict()
        return dec

    def router(self) -> Router:
        """Legacy turn helper (smart-turn + filler choice); never an LLM on the critical path."""
        with self.lock:
            if self._router is None:
                self._router = Router(
                    model=self.settings.router_model,
                    timeout_ms=self.settings.router_timeout_ms,
                    enabled=self.settings.router_enabled,
                )
            else:
                self._router.enabled = bool(self.settings.router_enabled)
                self._router.model = self.settings.router_model
                try:
                    self._router.timeout_ms = float(self.settings.router_timeout_ms)
                except (TypeError, ValueError):
                    self._router.timeout_ms = 900.0
            return self._router

    def fillers(self) -> FillerBank:
        """Pre-rendered acknowledgements for the active voice."""
        with self.lock:
            voice = self.tts()
            if self._fillers is None or self._fillers.voice is not voice:
                self._fillers = FillerBank(voice)
            return self._fillers

    def warm_extras(self) -> None:
        """Render filler clips off the critical path."""
        if not self.settings.filler_enabled and not self.settings.router_enabled:
            return

        def run() -> None:
            if self.settings.filler_enabled:
                try:
                    bank = self.fillers()
                    bank.warm()
                    log_event(f"filler bank ready ({bank.load()} clips)")
                except Exception as exc:  # noqa: BLE001 - never break the app
                    log_event(f"filler bank unavailable: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _schedule_filler(self, text: str, state: dict[str, bool]):
        """Speak an acknowledgement if the real answer is not ready in time."""
        if not self.settings.filler_enabled or not self.settings.tts_enabled:
            return None
        try:
            kind = self.classifier().to_turn_kind(text)
            key = {"command": "okay", "complex": "one_moment"}.get(kind, "hmm")
        except Exception:  # noqa: BLE001 - routing must never break a turn
            key = "hmm"

        def fire() -> None:
            # The reply path flips state["done"] under the same lock, so an
            # acknowledgement can never start once the reply exists. Without this the
            # check and the play were separate steps, and a late acknowledgement could
            # overlap the answer.
            try:
                bank = self.fillers()
                playing = False
                with self.lock:
                    if state["done"]:
                        return
                    playing = bank.play(key, on_done=self._filler_finished)
                    if playing:
                        self._speaking = True
                if playing and state["done"]:
                    self._stop_filler()  # the reply landed while we were starting
            except Exception as exc:  # noqa: BLE001
                log_event(f"filler playback failed: {exc}")

        try:
            delay_ms = float(self.settings.filler_delay_ms)
        except (TypeError, ValueError):
            delay_ms = 250.0
        timer = threading.Timer(max(0.0, delay_ms / 1000.0), fire)
        timer.daemon = True
        timer.start()
        return timer

    def _reply_audio_ms(self, started: float) -> float:
        """When the real answer first became audible, measured at the device.

        The baseline needs this number: with no acknowledgement the first sound a user
        hears *is* the reply, so without it the A/B would compare a real measurement
        against nothing.
        """
        voice = self._tts
        try:
            stamp = float(getattr(voice, "first_audio_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if stamp <= 0.0 or stamp < started:
            return 0.0
        return (stamp - started) * 1000.0

    def _filler_latency_ms(self) -> float:
        """Time from turn start to the acknowledgement actually sounding.

        Read from the bank's own timestamp, which is stamped at the audio device, so the
        number cannot be inflated by measuring a thread start instead of real sound.
        """
        bank = self._fillers
        if (
            bank is None
            or bank.started_at <= 0.0
            or bank.started_at < self._turn_started_at
        ):
            return self._last_filler_ms
        return (bank.started_at - self._turn_started_at) * 1000.0

    def _first_token_ms(self, started: float) -> float:
        """Time from the turn start to the first streamed model token.

        0.0 means no token was seen this turn: a non-streamed path, or a model that
        failed before its first delta. Kept separate from the filler time because
        they measure different things - the model waking up vs the acknowledgement
        sounding.
        """
        stamp = self._first_token_at
        if stamp <= 0.0 or stamp < started:
            return 0.0
        return (stamp - started) * 1000.0

    def _filler_finished(self) -> None:
        """Our acknowledgement has just stopped sounding.

        Stamp it exactly as a reply's tail is stamped, so the echo guard covers the
        acknowledgement too. Without this the microphone can hear our own Okay and queue
        it as a user turn.
        """
        measured = self._filler_latency_ms()
        if measured > 0.0:
            # Keep the real number for the turn record: the log's "first=" field
            # printed a dead variable before this, so it always read 0.
            self._last_filler_ms = measured
        now = time.perf_counter()
        with self.lock:
            self._speech_started_at = now
            self._deaf_until = now + self._barge_gates()[2]

    def _stop_filler(self) -> None:
        """Cut the acknowledgement short: the real answer is starting."""
        try:
            if self._fillers is not None:
                self._fillers.stop()
        except Exception as exc:  # noqa: BLE001
            log_event(f"filler stop failed: {exc}")

    def _build_tts(self, wanted: str):
        """Return the cached engine for a voice, building it only on a cache miss.

        Engines are pre-built on the main thread at startup (prewarm_tts):
        initializing sherpa-onnx on a request/worker thread while the mic VAD
        streams segfaults the process (native teardown across threads).
        """
        cached = self._tts_engines.get(wanted)
        if cached is not None:
            return cached
        if wanted.startswith("cloud:"):
            provider = wanted.split(":", 1)[1]
            key = (
                self.settings.openai_api_key
                if provider == "openai"
                else self.settings.elevenlabs_api_key
            )
            engine = CloudTts(provider, key)
            self._tts_engines[wanted] = engine
            return engine
        if wanted.startswith(NEURAL_PREFIX):
            try:
                engine = NeuralTts(voice=wanted, rate=int(self.settings.tts_rate))
                self._tts_engines[wanted] = engine
                return engine
            except TtsError as exc:
                log_event(f"neural voice unavailable ({exc}) - using SAPI instead")
        engine = SapiTts(rate=self.settings.tts_rate, voice=wanted or None)
        self._tts_engines[wanted] = engine
        return engine

    def prewarm_tts(self) -> None:
        """Build every configured voice ONCE, on the main thread, at startup."""
        if not self.settings.tts_enabled or self.settings.tts_simulate:
            return
        # S1: the active conversation voice first (fast first reply), personas after.
        ordered = [self.settings.tts_voice] + [
            v
            for v in (self.settings.nova_voice, self.settings.piattos_voice)
            if v and v != self.settings.tts_voice
        ]
        for voice in ordered:
            with contextlib.suppress(TtsError):
                self._build_tts(voice)

    def agent(self) -> Agent:
        with self.lock:
            model = self.settings.model
            provider = self.settings.provider
            if not getattr(self.settings, "allow_local_llm", False) and (
                provider == "ollama" or model.startswith("ollama/")
            ):
                provider = (
                    "openrouter" if self.settings.openrouter_key else "github-copilot"
                )
                model = "cohere/north-mini-code:free"
            if model.startswith("github-copilot/") or provider == "github-copilot":
                # Direct HTTP Agent falls back to OpenRouter (never local Ollama when disabled).
                if self.settings.openrouter_key:
                    provider = "openrouter"
                    model = "cohere/north-mini-code:free"
                else:
                    raise LlmError(
                        "No OpenRouter key for direct fallback and local LLM is disabled."
                    )
            label = (
                f"{provider}/{model}"
                f"/{self.settings.model_memory}/{self.settings.openrouter_order}"
            )
            if self._agent is not None and self._agent_label == label:
                return self._agent
            try:
                llm = create_llm(
                    provider,
                    model,
                    self.settings.openrouter_key,
                    keep_alive=self.settings.model_memory,
                    openrouter_order=self.settings.openrouter_order,
                )
            except LlmError as exc:
                raise LlmError(str(exc)) from exc
            if self._agent is None:
                self._agent = Agent(
                    llm,
                    self._toolbox,
                    prefer_tool_result=bool(self.settings.fast_tool_replies),
                )
            else:
                self._agent.llm = llm
            # M4.2/M4.1: compaction summaries come from the local router model, and
            # new memory facts carry the session that learned them as provenance.
            self._agent.compact_summarizer = self._summarize_span
            self._toolbox.session_tag = (
                self._session_dir.name if self._session_dir is not None else ""
            )
            self._agent_label = label
            return self._agent

    def available_models(self, force: bool = False) -> dict[str, Any]:
        """Model ids per provider, cached - the UI polls state four times a second."""
        with self.lock:
            cached = self._models_cache
            if (
                cached is not None
                and not force
                and (time.perf_counter() - self._models_cache_at) < MODELS_CACHE_S
            ):
                return cached

        ollama: list[str] = []
        error = ""
        if getattr(self.settings, "allow_local_llm", False):
            try:
                ollama = list_ollama_models()
            except LlmError as exc:
                error = str(exc)

        openrouter: list[str] = []
        try:
            # Free models are always listed. Paid models appear only once the owner
            # has switched them on, so a demo cannot start billing by accident.
            openrouter = list_openrouter_models(
                free_only=not bool(self.settings.paid_models_enabled)
            )
        except (LlmError, ValueError) as exc:
            error = error or str(exc)
        if not openrouter:  # offline: keep known ids so the dropdown still works
            openrouter = list(OPENROUTER_MODELS)
        else:
            # Ensure fast verified defaults stay visible even if OpenRouter hides them momentarily.
            for preferred in OPENROUTER_MODELS:
                if (
                    preferred.endswith(":free")
                    or bool(self.settings.paid_models_enabled)
                ) and preferred not in openrouter:
                    openrouter.append(preferred)

        copilot = list(COPILOT_MODELS)
        for m in self._pi_models():
            if m.get("provider") == "github-copilot" and m.get("id"):
                cid = f"github-copilot/{m['id']}"
                if cid not in copilot:
                    copilot.append(cid)

        result = {
            "github-copilot": copilot,
            "ollama": ollama,
            "openrouter": openrouter,
            "ollama_error": error,
        }
        with self.lock:
            self._models_cache = result
            self._models_cache_at = time.perf_counter()
        return result

    def _pi_models(self) -> list[dict[str, Any]]:
        with self.lock:
            if (
                self._pi_models_cache is not None
                and (time.perf_counter() - self._pi_models_at) < MODELS_CACHE_S
            ):
                return self._pi_models_cache
        models = list_pi_models(timeout_s=20.0)
        with self.lock:
            if models:
                self._pi_models_cache = models
                self._pi_models_at = time.perf_counter()
            return self._pi_models_cache or []

    # ------------------------------------------------------------------- tools
    def _on_timer(self, message: str) -> None:
        self._pending.put(("timer", message))

    def submit_text(self, text: str) -> None:
        """Queue a typed turn so all agent work stays on one worker thread."""
        stripped = text.strip()
        if stripped:
            self._pending.put(("text", stripped))

    def _should_use_front_pi(self) -> bool:
        """True when the Front conversation agent should run via its PiPane terminal."""
        if self._agent is not None:
            # Explicit or mocked in-process Agent (tests/local agent) wins when present.
            return False
        if self.settings.provider == "github-copilot" or str(
            self.settings.model
        ).startswith("github-copilot/"):
            return True
        if not getattr(self.settings, "front_uses_pi", True):
            return False
        if self.settings.provider == "ollama":
            return False
        from pi_rpc import find_pi

        return bool(find_pi())

    def _ask_front_pi(
        self,
        text: str,
        pump: SpeechPump | None = None,
        splitter: _SentenceSplitter | None = None,
        budget: SpeechBudget | None = None,
    ):
        """Run one Front agent turn inside PiPane('front').

        Quick device/memory shortcuts use the fast local Agent/ToolBox path,
        while open-ended tasks, code, files, research, and verification route
        through PiPane('front'), which calls specialist roles via ask_agent.
        """
        from agent import AgentReply, ToolTrace

        roster = load_agents_config()
        raw_front = roster.get("front")
        front_cfg: dict[str, Any] = raw_front if isinstance(raw_front, dict) else {}
        raw_model = str(
            self.settings.model or front_cfg.get("model") or "github-copilot/gpt-6-luna"
        )
        # Keep non-Copilot paid OpenRouter models behind paid_models_enabled unless explicitly switched on.
        if (
            not getattr(self.settings, "paid_models_enabled", False)
            and "/" in raw_model
            and not raw_model.startswith("github-copilot/")
            and not raw_model.endswith(":free")
            and self.settings.provider != "ollama"
        ):
            raw_model = str(front_cfg.get("model") or "github-copilot/gpt-6-luna")
        model = raw_model
        if "/" in model and not model.startswith("github-copilot/"):
            head = model.split("/", 1)[0]
            if (
                head not in {"ollama", "openrouter", "github-copilot"}
                and self.settings.provider == "openrouter"
            ):
                model = f"openrouter/{model}"
        elif self.settings.provider == "github-copilot" and not model.startswith(
            "github-copilot/"
        ):
            model = f"github-copilot/{model}"
        elif self.settings.provider == "openrouter" and "/" not in model:
            model = f"openrouter/{model}"
        thinking = str(front_cfg.get("thinking") or "medium").lower()
        dec = self._last_decision or self.classify_decision(text).to_dict()
        dec_route = str(dec.get("route") or "front")
        dec_mode = str(dec.get("suggested_mode") or "general")
        dec_conf = float(dec.get("confidence") or 0.0)
        dec_ms = float(dec.get("latency_ms") or 0.0)
        if dec_route == "front":
            level = str(getattr(self.settings, "chat_thinking", "off") or "off").lower()
            if level in {"off", "low", "medium", "high"}:
                thinking = level
        pane = self.pi_pane("front")
        pane.configure(model=model, thinking=thinking)

        memories = ""
        with contextlib.suppress(Exception):
            memories = self._toolbox.relevant_facts(text)

        prompt_parts = [text]
        if memories:
            prompt_parts.append(memories)
        if self._session_profile:
            prompt_parts.append(
                "Owner profile (learned this session):\n"
                + "\n".join(f"- {f}" for f in self._session_profile[-10:])
            )
        if self._recent_task_reports:
            prompt_parts.append(
                "Recent specialist reports / Doctor updates:\n"
                + "\n".join(f"- {r}" for r in self._recent_task_reports[-5:])
            )
        if self.turns:
            recent = self.turns[-4:]
            recap = "\n".join(
                f"User: {t.get('user_text', '')[:120]}\nAssistant: {t.get('assistant_text', '')[:160]}"
                for t in recent
                if t.get("user_text")
            )
            if recap:
                prompt_parts.append("Recent conversation:\n" + recap)
        if getattr(self.settings, "front_only", False):
            prompt_parts.append(
                "(FRONT-ONLY MODE ACTIVE: You are the ONLY agent allowed to answer. Answer directly using your own reasoning and available tools — do NOT call delegate_task, ask_agent, or start_crew.)"
            )
        elif dec_route == "doctor":
            prompt_parts.append(
                f"(decision classifier: route=doctor, conf={dec_conf:.2f}, {dec_ms:.2f}ms -> hand off immediately to 'doctor' via ask_agent or delegate_task; do NOT handle diagnostics yourself)"
            )
        elif dec_route == "back":
            prompt_parts.append(
                f"(decision classifier: route=back, mode={dec_mode}, conf={dec_conf:.2f}, {dec_ms:.2f}ms -> say 'Checking ...' and hand off immediately to role='back' via delegate_task; do NOT ask multiple questions or do multi-step work yourself)"
            )
        else:
            prompt_parts.append(
                f"(decision classifier: route=front (+10% bias), mode={dec_mode}, conf={dec_conf:.2f}, {dec_ms:.2f}ms -> reply directly in max 2 short natural sentences; never read markdown or URLs aloud)"
            )
        full_prompt = "\n\n".join(prompt_parts)
        with self.lock:
            self._last_context_chars = len(full_prompt)

        def on_delta(piece: str) -> None:
            if self._first_token_at <= 0.0:
                self._first_token_at = time.perf_counter()
            with self.lock:
                if self._live_turn is not None:
                    self._live_turn["partial"] = (self._live_turn["partial"] + piece)[
                        -2000:
                    ]
            if pump is not None and splitter is not None and budget is not None:
                for sentence in splitter.feed(piece):
                    spoken_piece = budget.take(sentence)
                    if spoken_piece:
                        pump.push(spoken_piece)

        def on_think(chunk: str) -> None:
            self._think(f"[front CoT] {chunk[:240]}", "think")

        res = pane.ask_rich(
            full_prompt,
            timeout_s=180.0,
            on_delta=on_delta,
            on_think=on_think,
        )
        pane.last_tokens = int(res.tokens or 0)
        if pump is not None and splitter is not None and budget is not None:
            tail = splitter.flush()
            if tail:
                spoken_tail = budget.take(tail)
                if spoken_tail:
                    pump.push(spoken_tail)

        traces = [
            ToolTrace(
                name=str(tc.get("name") or "tool"),
                arguments=dict(tc.get("arguments") or {}),
                result=str(tc.get("result") or ""),
                ok=bool(tc.get("ok", True)),
                seconds=float(tc.get("seconds") or 0.0),
            )
            for tc in (res.tools or [])
        ]
        return AgentReply(
            text=str(res.text or "").strip(),
            tools=traces,
            llm_seconds=float(res.seconds or 0.0),
            rounds=1,
            model=str(res.model or model),
            tokens=int(res.tokens or 0),
            cost_usd=float(res.cost_usd or 0.0),
            reasoning=[],
        )

    # -------------------------------------------------------------- turn logic
    def ask(
        self, text: str, source: str = "text", asr_ms: float = 0.0
    ) -> dict[str, Any]:
        """Run one user turn and return the record that the UI renders."""
        text = text.strip()
        if not text:
            return {}

        # #1: If a Doctor change is awaiting exact confirmation, intercept before LLM.
        if text == CONFIRM_PHRASE and self._doctor_gate.pending() is not None:
            applied = self._doctor_gate.confirm_and_apply(text, self)
            msg = (
                f"Confirmed and applied: {applied.get('summary') or applied.get('message') or applied.get('applied')}"
                if applied.get("ok")
                else f"Could not apply: {applied.get('error')}"
            )
            return self._record_turn(
                text, source, msg, [], asr_ms, 0.0, 0.0, time.perf_counter()
            )

        with self.lock:
            self.status = "thinking"
            self.error = ""
            self._speech_cut = False
            now_pc = time.perf_counter()
            self._last_activity = now_pc
            self._thinking_since = now_pc
            if source == "text":
                self._utterance_end_at = time.perf_counter()
            self._live_turn = {
                "user_text": text[:400],
                "partial": "",
                "started": time.time(),
            }
        decision = self.classify_decision(text)
        with self.lock:
            self._turn_router_label = (
                "complex"
                if decision.route in {"back", "doctor"}
                else ("command" if decision.suggested_mode == "device" else "chat")
            )
        self._think(
            f"[classifier] route={decision.route} conf={decision.confidence:.2f} "
            f"bias=+10% ({decision.latency_ms:.2f}ms, {decision.reason_code})",
            "gate",
        )
        started = time.perf_counter()
        turn_gen = self._interrupt_seq
        self._turn_started_at = started
        self._last_filler_ms = 0.0
        self._first_token_at = 0.0
        turn_state = {"done": False}
        filler_timer = self._schedule_filler(text, turn_state)

        # M2.2 (D-2): streamed speaking. When on, the model's sentences are spoken
        # as they close while the rest of the reply is still generating - the
        # Pipecat behaviour: audio starts at the first sentence, not the last.
        pump: SpeechPump | None = None
        streamed = bool(
            self.settings.stream_voice
            and self.settings.tts_enabled
            and not self.settings.tts_simulate
        )

        def _on_first_sentence() -> None:
            # The real answer started: cancel the acknowledgement under the same
            # lock its fire() uses, so the two can never overlap.
            with self.lock:
                turn_state["done"] = True
            if filler_timer is not None:
                filler_timer.cancel()
            self._stop_filler()
            with self.lock:
                self.status = "speaking"
                self._speaking = True
                self._speech_started_at = time.perf_counter()

        def _spoke(piece: str) -> None:
            with self.lock:
                self._last_spoken = piece
                self._last_spoken_at = time.perf_counter()

        if streamed:
            try:
                pump = SpeechPump(self.tts(), _on_first_sentence, started, _spoke)
                self._pump = pump
            except TtsError:
                pump = None
        self._turn_thoughts = []
        self._turn_a2a_tools = []
        splitter = _SentenceSplitter()
        # Long-form requests (stories, poems, ...) are spoken in full — the short-reply
        # cap must not truncate them mid-sentence.
        spoken_cap = 0 if _wants_long_form(text) else self.settings.speech_cap_chars
        budget = SpeechBudget(spoken_cap)
        agent: Agent | None = None
        reply: Any = None
        used_pi_front = self._should_use_front_pi()
        try:
            try:
                if used_pi_front:
                    reply = self._ask_front_pi(
                        text,
                        pump=pump,
                        splitter=splitter,
                        budget=budget,
                    )
                else:
                    agent = self.agent()
                    with contextlib.suppress(Exception):
                        memories = self._toolbox.relevant_facts(text)
                        if memories:
                            agent.extra_context = memories
                    if pump is not None:
                        active_pump = pump

                        def on_delta(piece: str) -> None:
                            if self._first_token_at <= 0.0:
                                self._first_token_at = time.perf_counter()
                            with self.lock:
                                if self._live_turn is not None:
                                    self._live_turn["partial"] = (
                                        self._live_turn["partial"] + piece
                                    )[-2000:]
                            for sentence in splitter.feed(piece):
                                spoken_piece = budget.take(sentence)
                                if spoken_piece:
                                    active_pump.push(spoken_piece)

                        reply = agent.respond(
                            text, stream=True, on_delta=on_delta, raise_errors=True
                        )
                        tail = splitter.flush()
                        if tail:
                            spoken_tail = budget.take(tail)
                            if spoken_tail:
                                pump.push(spoken_tail)
                    else:
                        reply = agent.respond(text, raise_errors=True)
            except (
                LlmError,
                AsrError,
                PiRpcError,
                OSError,
                RuntimeError,
            ) as first_error:
                if pump is not None:
                    # A stream that died mid-way may have spoken a sentence or two;
                    # the fallback attempt starts clean and non-streamed.
                    pump.stop()
                    pump = None
                    self._pump = None
                if used_pi_front:
                    with self._pi_lock:
                        stale = self._pi_panes.pop("front", None)
                    if stale is not None:
                        with contextlib.suppress(Exception):
                            stale.stop()
                turn_state["done"] = True
                if filler_timer is not None:
                    filler_timer.cancel()
                self._stop_filler()
                if self._interrupt_seq != turn_gen or self._speech_cut:
                    with self.lock:
                        self.status = "listening" if self.mode != "off" else "idle"
                        self._speaking = False
                    partial_said = pump.played_text() if pump is not None else ""
                    recorded_cut = (
                        f"{partial_said} (cut off)" if partial_said else "(cut off)"
                    )
                    return self._record_turn(
                        text, source, recorded_cut, [], asr_ms, 0.0, 0.0, started
                    )
                # W1.4 (D-12): a failed model no longer kills the turn. Walk the role's
                # fallback chain: each fallback gets a fresh pane/agent and one attempt.
                reply = None
                chain = self._role_fallback_chain()
                primary_model = self.settings.model
                appended = bool(
                    agent is not None
                    and agent.history
                    and agent.history[-1].get("role") == "user"
                    and str(agent.history[-1].get("content")) == text
                )
                with self.lock:
                    self.status = "thinking"
                self._think(
                    f"model failed ({str(first_error)[:80]}); trying {len(chain)} fallback(s)",
                    "fallback",
                )
                for fallback_model in chain:
                    if used_pi_front and (
                        fallback_model.startswith("github-copilot/")
                        or fallback_model.endswith(":free")
                    ):
                        try:
                            with self.lock:
                                self.settings.model = fallback_model
                            self._think(
                                f"pi fallback attempt: {fallback_model}", "fallback"
                            )
                            reply = self._ask_front_pi(text)
                            with self.lock:
                                self.settings.model = primary_model
                                self.error = ""
                            self._think(
                                f"pi fallback answered: {fallback_model}", "fallback"
                            )
                            log_event(
                                f"mid-turn pi fallback succeeded: {fallback_model}"
                            )
                            break
                        except (PiRpcError, LlmError, OSError, RuntimeError) as exc:
                            with self._pi_lock:
                                stale = self._pi_panes.pop("front", None)
                            if stale is not None:
                                with contextlib.suppress(Exception):
                                    stale.stop()
                            self._think(
                                f"pi fallback {fallback_model} failed ({str(exc)[:60]})",
                                "fallback",
                            )
                            reply = None
                            continue
                    if fallback_model.startswith("github-copilot/"):
                        continue
                    try:
                        with self.lock:
                            # Swap the client on the SAME agent: the conversation history
                            # survives the fallback instead of restarting empty.
                            self.settings.model = fallback_model
                        self._think(f"fallback attempt: {fallback_model}", "fallback")
                        agent = self.agent()
                    except (LlmError, AsrError) as exc:
                        self._think(
                            f"fallback {fallback_model} unusable ({str(exc)[:60]})",
                            "fallback",
                        )
                        reply = None
                        continue
                    need_append = not appended
                    try:
                        reply = agent.respond(
                            text, raise_errors=True, append_user=need_append
                        )
                        appended = True
                        with self.lock:
                            self.settings.model = primary_model
                            self.error = ""
                        self._think(f"fallback answered: {fallback_model}", "fallback")
                        log_event(f"mid-turn fallback succeeded: {fallback_model}")
                        break
                    except (LlmError, AsrError) as exc:
                        appended = appended or need_append
                        self._think(
                            f"fallback {fallback_model} failed ({str(exc)[:60]})",
                            "fallback",
                        )
                        reply = None
                if reply is None:
                    with self.lock:
                        self.settings.model = (
                            primary_model  # retry the real model next turn
                        )
                        self.status = "listening" if self.mode != "off" else "idle"
                        self.error = str(first_error)
                        self._speaking = False
                    # M5.2: a plain spoken-friendly sentence, never the raw error text.
                    return self._record_turn(
                        text,
                        source,
                        "I could not reach any model just now. Please try again in a moment.",
                        [],
                        asr_ms,
                        0.0,
                        0.0,
                        started,
                    )
        finally:
            turn_state["done"] = True
            if filler_timer is not None:
                with contextlib.suppress(Exception):
                    filler_timer.cancel()
            with self.lock:
                if self.status in {"thinking", "speaking"}:
                    self.status = "listening" if self.mode != "off" else "idle"
                self._speaking = False
                self._thinking_since = 0.0
                if self._live_turn is not None:
                    self._live_turn = None
                self._pump = None

        # Cost meter: usage comes back from the provider when it reports it.
        self._spend_tokens += int(getattr(reply, "tokens", 0) or 0)
        self._spend_usd += float(getattr(reply, "cost_usd", 0.0) or 0.0)
        for r_block in getattr(reply, "reasoning", []) or []:
            if str(r_block).strip():
                self._think(f"[front CoT] {str(r_block).strip()[:280]}", "think")

        if reply.error:
            # Never speak a model error: the mic would hear it and loop. This only
            # fires for a non-raising caller; the transcript still gets the plain
            # sentence instead of the raw provider text (M5.2).
            with self.lock:
                self.error = reply.text
            log_event(f"model error (not spoken): {reply.text[:120]}")
            reply.text = (
                "I could not reach the model just now. Please try again in a moment."
            )

        # The reply exists: stop any acknowledgement under the same lock fire() uses.
        with self.lock:
            turn_state["done"] = True
        if filler_timer is not None:
            filler_timer.cancel()
        self._stop_filler()

        tts_ms = 0.0
        stream_extras: dict[str, Any] = {}
        recorded = reply.text
        if self._interrupt_seq != turn_gen or self._speech_cut:
            spoken_cut = pump.played_text() if pump is not None else ""
            if spoken_cut:
                recorded = f"{spoken_cut} (cut off)"
            elif not recorded:
                recorded = "(cut off)"
            if pump is not None:
                pump.stop()
                self._pump = None
            with self.lock:
                self._speaking = False
                self.status = "listening" if self.mode != "off" else "idle"
        elif reply.text and self.settings.tts_enabled and not reply.error:
            if pump is not None:
                # M2.2: the sentences were already spoken while the model was still
                # generating. Let the queue drain, then record what was actually
                # said (a barge-in must not pretend the whole reply was spoken).
                pump.finish()
                tts_ms = pump.spoken_ms()
                spoken_text = pump.played_text()
                if pump.failed() and spoken_text.strip() != reply.text.strip():
                    # #63: a streamed sentence failed mid-way; finish with the fast voice.
                    with self.lock:
                        self.error = "neural TTS failed mid-stream; finishing with the fast voice"
                    log_event("streamed speech failed; finishing the reply with SAPI")
                    with contextlib.suppress(TtsError):
                        fallback_voice = (
                            SapiTts()
                            if not isinstance(self.tts(), SapiTts)
                            else self.tts()
                        )
                        fallback_voice.say(_clip_speech(reply.text, 600))
                        fallback_voice.wait(timeout_s=60)
                first_ms = pump.first_audio_ms()
                if first_ms > 0.0:
                    stream_extras["ttfa_ms"] = round(first_ms, 1)
                stream_extras["sentences"] = pump.sentence_marks()
                stream_extras["words"] = pump.word_marks()
                with self.lock:
                    self._speaking = False
                    self._last_spoken = spoken_text or reply.text
                    self._last_spoken_at = time.perf_counter()
                    self._deaf_until = time.perf_counter() + self._barge_gates()[2]
                if self._speech_cut and spoken_text:
                    recorded = spoken_text
                self._pump = None
            else:
                with self.lock:
                    self.status = "speaking"
                    self._speaking = True
                    self._speech_started_at = time.perf_counter()
                spoken = time.perf_counter()
                try:
                    voice = self.tts()
                    spoken_text = sanitize_for_speech(budget.take(reply.text))
                    if spoken_text:
                        voice.say(spoken_text)
                        voice.wait(timeout_s=90)
                except TtsError as exc:
                    # #63: two-stage fallback - if the neural voice cannot keep up
                    # (buffer underrun), this turn falls back to the faster SAPI voice
                    # instead of stalling, and the next turn returns to the neural voice.
                    with self.lock:
                        self.error = (
                            f"neural TTS failed ({exc}); retrying with the fast voice"
                        )
                    log_event(
                        f"neural voice failed ({exc}); falling back to SAPI for this turn"
                    )
                    try:
                        fallback = (
                            SapiTts()
                            if not isinstance(self.tts(), SapiTts)
                            else self.tts()
                        )
                        fallback.say(_clip_speech(reply.text, 600))
                        fallback.wait(timeout_s=60)
                    except TtsError as exc2:
                        with self.lock:
                            self.error = f"TTS failed twice: {exc2}"
                finally:
                    tts_ms = (time.perf_counter() - spoken) * 1000.0
                    with self.lock:
                        self._speaking = False
                        self._last_spoken = reply.text
                        self._last_spoken_at = time.perf_counter()
                        self._deaf_until = time.perf_counter() + self._barge_gates()[2]

        if pump is not None:
            # No speakable text (or an error reply): nothing queued should linger.
            pump.stop()
            self._pump = None
        with self.lock:
            self.status = "idle"
        combined_tools = [
            {
                "name": t.name,
                "arguments": t.arguments,
                "result": t.result,
                "ok": t.ok,
                "seconds": round(t.seconds, 3),
            }
            for t in reply.tools
        ] + list(self._turn_a2a_tools)
        return self._record_turn(
            text,
            source,
            self._speech_adjusted(recorded),
            combined_tools,
            asr_ms,
            reply.llm_seconds * 1000.0,
            tts_ms,
            started,
            model=reply.model,
            extra={
                **stream_extras,
                "tokens": int(getattr(reply, "tokens", 0) or 0),
                "tok_per_s": (
                    round(
                        float(getattr(reply, "tokens", 0) or 0)
                        / max(0.05, float(reply.llm_seconds or 0.0)),
                        1,
                    )
                    if getattr(reply, "tokens", 0) and getattr(reply, "llm_seconds", 0)
                    else 0.0
                ),
            },
        )

    def _incident(
        self,
        kind: str,
        message: str,
        severity: str = "error",
        where: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a structured incident AND keep the legacy error string.

        Every write goes through here so nothing is silent: the app shows it, the
        jsonl keeps it, and the doctor can read it back.
        """
        with self.lock:
            self.error = str(message)[:400]
        return self.incidents.record(
            kind, message, severity=severity, where=where, context=context
        )

    def agents_active(self) -> list[dict[str, Any]]:
        """#S4: which role agents are working right now (SONJ4 consolidated view)."""
        rows: list[dict[str, Any]] = []
        with self.lock:
            status = self.status
            speaking = self._speaking
            model = self.settings.model
        front_state = (
            "speaking" if speaking else ("thinking" if status == "thinking" else "idle")
        )
        rows.append({"role": "front", "state": front_state, "detail": model})
        clf = self.classifier()
        rows.append(
            {
                "role": "classifier",
                "state": "ready",
                "detail": f"non-LLM <100ms ({clf.stats.get('last_latency_ms', 0.0):.2f}ms, +10% front bias)",
            }
        )
        running: list[Any] = []
        try:
            running = [
                task
                for task in self.tasks_manager().list()
                if task.state in {"running", "queued"}
            ]
        except Exception:  # noqa: BLE001
            running = []
        with self._pi_lock:
            panes = dict(self._pi_panes)
        for slot in BACK_POOL:
            pane = panes.get(slot)
            pst = str(getattr(pane, "status", "idle") or "idle") if pane else "cold"
            slot_tasks = [
                t.id
                for t in running
                if any(
                    str(s.get("role") or "") == slot
                    for s in (getattr(t, "steps_plan", []) or [])
                )
            ]
            rows.append(
                {
                    "role": slot,
                    "state": f"{len(slot_tasks)} task"
                    if slot_tasks
                    else (pst if pst not in {"idle", "ready"} else "warm"),
                    "detail": ", ".join(slot_tasks[:2]) or pst,
                }
            )
        rows.append(
            {
                "role": "doctor",
                "state": self._doctor.state,
                "detail": self._doctor.last_agent,
            }
        )
        rows.append(
            {
                "role": "verifier",
                "state": "verifying" if self._doctor.state == "verifying" else "idle",
                "detail": f"{len(self._doctor.last_results)} repair(s) this session",
            }
        )
        rows.append(
            {
                "role": "voice",
                "state": "speaking" if speaking else "idle",
                "detail": self.settings.tts_voice or "sapi",
            }
        )
        return rows

    # ------------------------------------------------------- doctor (S2/S3)
    def diagnose(self, scope: str = "all", agent: str = "doctor") -> dict[str, Any]:
        """S2: the doctor's read-only pass, narrated by the chosen agent.

        Public on purpose: the desktop window runs in-process and must not reach
        into the private attribute, and every repair stays inside the catalogue.
        """
        return self._doctor.diagnose(str(scope or "all"), str(agent or "doctor"))

    def repair(self, repair_id: str, dry_run: bool = True) -> dict[str, Any]:
        """S3: one catalogue entry; the engine verifies and rolls back on failure."""
        return self._doctor.repair(str(repair_id or ""), bool(dry_run))

    def verify(self) -> dict[str, Any]:
        """S3: the independent verifier pass, callable on its own."""
        return self._doctor.verify()

    def doctor_status(self) -> dict[str, Any]:
        """S2/S4: what the Doctor pane shows without touching private state."""
        return self._doctor.status()

    def watch_tick(self) -> dict[str, Any] | None:
        """S4.1: the demo-mode watch, called by the idle watchdog."""
        if not getattr(self.settings, "doctor_watch", False):
            return None
        return self._doctor.watch_tick()

    def fix_top_finding(self, dry_run: bool = False) -> dict[str, Any]:
        """S4.1: one-click Fix now, through the same catalogue and verifier."""
        return self._doctor.fix_top_finding(dry_run)

    def escalate(self, question: str, evidence: str = "") -> dict[str, Any]:
        """S3.5: propose a code-level fix to the harness, behind the approval gate.

        The doctor never rewrites code. This queues the proposal as an approval row
        and records an incident, so "the app asked a person" is visible in the log.
        """
        return doctor_escalate(self, question, evidence)

    def write_incident_report(self) -> str:
        """S4.2: write what happened this session as out/sessions/<id>/incidents.md.

        Text only. The owner's rule stands: transcripts and diagnostics, never audio.
        """
        folder = self._session_dir
        if folder is None:
            return ""
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "incidents.md"
            findings = self._doctor.last_findings
            repairs = self._doctor.last_results
            lines = [
                f"# Session incidents - {folder.name}",
                "",
                f"- turns: {len(self.turns)}",
                f"- incidents recorded: {self.incidents.count()}",
                f"- repairs applied: {len(repairs)}",
                f"- doc_on: {self._doctor.enabled}, kill_switch: {self._doctor.kill_switch}",
                "",
                "## Repairs",
                "",
            ]
            lines += [
                f"- {row.get('id')} {row.get('title')}: "
                f"{'verified' if row.get('verified') else 'NOT verified'}"
                for row in repairs
            ] or ["- none this session"]
            lines += ["", "## Last diagnosis", ""]
            lines += [
                f"- {row.get('severity')}: {row.get('title')} "
                f"(repair: {row.get('repair') or 'needs a person'})"
                for row in findings
            ] or ["- nothing found"]
            lines += [
                "",
                "## Incident log",
                "",
                self.incidents.report(200) or "no incidents recorded",
                "",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")
            return str(path)
        except OSError as exc:
            log_event(f"could not write the incident report: {exc}")
            return ""

    def _record_turn(
        self,
        user_text: str,
        source: str,
        assistant_text: str,
        tools: list[dict[str, Any]],
        asr_ms: float,
        llm_ms: float,
        tts_ms: float,
        started: float,
        model: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total_ms = (time.perf_counter() - started) * 1000.0
        with self.lock:
            self._live_turn = None
            self._turn_id += 1
            turn = {
                "id": self._turn_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "source": source,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "tools": tools,
                "thinking": [t for t in self._turn_thoughts if t.get("kind") != "pane"],
                "timings": {
                    "asr_ms": round(asr_ms, 1),
                    "llm_ms": round(llm_ms, 1),
                    "tts_ms": round(tts_ms, 1),
                    "first_token_ms": round(self._first_token_ms(started), 1),
                    "total_ms": round(total_ms, 1),
                    "filler_ms": round(self._filler_latency_ms(), 1),
                    "reply_audio_ms": round(self._reply_audio_ms(started), 1),
                    # #53: TTFA = first audible audio this turn (spoken + measured);
                    # a true TTFAT (first answer token) needs a streamed LLM, noted in
                    # SONJ_STACK_LEARNINGS.md as the Pipecat metrics gap.
                    "ttfa_ms": round(self._reply_audio_ms(started), 1),
                    "tokens_out": int((extra or {}).get("tokens") or 0),
                    "tok_per_s": float((extra or {}).get("tok_per_s") or 0.0),
                    "context_chars": int(getattr(self, "_last_context_chars", 0)),
                    "context_tokens_est": int(
                        getattr(self, "_last_context_chars", 0) // 4
                    ),
                    "e2e_ms": (
                        round(
                            max(0.0, (started - self._utterance_end_at) * 1000.0)
                            + float(
                                (extra or {}).get("ttfa_ms")
                                or self._reply_audio_ms(started)
                            ),
                            1,
                        )
                        if self._utterance_end_at > 0
                        else 0.0
                    ),
                },
                "model": model or self._agent_label,
                "router": getattr(self, "_turn_router_label", ""),
            }
            if extra:
                # M2.3: streamed turns add per-sentence timing + approximate word
                # offsets, and override ttfa with the streamed first-audio stamp.
                turn["timings"].update(extra)
            self.turns.append(turn)
            if self._last_spoken:
                self._recent_spoken.append(self._last_spoken)
            if str(assistant_text).endswith("(cut off)"):
                self._gate_stats["cut_offs"] += 1
            if str(assistant_text).rstrip().endswith("?") and source in {"mic", "text"}:
                self._clarifications += 1
                self._last_turn_clarified = True
            elif source in {"mic", "text"}:
                self._last_turn_clarified = False
            self._thinking_since = 0.0
            if (
                not self._session_title
                and source in {"text", "mic"}
                and str(user_text or "").strip()
            ):
                self._session_title = self._derive_session_title(user_text)
        self._save_session_meta()
        self._persist_message(turn)
        self._maybe_learn(turn)
        log_event(
            f"turn {turn['id']} [{source}] asr={asr_ms:.0f} llm={llm_ms:.0f} "
            f"tts={tts_ms:.0f} total={total_ms:.0f} "
            f"first={self._filler_latency_ms():.0f} ttft={self._first_token_ms(started):.0f} ms | "
            f"{user_text[:50]!r} -> {assistant_text[:60]!r}"
        )
        return turn

    # ---------------------------------------------------------------- listening
    def set_input_device(self, index: int) -> dict[str, Any]:
        """A: switch the microphone; restart the listener if it was running."""
        with self.lock:
            listener = self._listener
            was_mode = (
                self.mode if (listener is not None and listener.is_alive()) else None
            )
            self.input_device = int(index)
        if was_mode:
            self.stop_listening()
            self.start_listening(was_mode)
        return {"ok": True, "input_index": self.input_device}

    def start_listening(self, mode: str, source: AudioSource | None = None) -> None:
        """Listen in 'continuous' (VAD driven) or 'ptt' mode.

        A source argument overrides the microphone: passing a ScriptedStream
        replays WAV files instead, which is how the scripted full-duplex test
        drives the real pipeline on a machine with no audio hardware.
        """
        with self.lock:
            current = self._listener
            already = current is not None and current.is_alive() and self.mode == mode
        if already and source is None:
            return
        if current is not None and current.is_alive():
            # Switching modes: stop the old listener and WAIT for it to release the
            # microphone. Without the join the new thread is refused, listening
            # silently stays off, and the app "does not respond to any voice".
            self._stop.set()
            current.join(timeout=2.5)
        if source is None:
            # S1: wait for background voice warm-up before opening the mic —
            # sherpa-onnx hot-init while the mic streams can segfault.
            self._prewarm_done.wait(timeout=30.0)
            # Warm the ASR before the mic opens so the first utterance after an idle
            # unload is not stuck cold-loading the model ("not responding immediately").
            # Loaded before streaming to stay clear of the sherpa hot-init race.
            if self._asr is None:
                with contextlib.suppress(Exception):
                    self.asr()
        with self.lock:
            self.mode = mode
            self._stop.clear()
            self.error = ""
            # Reset block heartbeat BEFORE spawning so a slow VAD/ASR/mic init
            # cannot trip the 6s stall watchdog before the first block arrives.
            self._last_mic_block_at = time.monotonic()
            self._listener = threading.Thread(
                target=self._listen_loop, args=(mode, source), daemon=True
            )
            self._listener.start()

    def stop_listening(self) -> None:
        self._stop.set()
        with self.lock:
            self.mode = "off"
            self.status = "idle"

    def _listen_loop(self, mode: str, source: AudioSource | None = None) -> None:
        try:
            detector = self.vad()
            # The VAD object is reused across sessions, so clear whatever the last
            # one left buffered: stale audio used to be judged as an interruption
            # the moment a new listening session started.
            detector.reset()
            mic = source if source is not None else MicStream(device=self.input_device)
            mic.start()
        except (VadError, AudioError) as exc:
            with self.lock:
                self.error = str(exc)
                self.mode = "off"
            return

        self._mic = mic
        self._mic_is_scripted = source is not None
        self._last_mic_block_at = time.monotonic()
        buffer: list[np.ndarray] = []
        live_ring: list[np.ndarray] = []
        bg_ring: list[np.ndarray] = []
        bg_scan_at = time.monotonic()
        with self.lock:
            self.status = "listening"

        try:
            while not self._stop.is_set():
                block = mic.read(timeout=0.3)
                if block is None:
                    if not self._mic_is_scripted and not mic.active:
                        log_event(
                            "mic stream became inactive — exiting listener for recovery"
                        )
                        break
                    with self.lock:
                        if self._speaking:
                            self._last_mic_block_at = time.monotonic()
                    continue

                self._last_mic_block_at = time.monotonic()
                self._stall_streak = 0
                with self.lock:
                    blk_rms = float(np.sqrt(np.mean(np.square(block))) or 0.0)
                    self.level = blk_rms
                    speaking_now = self._speaking
                if not speaking_now and not self.recording:
                    self._voiceprint.update_noise_floor(blk_rms)
                if speaking_now:
                    self._echo_peak_rms = max(self._echo_peak_rms * 0.96, blk_rms)
                else:
                    self._echo_peak_rms = 0.0

                if mode == "ptt":
                    buffer.append(block)
                    with self.lock:
                        self.recording = True
                    if not self._mic_is_scripted and len(buffer) >= 10:
                        self._kick_live_asr(np.concatenate(buffer[-560:]))
                    continue

                # While the agent speaks, keep VAD and echo DSP IDLE: running
                # sherpa VAD + numpy correlation concurrently with TTS generation
                # and playback intermittently segfaults the native audio stack.
                # Barge-in is judged on raw accumulated audio (RMS + duration +
                # transcript gates), so real interruptions still cut the reply.
                if speaking_now:
                    live_ring.clear()
                    with self.lock:
                        self.recording = True
                    if not self.settings.barge_in:
                        self._speech_run.clear()
                        continue
                    # Pure-NumPy TTS-reference AEC + telecon NS run live during playback:
                    # cancels our own speaker echo before ASR/barge-in while keeping double-talk intact.
                    if self.settings.echo_cancellation:
                        block, was_echo = self._echo.process(block)
                        if was_echo:
                            with self.lock:
                                self.echo_drops += 1
                    if getattr(self.settings, "noise_suppression", True):
                        block = self._ns.process(block)
                    rms_blk = float(np.sqrt(np.mean(np.square(block))) or 0.0)
                    if rms_blk >= 0.014:
                        self._speech_run.append(block)
                        self._barge_silence = 0
                    else:
                        self._barge_silence += 1
                        if self._barge_silence > 12:
                            self._speech_run.clear()
                            self._barge_burst_counted = False
                            self._barge_judged_at = 0.0
                    if self._speech_run:
                        partial = np.concatenate(self._speech_run[-64:])
                        self._judge_interruption(partial)
                    continue

                self._speech_run.clear()
                self._barge_burst_counted = False
                self._barge_judged_at = 0.0
                self._barge_silence = 0

                if self.settings.echo_cancellation:
                    block, was_echo = self._echo.process(block)
                    if was_echo:
                        with self.lock:
                            self.echo_drops += 1
                        continue
                if getattr(self.settings, "noise_suppression", True):
                    block = self._ns.process(block)

                # Background room-voice scanner: runs only when Voices dashboard is open
                # (seen within 5s) OR manual toggle bg_voice_scan_always is ON, and someone
                # is actually speaking in the room (0% extra CPU in silence / other views).
                if self._voiceprint.model_present:
                    want_bg = bool(
                        getattr(self.settings, "bg_voice_scan_always", False)
                    ) or ((time.monotonic() - self._voices_view_at) < 5.0)
                    if want_bg:
                        bg_ring.append(block)
                        if len(bg_ring) > 40:
                            bg_ring = bg_ring[-40:]
                        now_bg = time.monotonic()
                        if (
                            (now_bg - bg_scan_at) >= 0.9
                            and not self._bg_scan_busy
                            and len(bg_ring) >= 16
                        ):
                            chunk = np.concatenate(bg_ring[-32:])
                            rms_bg = float(np.sqrt(np.mean(np.square(chunk))) or 0.0)
                            bg_scan_at = now_bg
                            if rms_bg >= 0.022:
                                self._bg_scan_busy = True

                                def _scan_bg(arr: np.ndarray = chunk) -> None:
                                    try:
                                        self._voiceprint.observe_background(arr)
                                    finally:
                                        self._bg_scan_busy = False

                                threading.Thread(
                                    target=_scan_bg, daemon=True, name="bg-voice"
                                ).start()
                    else:
                        bg_ring.clear()

                detector.accept(block)
                speech_now = bool(detector.is_speech())
                with self.lock:
                    self.recording = speech_now

                if speech_now:
                    live_ring.append(block)
                    if len(live_ring) > 560:
                        live_ring = live_ring[-560:]
                    if (
                        self.status == "thinking"
                        and getattr(self.settings, "instant_barge", False)
                        and self.settings.barge_in
                    ):
                        self._think_speech_ms += len(block) / 16.0
                        if self._think_speech_ms >= 190.0 and blk_rms >= max(
                            0.024, self.settings.barge_rms * 0.5
                        ):
                            log_event(
                                f"instant live barge while thinking ({self._think_speech_ms:.0f}ms, rms={blk_rms:.3f}) — aborting LLM stream"
                            )
                            self._think_speech_ms = 0.0
                            self.interrupt()
                    else:
                        self._think_speech_ms = 0.0
                    if not self._mic_is_scripted:
                        partial_audio = detector.partial()
                        if partial_audio is None or len(partial_audio) < 4800:
                            partial_audio = (
                                np.concatenate(live_ring)
                                if len(live_ring) >= 10
                                else None
                            )
                        if partial_audio is not None:
                            self._kick_live_asr(partial_audio)
                else:
                    live_ring.clear()
                    self._think_speech_ms = 0.0

                emitted_seg = False
                for segment in detector.segments():
                    emitted_seg = True
                    if time.perf_counter() < self._deaf_until:
                        # The echo tail of our own reply: dropping it here is what
                        # stops the assistant from answering itself.
                        continue
                    rms = float(np.sqrt(np.mean(np.square(segment))) or 0.0)
                    segment_ms = len(segment) / 16.0
                    if self._speaking:
                        if time.perf_counter() < getattr(self, "_protected_until", 0.0):
                            # Protected announce delivery: ignore barge material.
                            continue
                        if self.settings.echo_cancellation and self._echo.is_echo(
                            segment
                        ):
                            with self.lock:
                                self.echo_drops += 1
                            continue
                        playback_ms = (
                            time.perf_counter() - self._speech_started_at
                        ) * 1000.0
                        threshold, min_ms, _tail = self._barge_gates()
                        if self.settings.barge_in and should_interrupt(
                            rms=rms,
                            segment_ms=segment_ms,
                            playback_ms=playback_ms,
                            threshold=threshold,
                            min_segment_ms=min_ms,
                        ):
                            log_event(
                                f"barge-in accepted (rms={rms:.3f}, {segment_ms:.0f} ms "
                                f"speech, {playback_ms:.0f} ms into the reply)"
                            )
                            self.interrupt()
                            self._pending.put(("speech", segment))
                        else:
                            with self.lock:
                                self.ignored_while_speaking += 1
                            log_event(
                                f"ignored while speaking (rms={rms:.3f}, {segment_ms:.0f} ms, "
                                f"{playback_ms:.0f} ms in) - likely own speaker"
                            )
                        continue
                    if (
                        self.status == "thinking"
                        and getattr(self.settings, "instant_barge", False)
                        and self.settings.barge_in
                        and rms >= max(0.03, self.settings.barge_rms)
                    ):
                        log_event(
                            "instant barge while thinking — aborting LLM stream to listen"
                        )
                        self.interrupt()
                    self._pending.put(("speech", segment))

                if (
                    not speech_now
                    and not emitted_seg
                    and self.status == "listening"
                    and (time.perf_counter() - self._live_asr_at) > 1.2
                ):
                    self._clear_speaking_live_turn()

            if mode == "ptt" and buffer:
                samples = np.concatenate(buffer)
                detector.reset()
                for start in range(0, len(samples), detector.window_size):
                    detector.accept(samples[start : start + detector.window_size])
                segments = detector.flush()
                if not segments:
                    segments = [samples]
                longest = max(segments, key=len)
                self._pending.put(("speech", longest))
        finally:
            mic.stop()
            self._mic = None
            if mode != "ptt":
                self._clear_speaking_live_turn()
            with self.lock:
                self.recording = False
                if self.mode != "off":
                    self.mode = "off"
                self.status = "idle"

    def release_idle_models(self, force: bool = False) -> list[str]:
        """Give memory back when nothing is happening.

        Frees the offline ASR/VAD objects and asks Ollama to evict its model, so
        an idle app is not sitting on several hundred megabytes. Everything
        reloads lazily on the next turn.
        """
        with self.lock:
            idle_for = time.perf_counter() - self._last_activity
            if not force and idle_for < self.settings.idle_unload_s:
                return []
            busy = (
                self._speaking
                or self.mode != "off"
                or getattr(self, "_recovering_audio", False)
            )
            if busy and not force:
                return []

            freed: list[str] = []
            if self._asr is not None:
                freed.append(f"asr:{self._asr_backend}")
                self._asr = None
                self._asr_backend = ""
            if self._vad is not None:
                freed.append("vad")
                self._vad = None
            provider = self.settings.provider

        if provider == "ollama":
            try:
                for entry in loaded_ollama_models():
                    name = str(entry.get("name", ""))
                    if name and unload_ollama_model(name):
                        freed.append(f"ollama:{name}")
            except LlmError as exc:
                log_event(f"could not unload models: {exc}")

        if freed:
            log_event(
                f"released idle memory after {idle_for:.0f}s idle: {', '.join(freed)}"
            )
        return freed

    def _idle_watch_loop(self) -> None:
        """Check twice a minute whether the app can give memory back.

        S4.1 rides the same loop: in demo mode the doctor also looks for trouble,
        detect-only, so a failure during a demo becomes a visible banner.
        """
        while not getattr(self, "_closing", False):
            time.sleep(30)
            if getattr(self, "_closing", False):
                return
            try:
                self.release_idle_models()
            except Exception as exc:  # noqa: BLE001 - watchdog must not die
                log_event(f"idle watchdog error: {type(exc).__name__}: {exc}")
            try:
                self.watch_tick()
            except Exception as exc:  # noqa: BLE001 - watching must not die
                log_event(f"doctor watch error: {type(exc).__name__}: {exc}")

    def _recovery_and_announce_loop(self) -> None:
        """#9 Auto-recovery for hung turns/mic + #13 Idle task-completion voice report."""
        while not getattr(self, "_closing", False):
            time.sleep(0.5)
            if getattr(self, "_closing", False):
                return
            # #2: Doctor 15s auto-confirm pop-up timeout check
            try:
                auto_applied = self._doctor_gate.auto_confirm_if_expired(self)
                if auto_applied:
                    self._notify_doctor_outcome(auto_applied, source="auto-timeout")
            except Exception as exc:  # noqa: BLE001
                log_event(f"doctor auto-confirm error: {type(exc).__name__}: {exc}")

            # #9a: Hung thinking turn (> 180 s) -> abort pane + reset to idle
            try:
                with self.lock:
                    stuck = (
                        self.status == "thinking"
                        and self._thinking_since > 0
                        and (time.perf_counter() - self._thinking_since) > 180.0
                    )
                if stuck:
                    log_event("watchdog: turn stuck >180s — recovering to idle")
                    self.incidents.record(
                        "watchdog",
                        "turn hung >180s; aborted front pane",
                        "warning",
                        where="watchdog",
                    )
                    with contextlib.suppress(Exception):
                        self.pi_abort("front")
                    with self.lock:
                        self.status = "listening" if self.mode != "off" else "idle"
                        self._live_turn = None
                        self._thinking_since = 0.0
                        self.error = "Recovered from a hung turn (>180s)."
            except Exception as exc:  # noqa: BLE001
                log_event(f"recovery watchdog error: {type(exc).__name__}: {exc}")

            # #9b: Dead listener thread OR stalled mic stream (>6s silent heartbeat) -> recover
            # Cooldown (>=10s) prevents the 0.5s watchdog from self-killing a fresh listener
            # before its VAD/ASR/PortAudio stream finishes opening.
            try:
                now_mono = time.monotonic()
                with self.lock:
                    want_mode = self.mode
                    lst = self._listener
                    speaking_now = self._speaking
                    cooldown_ok = (now_mono - self._last_recovery_at) >= 10.0
                    dead = want_mode in {"continuous", "ptt"} and (
                        lst is None or not lst.is_alive()
                    )
                    stalled = (
                        want_mode in {"continuous", "ptt"}
                        and not speaking_now
                        and not self._mic_is_scripted
                        and (now_mono - self._last_mic_block_at) > 6.0
                    )
                if (dead or stalled) and cooldown_ok:
                    self._recovering_audio = True
                    self._last_recovery_at = now_mono
                    self._stall_streak = self._stall_streak + 1 if stalled else 0
                    why = "stopped" if dead else "stalled >6s"
                    try:
                        # If the current mic stalled twice in a row with no audio blocks
                        # or is no longer openable, probe PortAudio for a working device
                        # that actually delivers real audio callbacks (no 0<->1 ping-pong).
                        prev_dev = self.input_device
                        if self._stall_streak >= 2 or (
                            self.input_device is not None
                            and not can_open_input(self.input_device)
                        ):
                            candidates: list[int] = []
                            fb_in, _ = default_devices()
                            if fb_in is not None and fb_in != self.input_device:
                                candidates.append(fb_in)
                            for cand in input_devices():
                                if (
                                    cand.index != self.input_device
                                    and cand.index not in candidates
                                ):
                                    candidates.append(cand.index)
                            for cand_idx in candidates:
                                if probe_input_blocks(cand_idx, timeout_s=0.35):
                                    log_event(
                                        f"watchdog: fallback mic {self.input_device} -> {cand_idx}"
                                    )
                                    self.input_device = cand_idx
                                    break
                        log_event(
                            f"watchdog: audio listener {why}; recovering ({want_mode}, dev={self.input_device})"
                        )
                        self._gate_stats["recoveries"] += 1
                        if (
                            self.input_device != prev_dev
                            or (now_mono - self._last_watchdog_inc_at) >= 120.0
                        ):
                            self._last_watchdog_inc_at = now_mono
                            self.incidents.record(
                                "watchdog",
                                f"audio listener {why}; restarted (dev={self.input_device})",
                                "warning",
                                where="audio-watchdog",
                            )
                        if stalled:
                            self.stop_listening()
                        self.start_listening(want_mode)
                    finally:
                        self._recovering_audio = False
            except Exception as exc:  # noqa: BLE001
                self._recovering_audio = False
                log_event(f"listener recovery failed: {type(exc).__name__}: {exc}")

            # W2: flush an incomplete held thought after ~2.5s with no continuation,
            # so a trailing fragment is answered instead of hanging (semantic endpointing).
            try:
                flush_text = ""
                with self.lock:
                    if self._held_text and (time.perf_counter() - self._held_at) > 2.5:
                        flush_text, self._held_text = self._held_text, ""
                        self._held_at = 0.0
                if flush_text:
                    self._think(
                        f"held thought flushed after pause: {flush_text[-40:]!r}",
                        "turn",
                    )
                    self._pending.put(("text", flush_text))
            except Exception as exc:  # noqa: BLE001
                log_event(f"held-flush error: {type(exc).__name__}: {exc}")

            # #13: Front speaks task completion iff idle/available
            try:
                self._announce_pending_once()
            except Exception as exc:  # noqa: BLE001
                log_event(f"announcement error: {type(exc).__name__}: {exc}")

    def _announce_pending_once(self) -> bool:
        """Speak at most one queued task report when Front is free.

        Returns True when a report was delivered (spoken or posted). A noisy room
        (recording=True) used to starve announcements forever; after 12 s the owner
        hears the outcome regardless. A failed primary voice falls back to SAPI, and
        the announce turn is always persisted so the result is never lost.
        """
        now = time.perf_counter()
        with self.lock:
            if not self._pending_announcements:
                return False
            head = self._pending_announcements[0]
            waited = now - float(head.get("queued_at", now))
            front_free = self.status in {"idle", "listening"} and not self._speaking
            pause_ok = (now - self._last_user_speech_at) > 1.2
            may_speak = front_free and (
                (pause_ok and not self.recording) or waited > 12.0
            )
            if not may_speak:
                return False
            ann = self._pending_announcements.pop(0)

        summary = re.sub(r"[*`#_]+", "", str(ann.get("summary") or "")).strip()
        first_line = summary.splitlines()[0] if summary else "done"
        title = re.sub(r"[*`#_]+", "", str(ann.get("title") or "")).strip()
        state_kind = str(ann.get("state") or "completed").lower()
        err_txt = re.sub(r"[*`#_]+", "", str(ann.get("error") or "")).strip()
        role = str(ann.get("role") or "agent")
        if state_kind == "failed":
            reason = err_txt or first_line
            spoken = (
                f"Task failed: {title[:80]}. {reason[:120]}"
                if title
                else f"Your {role} task failed: {reason[:160]}"
            )
        elif state_kind == "cancelled":
            spoken = (
                f"Task cancelled: {title[:80]}."
                if title
                else f"Your {role} task was cancelled."
            )
        else:
            spoken = (
                f"Task completed: {title[:80]}. {first_line[:120]}"
                if title
                else f"Your {role} task is done. {first_line[:180]}"
            )

        tts_ms = 0.0
        attempted = False
        speech_failed = False
        if (
            getattr(self.settings, "announce_task_completion", True)
            and self.settings.tts_enabled
        ):
            attempted = True
            started = time.perf_counter()
            try:
                spoken = sanitize_for_speech(spoken)
                tts = self.tts()
                with self.lock:
                    self._speaking = True
                    self._speech_started_at = time.perf_counter()
                    self._last_spoken = spoken
                    # Protect the first 2s so the outcome lands before any barge-in.
                    self._protected_until = time.perf_counter() + 2.0
                try:
                    tts.say(spoken)
                    tts.wait(timeout_s=25)
                except TtsError as exc:
                    log_event(f"announcement voice failed ({exc}); trying SAPI")
                    speech_failed = True
                    if not isinstance(tts, SapiTts):
                        with contextlib.suppress(TtsError):
                            fallback = SapiTts()
                            fallback.say(_clip_speech(sanitize_for_speech(spoken), 600))
                            fallback.wait(timeout_s=60)
                            speech_failed = False
            finally:
                tts_ms = (time.perf_counter() - started) * 1000.0
                with self.lock:
                    self._speaking = False
                    self._last_spoken_at = time.perf_counter()
                    self._deaf_until = time.perf_counter() + self._barge_gates()[2]
            if speech_failed:
                self.incidents.record(
                    "announce",
                    f"task {ann.get('task_id')} speech failed; result shown in chat",
                    "warning",
                    where="announce",
                )

        reported_tts = max(round(tts_ms, 1), 0.1) if attempted else 0.0
        # Always post Front's explicit report turn so the owner is never left hanging.
        with self.lock:
            self._turn_id += 1
            rep_turn = {
                "id": self._turn_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "source": "announce",
                "user_text": "",
                "assistant_text": spoken,
                "tools": [],
                "error": state_kind == "failed",
                "timings": {
                    "asr_ms": 0.0,
                    "llm_ms": 0.0,
                    "tts_ms": reported_tts,
                    "total_ms": reported_tts,
                    "filler_ms": 0.0,
                    "reply_audio_ms": reported_tts,
                },
                "model": "front",
            }
            self.turns.append(rep_turn)
        self._persist_message(rep_turn)
        return True

    def schedule_restart(self, http_server: Any = None) -> None:
        """Spawn a detached replacement engine (--force) and exit cleanly."""

        def _do() -> None:
            time.sleep(0.25)
            with self._pi_lock:
                panes = list(self._pi_panes.values())
                self._pi_panes.clear()
            for pane in panes:
                with contextlib.suppress(Exception):
                    pane.stop()
            _release_engine()
            if http_server is not None:
                with contextlib.suppress(Exception):
                    http_server.server_close()
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
                )
            with contextlib.suppress(Exception):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                out_f = (OUT_DIR / "server_run.txt").open("a", encoding="utf-8")
                err_f = (OUT_DIR / "server_err.txt").open("a", encoding="utf-8")
                subprocess.Popen(
                    [
                        sys.executable,
                        str(BASE / "server.py"),
                        "--port",
                        str(int(getattr(self, "_listen_port", 8766) or 8766)),
                        "--no-browser",
                        "--force",
                    ],
                    cwd=str(BASE),
                    stdin=subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    close_fds=True,
                    creationflags=flags,
                )
            time.sleep(0.15)
            with contextlib.suppress(Exception):
                os._exit(0)

        threading.Thread(target=_do, daemon=True, name="self-restart").start()

    def total_spend_summary(self) -> dict[str, Any]:
        """#5: Combined LLM + ElevenLabs character/USD spend."""
        from tts import cloud_chars

        el_chars = int(cloud_chars().get("elevenlabs", 0))
        amortize = bool(getattr(self.settings, "elevenlabs_amortize_spend", True))
        # Starter $5 / 40,000 chars = $0.000125/char
        el_usd = round(el_chars * 0.000125, 6) if amortize else 0.0
        llm_usd = float(self._spend_usd)
        return {
            "llm_tokens": int(self._spend_tokens),
            "llm_usd": round(llm_usd, 6),
            "elevenlabs_chars": el_chars,
            "elevenlabs_usd": el_usd,
            "total_usd": round(llm_usd + el_usd, 6),
        }

    def doctor_control(self, body: dict[str, Any]) -> dict[str, Any]:
        """#1: Doctor internal control — read-only full access; mutations require exact confirm."""
        action = str(body.get("action") or "").strip()
        key = str(body.get("key") or "").strip()
        val = body.get("value")
        summary = str(body.get("summary") or "").strip()

        if action == "inspect_state":
            return {
                "status": self.status,
                "mode": self.mode,
                "settings": self.settings.persistable(),
                "agents": load_agents_config(),
                "spend": self.total_spend_summary(),
                "quality": self.quality_stats(),
                "startup_check": getattr(self, "startup_check", {}),
            }
        if action == "inspect_tokens":
            return {
                "spend": self.total_spend_summary(),
                "elevenlabs": elevenlabs_usage(self.settings.elevenlabs_api_key),
                "panes": {
                    k: p.get("tokens", 0) for k, p in self.pi_panes_snapshot().items()
                },
            }
        if action == "inspect_logs":
            log_path = OUT_DIR / "app.log"
            lines = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -50:
                ]
                if log_path.is_file()
                else []
            )
            return {"app_log_tail": lines, "incidents": self.incidents.snapshot(15)}
        if action == "performance_review":
            return performance_review(self)
        if action == "system_health":
            return system_health()
        if action == "feature_distribution":
            return feature_distribution(self)
        if action == "introspection":
            return app_introspection(self)
        if action == "settings_catalog":
            return settings_catalog(self)
        if action == "inspect_architecture":
            return inspect_architecture(self)
        if action == "app_guide":
            return {
                "views": [
                    "Dashboard",
                    "Chat",
                    "Sessions",
                    "Files",
                    "Sources",
                    "Agents",
                    "Documentation",
                    "Pipeline",
                    "Performance",
                    "Voices",
                    "Thinking",
                    "Settings",
                    "System Inspector (left rail bottom)",
                ],
                "wake_words": ["Nova", "Piattos"],
                "tips": [
                    "Click Pipeline (⇶) to swap VAD/ASR/LLM/TTS backends (local or cloud).",
                    "Click Performance (📈) for live latency blocks, per-feature CPU/RAM, and ElevenLabs credit score.",
                    "Top-right wake toggle turns wake-word + speaker lock ON/OFF.",
                    "Doctor setting proposals pop up a 15-second auto-confirm prompt (Confirm Now or Cancel).",
                    "Left rail bottom scrolls to open read-only System Prompts, Context File, Skill Directory, and Tool Definitions.",
                ],
            }
        if action == "propose_setting":
            if not key or not hasattr(self.settings, key):
                return {"ok": False, "error": f"unknown setting: {key}"}
            if key in {
                "openrouter_key",
                "elevenlabs_api_key",
                "deepgram_api_key",
                "openai_api_key",
            }:
                return {"ok": False, "error": "API keys cannot be changed via Doctor"}
            res = self._doctor_gate.propose(
                "set_settings", {key: val}, summary or f"Set {key} = {val!r}"
            )
            if res.get("ok"):
                self._think(
                    f"doctor -> front [PROPOSAL 15s]: {summary or f'Set {key} = {val!r}'}",
                    "a2a",
                )
            return res
        if action == "propose_settings":
            updates = val if isinstance(val, dict) else body.get("updates")
            if not isinstance(updates, dict) or not updates:
                return {
                    "ok": False,
                    "error": "value must be a non-empty {key: value} dict",
                }
            secrets = {
                "openrouter_key",
                "elevenlabs_api_key",
                "deepgram_api_key",
                "openai_api_key",
            }
            bad = [k for k in updates if not hasattr(self.settings, k) or k in secrets]
            if bad:
                return {
                    "ok": False,
                    "error": f"disallowed or unknown setting(s): {bad}",
                }
            desc = summary or (
                "Set " + ", ".join(f"{k}={v!r}" for k, v in updates.items())
            )
            res = self._doctor_gate.propose("set_settings", dict(updates), desc)
            if res.get("ok"):
                self._think(f"doctor -> front [PROPOSAL 15s]: {desc}", "a2a")
            return res
        if action == "propose_role_model":
            if not key or not val:
                return {"ok": False, "error": "key (role) and value (model) required"}
            res = self._doctor_gate.propose(
                "set_role_model",
                {"role": key, "model": str(val)},
                summary or f"Switch {key} model to {val}",
            )
            if res.get("ok"):
                self._think(
                    f"doctor -> front [PROPOSAL 15s]: Switch {key} model to {val}",
                    "a2a",
                )
            return res
        if action == "propose_restart":
            res = self._doctor_gate.propose(
                "restart", {}, summary or "Restart SONJ engine"
            )
            if res.get("ok"):
                self._think(
                    "doctor -> front [PROPOSAL 15s]: Restart SONJ engine", "a2a"
                )
            return res
        if action == "propose_reset_session":
            res = self._doctor_gate.propose(
                "reset_session", {}, summary or "Reset current conversation"
            )
            if res.get("ok"):
                self._think("doctor -> front [PROPOSAL 15s]: Reset session", "a2a")
            return res
        if action in {"edit_code", "write_file", "run_shell", "patch"}:
            return self._doctor_gate.propose(action, {}, summary)
        return {"ok": False, "error": f"unknown doctor_control action: {action}"}

    def _notify_doctor_outcome(
        self, outcome: dict[str, Any], source: str = "user"
    ) -> None:
        """Close the Doctor loop: record applied/cancelled outcome in A2A chat & Front memory."""
        if not outcome or not outcome.get("ok"):
            return
        if outcome.get("cancelled"):
            msg = f"Cancelled Doctor change: {outcome.get('summary') or 'proposal'}"
            self._think(f"front -> doctor [CANCELLED]: {msg}", "a2a")
        else:
            tag = (
                "AUTO-CONFIRMED (15s)"
                if outcome.get("auto_confirmed")
                else f"CONFIRMED ({source})"
            )
            detail = (
                outcome.get("summary")
                or outcome.get("message")
                or outcome.get("applied")
            )
            msg = f"Applied Doctor change ({tag}): {detail}"
            self._think(f"doctor -> front [{tag}]: {detail}", "a2a")
        with self.lock:
            self._recent_task_reports.append(f"[doctor] {msg}")
            if len(self._recent_task_reports) > 8:
                del self._recent_task_reports[:-8]
            self._turn_id += 1
            turn = {
                "id": self._turn_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "source": "announce",
                "user_text": "",
                "assistant_text": msg,
                "tools": [],
                "timings": {
                    "asr_ms": 0.0,
                    "llm_ms": 0.0,
                    "tts_ms": 0.0,
                    "total_ms": 0.0,
                    "filler_ms": 0.0,
                    "reply_audio_ms": 0.0,
                },
                "model": "doctor",
            }
            self.turns.append(turn)
        self._persist_message(turn)
        log_event(msg)

    def _think(self, msg: str, kind: str = "step") -> None:
        # Record one decision for the Thinking pane (newest last, capped).
        thoughts = getattr(self, "_thoughts", None)
        if thoughts is None:
            thoughts = []
            self._thoughts = thoughts
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with self.lock:
            entry = {"ts": stamp, "kind": kind, "msg": msg}
            thoughts.append(entry)
            if len(thoughts) > 240:
                del thoughts[: len(thoughts) - 240]
            self._turn_thoughts.append(entry)
            if len(self._turn_thoughts) > 120:
                del self._turn_thoughts[: len(self._turn_thoughts) - 120]
        self._persist_event(
            kind, {"msg": msg, "persona": getattr(self, "_persona", "nova")}
        )

    def _persist_event(self, kind: str, payload: dict) -> None:
        # Append one engine event to the session events log (best effort).
        if self._session_dir is None:
            return
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line = json.dumps(
                {"ts": stamp, "kind": kind, **payload}, ensure_ascii=False
            )
            path = self._session_dir / "events.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + chr(10))
        except OSError as exc:
            log_event("could not append event: " + str(exc))

    def _hears_own_voice(self, audio: np.ndarray) -> bool:
        """Is this audio our own reply arriving back through the microphone?

        Loudness cannot answer that: this apps own voice measured rms 0.065-0.081
        through this laptops microphone while the interruption threshold is 0.06, so a
        level-only decision cut replies off with the reply itself. Content can answer it,
        so the captured run is transcribed and compared with what is being said right
        now. Someone talking over the reply adds their own words, so the transcript no
        longer matches our sentence and their interruption still goes through.
        """
        with self.lock:
            spoken = self._last_spoken
        if not spoken:
            return False
        try:
            heard = self.asr().transcribe(np.asarray(audio, dtype=np.float32)).strip()
        except Exception as exc:  # noqa: BLE001 - a failed check must not eat an interruption
            log_event(
                f"own-voice check unavailable ({type(exc).__name__}: {exc}); "
                f"treating this as a person"
            )
            return False
        if not heard:
            # Nothing intelligible came back: not a person asking for something, since a
            # real interruption transcribes to words. Treating it as our own voice is
            # the safe direction - noise cannot cut a reply - and someone who actually
            # speaks is still judged on what they said. The verification audit found
            # that an empty transcript used to fall through and cut the reply anyway.
            with self.lock:
                self.ignored_while_speaking += 1
            log_event("kept speaking - no words in the captured audio, so not a person")
            return True
        if not is_echo(heard, spoken, recent=True):
            # Different words, so not our own voice. Protect against single non-stop
            # artifacts (like "of") and hallucinated noise. When instant_barge is ON
            # (and stop-word-only mode is OFF), any real multi-word non-echo speech
            # interrupts immediately — not only literal "stop" words.
            word = _heard_trigger(heard, tuple(self.settings.barge_triggers))
            if word is not None:
                log_event(f"stop word {word!r} heard - cutting the reply")
                self._think(f"you said {word!r} - stopping", "you")
                return False
            words_tok = [t for t in re.findall(r"[a-zA-Z0-9']+", heard) if t]
            instant = bool(getattr(self.settings, "instant_barge", False))
            if (
                instant
                and not self.settings.wake_to_interrupt
                and not self._is_noise_transcript(heard)
                and (len(words_tok) >= 2 or len(heard.strip()) >= 8)
            ):
                log_event(
                    f"instant speech barge accepted ({heard[:50]!r}) - cutting reply"
                )
                self._think(f"you spoke ({heard[:40]!r}) - stopping to listen", "you")
                return False
            with self.lock:
                self.ignored_while_speaking += 1
            log_event(f"kept speaking - heard {heard[:50]!r} but no stop word in it")
            self._think(f"kept speaking - {heard[:40]!r} has no stop word", "echo")
            return True
        with self.lock:
            self.echo_drops += 1
            self._barge_echo_checks += 1
        log_event(f"kept speaking - the microphone heard our own reply: {heard[:60]!r}")
        self._think(
            "kept speaking - the microphone heard our own reply, not you", "echo"
        )
        return True

    def _should_cut_for(
        self, audio: np.ndarray, rms: float, segment_ms: float, playback_ms: float
    ) -> bool:
        """Should this audio cut the reply short - a person, or our own voice?

        The one gate both microphone paths ask: the level guards first (grace window,
        minimum speech, loudness), then the content check. The live in-progress judge
        adds a re-check throttle on top; the finished-segment path asks this directly.
        They were separate decisions before, and the finished-segment path was still
        cutting replies on loudness alone - found by the independent audit.
        """
        if not self.settings.barge_in:
            return False
        if not should_interrupt(
            rms=rms,
            segment_ms=segment_ms,
            playback_ms=playback_ms,
            threshold=self.settings.barge_rms,
            min_segment_ms=self.settings.barge_min_ms,
        ):
            return False
        return not self._hears_own_voice(audio)

    def _barge_gates(self) -> tuple[float, float, float]:
        """Playback-aware barge-in gates (threshold, min_ms, deaf tail s).

        With echo cancellation on, the app's own voice (measured rms
        0.065-0.081 at the mic) must not qualify as an interruption: require
        louder, longer speech and a longer deaf tail after playback.
        """
        # When stop-word/wake-word barge is active, ASR trigger matching is the real
        # gate — allow normal speaking volume (0.06) so the user needn't shout "stop".
        if self.settings.wake_to_interrupt:
            # The ASR stop-word check is the real gate, so a quick "stop" (~300 ms)
            # qualifies — no need to demand a long utterance to interrupt.
            return (
                max(self.settings.barge_rms, 0.06),
                max(self.settings.barge_min_ms, 200.0),
                1.2,
            )
        if self.settings.echo_cancellation:
            mode = str(getattr(self.settings, "barge_strictness", "normal") or "normal")
            if mode == "relaxed":
                return (
                    max(self.settings.barge_rms * 0.7, 0.04),
                    max(self.settings.barge_min_ms, 150.0),
                    0.7,
                )
            if mode == "normal":
                return (
                    max(self.settings.barge_rms, 0.06),
                    max(self.settings.barge_min_ms, 220.0),
                    0.9,
                )
            return (
                max(self.settings.barge_rms, 0.10),
                max(self.settings.barge_min_ms, 380.0),
                1.1,
            )
        return self.settings.barge_rms, self.settings.barge_min_ms, ECHO_TAIL_S

    def _judge_interruption(self, audio: np.ndarray) -> None:
        """Decide whether in-progress speech is an interruption, and act on it.

        Called for every block while the agent is speaking, so the reaction time
        is the speech needed to pass the guards rather than the length of the
        whole utterance.
        """
        # Protected delivery window: while a task-completion announce is landing its
        # first "Task completed: {title}" line, no voice barge-in may cut it.
        if time.perf_counter() < getattr(self, "_protected_until", 0.0):
            return
        instant = bool(getattr(self.settings, "instant_barge", False))
        if not instant and not self.settings.wake_to_interrupt:
            # When Speak->Stop is explicitly toggled OFF, require a stop word instead of cutting on speech.
            pass
        rms = float(np.sqrt(np.mean(np.square(audio))) or 0.0)
        spoken_ms = len(audio) / 16.0
        playback_ms = (time.perf_counter() - self._speech_started_at) * 1000.0
        threshold, min_ms, _tail = self._barge_gates()
        # Adaptive echo-aware threshold: when open-mic barge is on, speech must beat speaker echo
        if (
            instant
            and not self.settings.wake_to_interrupt
            and self._echo_peak_rms > 0.0
        ):
            threshold = max(threshold, min(0.14, self._echo_peak_rms * 1.05))
        accepted = instant and should_interrupt(
            rms=rms,
            segment_ms=spoken_ms,
            playback_ms=playback_ms,
            threshold=threshold,
            min_segment_ms=min_ms,
        )
        if not accepted:
            # Stop-word fast path: a deliberate stop word ("stop", "cancel", "quiet")
            # ALWAYS interrupts, in any mode, even a quick/quiet one the strict gate
            # rejected. Permissive level gate + echo guard so it is reliable and the
            # agent's own reply (which may contain a stop word) never cuts itself.
            if should_interrupt(
                rms=rms,
                segment_ms=spoken_ms,
                playback_ms=playback_ms,
                threshold=0.035,
                min_segment_ms=150.0,
            ):
                _now_sw = time.perf_counter()
                if _now_sw - self._barge_judged_at >= BARGE_REJUDGE_S:
                    self._barge_judged_at = _now_sw
                    try:
                        _heard_sw = self.asr().transcribe(audio[-32000:]).strip()
                    except AsrError:
                        _heard_sw = ""
                    _sw = _heard_trigger(_heard_sw, tuple(self.settings.barge_triggers))
                    with self.lock:
                        _spoken_sw = self._last_spoken
                    if _sw is not None and not is_echo(
                        _heard_sw, _spoken_sw, recent=True
                    ):
                        log_event(f"stop word {_sw!r} heard - cutting the reply")
                        self._think(f"you said {_sw!r} - stopping", "you")
                        self.interrupt()
                        self._pending.put(("speech", audio))
                        self._barge_burst_counted = True
                        return
            if not self._barge_burst_counted:
                with self.lock:
                    self.ignored_while_speaking += 1
                self._barge_burst_counted = True
                log_event(
                    f"ignored while speaking (rms={rms:.3f}, {spoken_ms:.0f} ms, "
                    f"{playback_ms:.0f} ms in) - not interruption material"
                )
            return
        now = time.perf_counter()
        if now - self._barge_judged_at < BARGE_REJUDGE_S:
            return
        self._barge_judged_at = now
        heard_barge = ""
        if self.settings.wake_to_interrupt or self.settings.wake_speaker_lock:
            try:
                heard_barge = self.asr().transcribe(audio[-32000:])
            except AsrError:
                heard_barge = ""
        if self._hears_own_voice(audio):
            self._gate_stats["barge_rejected"] += 1
            return
        wake_barge = self._wake_ok(heard_barge)
        stop_word = _heard_trigger(heard_barge, tuple(self.settings.barge_triggers))
        if self.settings.wake_to_interrupt and (stop_word is None and not wake_barge):
            self._gate_stats["barge_rejected"] += 1
            return
        if (
            self.settings.wake_to_interrupt
            and self.settings.wake_speaker_lock
            and self._voiceprint.model_present
        ):
            cls_b = self._voiceprint.classify_and_track(
                audio[-32000:],
                is_wake=wake_barge,
                wake_lock_enabled=True,
                # Intentional stop words ("stop", "cancel", "quiet") always quiet an
                # active reply even if no wake master is currently active.
                require_wake_when_idle=(stop_word is None)
                and bool(getattr(self.settings, "require_wake_when_idle", True)),
            )
            if not cls_b.get("allowed", True):
                self._gate_stats["barge_rejected"] += 1
                log_event(f"barge rejected by wake-master lock ({cls_b.get('label')})")
                return
        if (
            self.settings.voiceprint_barge
            and self._voiceprint.enrolled
            and self._voiceprint.score(audio[-32000:]) < self._voiceprint.threshold
        ):
            self._gate_stats["barge_rejected"] += 1
            return
        log_event(
            f"barge-in accepted at {spoken_ms:.0f} ms of speech "
            f"({playback_ms:.0f} ms into the reply, rms={rms:.3f}, trigger={stop_word or 'open-mic'})"
        )
        self._think(
            f"you barged in ({stop_word or 'open-mic'}) at {playback_ms:.0f} ms",
            "you",
        )
        self.interrupt()
        self._pending.put(("speech", audio))
        self._barge_burst_counted = True

    def interrupt(self) -> None:
        """Barge-in: stop filler/TTS playback and active Front LLM stream, drop stale audio, listen immediately."""
        with self.lock:
            self._interrupt_seq += 1
            self.barge_ins += 1
            self._think_speech_ms = 0.0
            self._echo_peak_rms = 0.0
            if self._speaking or self.status == "thinking":
                self._speech_cut = True  # #41: mark turn as interrupted mid-reply
        with contextlib.suppress(Exception):
            self._stop_filler()
        if self._pump is not None:
            # M2.2: streamed speech is cut through the pump too, so nothing queued
            # keeps talking after the barge-in.
            self._pump.stop()
        try:
            self.tts().stop()
        except TtsError as exc:
            with self.lock:
                self.error = f"stop failed: {exc}"
        self._player.stop()
        with contextlib.suppress(Exception):
            self.pi_abort("front")
        with self.lock:
            self._speech_run.clear()
            self._held_text = ""
            self._held_at = 0.0
            self._live_turn = None
            self._speaking = False
            self._thinking_since = 0.0
            self.status = "listening" if self.mode != "off" else "idle"

    def _work_loop(self) -> None:
        """Serialises transcription and agent turns off the audio thread."""
        while not getattr(self, "_closing", False):
            kind, payload = self._pending.get()
            if kind == "shutdown" or getattr(self, "_closing", False):
                return
            try:
                if kind == "speech":
                    self._handle_speech(payload)
                elif kind == "text":
                    self.ask(str(payload), source="text")
                elif kind == "timer":
                    self._handle_timer(str(payload))
            except Exception as exc:  # noqa: BLE001 - this worker must never die
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.status = "idle"
                traceback.print_exc()

    @staticmethod
    def _is_noise_transcript(text: str) -> bool:
        """Whisper hallucinations of ambient noise, not words a user said.

        A USB mic left in the open hears wind and birds; Whisper transcribes
        those as "(wind blowing)", "(birds chirping)", "[BLANK_AUDIO]", "..." or
        a bare "Thank you." — and the agent then answers the weather. Garbage
        like "kll" (a single vowel-less token) is also noise, not speech.
        """
        stripped = text.strip()
        low = stripped.lower()
        tokens = [t for t in stripped.split() if t]
        return bool(
            stripped
            and (
                (stripped.startswith("(") and stripped.endswith(")"))
                or (stripped.startswith("[") and stripped.endswith("]"))
                or set(stripped) <= {".", "…", "-", " "}
                or low
                in {
                    "thank you.",
                    "thank you",
                    "thanks for watching!",
                    "thanks for watching.",
                    "thank you for watching",
                    "bye.",
                    "bye!",
                    "um",
                    "uh",
                    "hmm",
                }
                or (
                    len(tokens) == 1
                    and len(tokens[0]) <= 3
                    and not any(c in tokens[0].lower() for c in "aeiou")
                )
            )
        )

    def _clear_speaking_live_turn(self) -> None:
        """Remove an in-progress user-speech bubble if it did not become a turn."""
        with self.lock:
            if self._live_turn is not None and self._live_turn.get("speaking"):
                self._live_turn = None

    def _kick_live_asr(self, audio: np.ndarray) -> None:
        """Transcribe in-progress speech asynchronously so the chat updates live."""
        now = time.perf_counter()
        if self._live_asr_busy or (now - self._live_asr_at) < 0.45:
            return
        if len(audio) < 4800:
            return
        limit = int(ASR_MAX_AUDIO_S * 16000)
        chunk = np.asarray(audio[-limit:], dtype=np.float32).copy()
        self._live_asr_busy = True
        self._live_asr_at = now

        def _run_live() -> None:
            try:
                with contextlib.suppress(Exception):
                    engine = self.asr()
                    preview = engine.transcribe(chunk).strip()
                    if not preview or self._is_noise_transcript(preview):
                        return
                    with self.lock:
                        spoken = self._last_spoken
                        spoken_at = self._last_spoken_at
                    recent = (time.perf_counter() - spoken_at) < ECHO_WINDOW_S
                    if is_echo(preview, spoken, threshold=0.42, recent=recent) or any(
                        is_echo(preview, prev, threshold=0.45, recent=True)
                        for prev in getattr(self, "_recent_spoken", ())
                    ):
                        return
                    with self.lock:
                        if (self.recording or self.status == "transcribing") and (
                            self._live_turn is None or self._live_turn.get("speaking")
                        ):
                            self._live_turn = {
                                "user_text": preview[:400],
                                "partial": "",
                                "started": time.time(),
                                "speaking": True,
                            }
            finally:
                self._live_asr_busy = False

        threading.Thread(target=_run_live, daemon=True, name="live-asr").start()

    def _handle_speech(self, samples: np.ndarray) -> None:
        # The utterance ended when VAD delivered the segment: start the true
        # utterance-end → first-audio clock here (includes ASR + queue + LLM + TTS).
        self._utterance_end_at = time.perf_counter()
        with self.lock:
            self.status = "transcribing"
        try:
            engine = self.asr()
        except AsrError as exc:
            self._clear_speaking_live_turn()
            with self.lock:
                self.error = str(exc)
                self.status = "idle"
            return

        if self.settings.speech_rms_floor > 0.0:
            rms = float(np.sqrt(np.mean(np.square(samples))) or 0.0)
            if rms < self.settings.speech_rms_floor:
                log_event(
                    f"skipped quiet segment (rms={rms:.3f} < floor {self.settings.speech_rms_floor})"
                )
                self._clear_speaking_live_turn()
                with self.lock:
                    self.status = "listening" if self.mode != "off" else "idle"
                return
        started = time.perf_counter()
        # Whisper decodes at most 30 s and sherpa aborts the process on oversized pushes
        # (server_err.txt showed a circular-buffer overflow plus 'only waves less than 30
        # seconds are supported'), which is what kept killing the engine. Clamp here -
        # the last point before the native library sees the audio.
        limit = int(ASR_MAX_AUDIO_S * 16000)
        if len(samples) > limit:
            log_event(
                f"clamped {len(samples) / 16000:.1f}s of audio to {ASR_MAX_AUDIO_S:.0f}s for ASR"
            )
            samples = samples[-limit:]
        text = engine.transcribe(samples)
        asr_ms = (time.perf_counter() - started) * 1000.0
        with self.lock:
            spoken = self._last_spoken
            spoken_at = self._last_spoken_at
        # Audio arriving right after the agent stopped speaking gets the benefit of
        # a fuzzier match; later audio has to match the words closely to count.
        recent = (time.perf_counter() - spoken_at) < ECHO_WINDOW_S
        if text and (
            is_echo(text, spoken, threshold=0.42, recent=recent)
            or any(
                is_echo(text, prev, threshold=0.45, recent=True)
                for prev in getattr(self, "_recent_spoken", ())
            )
        ):
            with self.lock:
                self.echo_drops += 1
                self._gate_stats["echo_dropped"] += 1
            log_event(f"dropped echo of own speech: {text[:70]!r}")
            self._clear_speaking_live_turn()
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        if not text:
            self._clear_speaking_live_turn()
            # #43: noise with no transcript still gets one spoken prompt per burst,
            # so an interrupted turn never hangs silently (throttled, never a loop).
            # #43: one spoken "say that again" per noise burst, never a loop.
            if (
                self.settings.empty_turn_reply
                and self.mode == "continuous"
                and time.perf_counter() - self._last_empty_reply > 8.0
            ):
                self._last_empty_reply = time.perf_counter()
                # Register the prompt as spoken so the mic's echo of it is rejected
                # instead of becoming a new turn (it was polluting the wake gate:
                # "wake gate held: 'Sorry, I did not catch that...'").
                _prompt = "Sorry, I did not catch that. Say it again?"
                try:
                    _v = self.tts()
                    with self.lock:
                        self._speaking = True
                        self._speech_started_at = time.perf_counter()
                        self._last_spoken = _prompt
                    _v.say(_prompt)
                    _v.wait(timeout_s=6)
                except TtsError:
                    pass  # A failed prompt must not eat the turn.
                finally:
                    with self.lock:
                        self._speaking = False
                        self._last_spoken_at = time.perf_counter()
                        self._deaf_until = time.perf_counter() + self._barge_gates()[2]
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        with self.lock:
            self._live_turn = {
                "user_text": text[:400],
                "partial": "",
                "started": time.time(),
                "speaking": True,
            }
        # M2.5 (Pipecat smart-turn): if the previous utterance was held as an
        # unfinished thought, decide whether this one completes it.
        if self._held_text:
            held, self._held_text = self._held_text, ""
            if time.perf_counter() - self._held_at > 8.0:
                # Too stale to be one thought: answer it on its own first.
                self.ask(held, source="mic", asr_ms=0.0)
            else:
                text = (held + " " + text).strip()
        if (
            self.settings.smart_turn
            and self.mode == "continuous"
            and not self._thought_complete(text)
        ):
            self._held_text = text
            self._held_at = time.perf_counter()
            self._think(f"turn kept open (unfinished): {text[-48:]!r}", "turn")
            self._clear_speaking_live_turn()
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        if self.settings.noise_gate and self._is_noise_transcript(text):
            # Ambient-noise hallucinations must not become conversation turns
            # (a USB mic in the open transcribed wind as "(wind blowing)").
            log_event(f"dropped noise transcript: {text[:60]!r}")
            self._think(f"noise gate dropped {text[:40]!r}", "gate")
            self._clear_speaking_live_turn()
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        self._last_user_speech_at = time.perf_counter()
        self._gate_stats["vad_segments"] += 1
        is_wake = self._wake_ok(text)
        wake_on = bool(self.settings.wake_gate)
        wake_lock = bool(wake_on and getattr(self.settings, "wake_speaker_lock", True))
        idle_wake = bool(
            wake_on and getattr(self.settings, "require_wake_when_idle", True)
        )
        # #3/#6-add: multi-master wake lock + stay quiet until Nova when idle.
        # Only lock speakers or demand "Nova" when the top-right wake gate is ON;
        # when wake is OFF, any speaker is unlocked and no wake word is required.
        if self._voiceprint.model_present:
            cls = self._voiceprint.classify_and_track(
                samples,
                is_wake=is_wake,
                wake_lock_enabled=wake_lock,
                require_wake_when_idle=idle_wake,
            )
            if is_wake and wake_lock:
                self._think(
                    f"wake master -> {cls.get('master_label')} (5m idle lock)", "gate"
                )
            if wake_lock and not cls.get("allowed", True):
                self._gate_stats["wake_blocked"] += 1
                log_event(
                    f"wake-master lock rejected {cls.get('label')} "
                    f"(sim={cls.get('similarity')}, master={cls.get('master_label')}): {text[:40]!r}"
                )
                self._think(
                    f"ignored {cls.get('label')} — locked/awaiting Nova ({cls.get('master_label') or 'idle'})",
                    "gate",
                )
                self._clear_speaking_live_turn()
                with self.lock:
                    self.status = "listening" if self.mode != "off" else "idle"
                return
            self._gate_stats["wake_allowed"] += 1
        if wake_on and not is_wake:
            self._gate_stats["wake_blocked"] += 1
            log_event(f"wake gate held: {text[:60]!r}")
            self._think(f"held {text[:40]!r} - no wake phrase", "gate")
            self._clear_speaking_live_turn()
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        # Bare wake word ("Nova" / "Hey Nova" alone): fast cached ack, skip 5s LLM echo
        bare_clean = re.sub(r"[^a-z]+", "", text.lower())
        if is_wake and bare_clean in {
            "nova",
            "heynova",
            "oknova",
            "piattos",
            "heypiattos",
        }:
            self._think("wake ack -> listening", "gate")
            self._clear_speaking_live_turn()
            if self.settings.tts_enabled and not self.settings.tts_simulate:
                try:
                    bank = self._fillers
                    if bank is None or not bank.play("chat"):
                        v = self.tts()
                        with self.lock:
                            self._speaking = True
                            self._speech_started_at = time.perf_counter()
                            self._last_spoken = "Yes?"
                        v.say("Yes?")
                        v.wait(timeout_s=5)
                finally:
                    with self.lock:
                        self._speaking = False
                        self._last_spoken_at = time.perf_counter()
                        self._deaf_until = time.perf_counter() + self._barge_gates()[2]
            with self.lock:
                self.status = "listening" if self.mode != "off" else "idle"
            return
        if wake_on and self.settings.voiceprint_enabled and self._voiceprint.enrolled:
            score = self._voiceprint.score(samples)
            if score < self._voiceprint.threshold:
                self._gate_stats["wake_blocked"] += 1
                log_event(f"voiceprint rejected (score={score:.2f}): {text[:40]!r}")
                self._think(f"voiceprint rejected {score:.2f} - not the owner", "gate")
                self._clear_speaking_live_turn()
                with self.lock:
                    self.status = "listening" if self.mode != "off" else "idle"
                return
        # #2: Ask-Detector — drop empty backchannels unless a clarification is pending
        if getattr(self.settings, "ask_detector", True) and not is_wake:
            asking, why = is_asking(text, pending_clarify=self._last_turn_clarified)
            if not asking:
                self._gate_stats["ask_dropped"] += 1
                log_event(f"ask-detector dropped ({why}): {text[:40]!r}")
                self._think(f"ask-detector skipped {why}", "gate")
                self._clear_speaking_live_turn()
                with self.lock:
                    self.status = "listening" if self.mode != "off" else "idle"
                return
        self.ask(text, source="mic", asr_ms=asr_ms)
        with self.lock:
            self.status = "listening" if self.mode != "off" else "idle"

    def _handle_timer(self, message: str) -> None:
        with self.lock:
            self._turn_id += 1
            turn = {
                "id": self._turn_id,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "source": "timer",
                "user_text": "",
                "assistant_text": message,
                "tools": [],
                "timings": {
                    "asr_ms": 0.0,
                    "llm_ms": 0.0,
                    "tts_ms": 0.0,
                    "total_ms": 0.0,
                },
                "model": "timer",
            }
            self.turns.append(turn)
        if self.settings.tts_enabled:
            try:
                voice = self.tts()
                voice.say(message)
            except TtsError as exc:
                with self.lock:
                    self.error = f"TTS failed: {exc}"

    # ------------------------------------------------------------------- state
    def list_skills(self) -> list[dict]:
        # Skills are markdown instruction files under skills/ (loaded on demand).
        root = BASE / "skills"
        found = []
        if not root.is_dir():
            return found
        for path in sorted(root.glob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            title = next(
                (l.lstrip("# ").strip() for l in lines if l.startswith("# ")), path.stem
            )
            body = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
            found.append(
                {
                    "name": path.stem,
                    "title": title,
                    "description": body[0][:180] if body else "",
                    "path": str(path),
                }
            )
        return found

    def mcp_call(
        self, server: str, tool: str, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """R14: call a tool on an MCP server (lazy connect, kept alive across calls)."""
        try:
            cfg = json.loads((BASE / "mcp_servers.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"mcp_servers.json unreadable: {exc}"}
        entry = next(
            (s for s in cfg.get("servers", []) if s.get("name") == server), None
        )
        if entry is None:
            return {"ok": False, "error": f"unknown MCP server '{server}'"}
        with self._mcp_lock:
            client = self._mcp_clients.get(server)
            if client is None or client.returncode is not None:
                command = [
                    str(BASE.parent / part)
                    if str(part).endswith(".py")
                    and not Path(part).is_absolute()
                    and (BASE.parent / part).is_file()
                    else part
                    for part in entry["command"]
                ]
                client = McpClient(server, command)
                try:
                    client.start()
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "error": f"could not start {server}: {exc}"}
                self._mcp_clients[server] = client
        try:
            result = client.call(tool, args or {})
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{server}.{tool} failed: {exc}"}
        return {"ok": True, "server": server, "tool": tool, "result": result}

    def mcp_status(self) -> dict:
        # MCP servers this engine can offer. Connect/disconnect still planned (R14).
        servers = []
        for name, script, tools in (
            ("tts-mcp", "tts_mcp.py", ["speak"]),
            ("asr-mcp", "asr_mcp.py", ["transcribe", "asr_info"]),
            (
                "browser-mcp",
                "browser_mcp.py",
                [
                    "navigate",
                    "click",
                    "type",
                    "press",
                    "read_page",
                    "page_title",
                    "screenshot",
                ],
            ),
            (
                "tasks-mcp",
                "tasks_mcp.py",
                [
                    "tasks_info",
                    "get_time",
                    "set_timer",
                    "web_search",
                    "read_page",
                    "write_file",
                    "run_code",
                ],
            ),
            ("agents-mcp", "agents_mcp.py", ["list_agents", "ask_agent"]),
        ):
            script = BASE / script
            if script.is_file():
                servers.append(
                    {
                        "name": name,
                        "transport": "stdio",
                        "script": str(script),
                        "status": "available (started on demand by an MCP client)",
                        "tools": tools,
                    }
                )
        return {
            "servers": servers,
            "registry": "discovery live; connect/call from the console is planned (R14)",
        }

    def registry(self) -> dict:
        # Everything the agent has available, for the Registry view: tools with their
        # descriptions, skills, prompt templates, MCP servers, and stored attachments.
        import time as _t

        _t0 = _t.perf_counter()
        log_event("registry called")
        tools = []
        box = getattr(self, "_toolbox", None)
        if box is not None:
            try:
                for schema in box.schemas():
                    fn = schema.get("function", {})
                    tools.append(
                        {
                            "name": fn.get("name", "?"),
                            "description": (fn.get("description") or "")[:140],
                        }
                    )
            except (
                KeyError,
                TypeError,
            ) as exc:  # a malformed schema must not kill the state poll
                log_event("registry tool listing failed: " + str(exc)[:80])
        prompts = []
        pdir = BASE / "prompts"
        if pdir.is_dir():
            prompts = [x.stem for x in sorted(pdir.glob("*.md"))]
        skills = [x.get("name") for x in self.list_skills()]
        mcps = [x.get("name") for x in self.mcp_status().get("servers", [])]
        resources = []
        att = BASE / "out" / "attachments"
        if att.is_dir():
            resources = [x.name for x in sorted(att.iterdir())[-20:]]
        log_event(f"registry built in {(time.perf_counter() - _t0) * 1000:.0f} ms")
        return {
            "tools": tools,
            "skills": skills,
            "prompts": prompts,
            "mcp": mcps,
            "resources": resources,
        }

    def recent_events(self, limit: int = 120) -> list[dict]:
        # The durable event stream (events.jsonl), newest last.
        if self._session_dir is None:
            return []
        try:
            path = self._session_dir / "events.jsonl"
            if not path.is_file():
                return []
            out = []
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
            return out
        except OSError:
            return []

    def _active_voice(self) -> str:
        # Nova by default; Piattos when addressed. Falls back so nothing is voiceless.
        persona = getattr(self, "_persona", "nova")
        if persona in {"nova", "piattos"}:
            wanted = (
                self.settings.piattos_voice
                if persona == "piattos"
                else self.settings.nova_voice
            )
        else:
            wanted = str(
                self._custom_personas().get(persona, {}).get("voice")
                or self.settings.nova_voice
            )
        return wanted or self.settings.tts_voice

    def _custom_personas(self) -> dict:
        # Owner-editable personas.json: name -> voice + optional speed. Unknown
        # names fall back to Nova so a typo can never leave the agent voiceless.
        path = BASE / "personas.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def set_persona(self, name: str) -> str:
        wanted = str(name).strip().lower()
        if wanted in {"piattos"}:
            persona, voice = "piattos", self.settings.piattos_voice
        elif wanted in {"nova"}:
            persona, voice = "nova", self.settings.nova_voice
        else:
            custom = self._custom_personas().get(wanted)
            if not custom:
                return "unknown persona: " + wanted
            persona = wanted
            voice = str(custom.get("voice") or self.settings.nova_voice)
        persona_model = str(
            self._custom_personas().get(persona, {}).get("model") or ""
        ).strip()
        with self.lock:
            self._persona = persona
            self._persona_voice = voice
            # Swap to the persona's pre-built engine on the next tts() call;
            # never re-initialize sherpa-onnx here (hot-thread init segfaults).
            self._tts_wanted = ""
            if persona_model and persona_model != self.settings.model:
                # W2.3 (D-6): a persona is a different MIND, not just a voice -
                # rebuild the agent on this persona's own model (routing is data).
                self.settings.model = persona_model
                self._agent = None
                self._think(f"{persona} mind set to {persona_model}", "persona")
                log_event(f"persona {persona} -> model {persona_model}")
        self._think("persona switched to " + persona, "step")
        return persona

    def _prewarm_voices(self) -> None:
        # The voice list builds a SAPI COM object and loads the TTS engine. Doing
        # either in a background thread killed the whole process silently (COM/native
        # teardown across threads), so the load runs ONCE on the main thread at boot.
        # It costs a few seconds of startup; state() then only reads the cache.
        if self._voices_cache:
            return
        try:
            self._voices_cache = list_all_voices()
            log_event("voice list loaded: " + str(len(self._voices_cache)) + " voices")
        except TtsError as exc:
            self._voices_cache = []
            self.error = self.error or str(exc)

    # ------------------------------------------------------- sessions (#9/#26/#34/#60)
    def _sessions_root(self) -> Path:
        return BASE / "out" / "sessions"

    @staticmethod
    def _derive_session_title(text: str) -> str:
        """Build a short readable title from the session's first user chat."""
        cleaned = re.sub(r"\[image:[^\]]+\]", "", str(text or ""))
        cleaned = re.sub(
            r"^\s*(?:hey|hi|hello|okay|ok|um|uh|please|nova|piattos)[,:\s]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:can you|could you|would you|please)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        words = [w for w in re.split(r"\s+", cleaned.strip()) if w]
        if not words:
            words = [w for w in re.split(r"\s+", str(text or "").strip()) if w]
        title = " ".join(words[:7]).strip(" .?!,:;")
        if len(title) > 46:
            title = title[:43].rstrip() + "..."
        return (title[:1].upper() + title[1:]) if title else "New Chat"

    def _save_session_meta(self) -> None:
        if self._session_dir is None or not self._session_title:
            return
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self._session_dir / "meta.json"
            payload = {
                "id": self._session_dir.name,
                "title": self._session_title,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            log_event(f"could not save session meta: {exc}")

    @staticmethod
    def _read_session_title(folder: Path) -> str:
        meta = folder / "meta.json"
        if meta.is_file():
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(meta.read_text(encoding="utf-8"))
                if isinstance(data, dict) and str(data.get("title") or "").strip():
                    return str(data["title"]).strip()[:60]
        return ""

    def new_session(self) -> dict[str, Any]:
        """Start a fresh named session folder and reset active chat/pane context."""
        with contextlib.suppress(Exception):
            self.interrupt()
        panes_to_stop: list[Any] = []
        with self._pi_lock:
            panes_to_stop = list(self._pi_panes.values())
            self._pi_panes.clear()
        for pane in panes_to_stop:
            with contextlib.suppress(Exception):
                pane.stop()
        root = self._sessions_root()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = root / stamp
        suffix = 1
        while target.exists():
            suffix += 1
            target = root / f"{stamp}-{suffix}"
        with self.lock:
            self.turns.clear()
            self._turn_id = 0
            self._live_turn = None
            self._thinking_since = 0.0
            self._recent_task_reports.clear()
            self._pending_announcements.clear()
            self._thoughts = []
            self._turn_thoughts.clear()
            self.error = ""
            self._session_dir = target
            self._session_title = ""
            self._tasks = None
            if self._agent is not None:
                self._agent.reset()
        self._persist_event("session_created", {"id": target.name})
        return {"ok": True, "id": target.name, "title": "New Chat"}

    def list_sessions(self, limit: int = 40) -> list[dict[str, Any]]:
        """Past conversations, newest first, for the Sessions pane.

        A session is a folder under out/sessions holding messages.jsonl; the title
        comes from meta.json or the first thing the user said.
        """
        root = self._sessions_root()
        if not root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for folder in sorted(root.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            messages = folder / "messages.jsonl"
            turns = 0
            title = self._read_session_title(folder) or "(empty)"
            if messages.exists():
                try:
                    with messages.open(encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            turns += 1
                            if title == "(empty)":
                                record = json.loads(line)
                                text = str(
                                    record.get("user_text")
                                    or record.get("assistant_text")
                                    or ""
                                )
                                if text.strip():
                                    title = self._derive_session_title(text)
                except (OSError, ValueError):
                    continue
            if turns == 0 and not (self._session_dir and folder == self._session_dir):
                continue
            rows.append(
                {
                    "id": folder.name,
                    "title": title,
                    "turns": turns,
                    "path": str(folder),
                    "current": bool(self._session_dir and folder == self._session_dir),
                }
            )
        return rows[:limit]

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Read a past conversation back (restore: backup server.py.broken-20261108:584)."""
        path = self._sessions_root() / session_id / "messages.jsonl"
        if not path.exists():
            return []
        turns: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record.setdefault("id", index)
                    turns.append(record)
        except (OSError, ValueError) as exc:
            log_event(f"could not read session {session_id}: {exc}")
            return []
        return turns

    def activate_session(self, session_id: str) -> dict[str, Any]:
        """Load a past session into the active conversation so chat continues seamlessly."""
        sid = str(session_id or "").strip()
        target = self._sessions_root() / sid
        if not sid or not target.is_dir():
            return {"ok": False, "error": f"no such session: {sid}"}
        turns = self.load_session(sid)
        title = self._read_session_title(target)
        if not title:
            for t in turns:
                ut = str(t.get("user_text") or "").strip()
                if ut and not ut.startswith("(background task"):
                    title = self._derive_session_title(ut)
                    break
        with self.lock:
            self._session_dir = target
            self._session_title = title
            self.turns = list(turns)
            self._turn_id = max([int(t.get("id") or 0) for t in turns] or [len(turns)])
            self._live_turn = None
            self._thinking_since = 0.0
            self.error = ""
            self._tasks = None
            if self._agent is not None:
                self._agent.reset()
        self._persist_event("session_loaded", {"id": sid, "turns": len(turns)})
        return {
            "ok": True,
            "id": sid,
            "title": title or "New Chat",
            "turns": turns,
        }

    def resume_session(self, session_id: str) -> str:
        """Make a past session current again: new turns append to its folder (#26)."""
        target = self._sessions_root() / session_id
        if not target.is_dir():
            return "no such session: " + session_id
        with self.lock:
            self._session_dir = target
        self._persist_event("session_resumed", {"id": session_id})
        self._think(
            "session " + session_id + " resumed; new turns append there", "step"
        )
        return session_id

    def fork_session(self, session_id: str) -> str:
        """Copy a session folder to a new id (#9 fork: same past, independent future)."""
        import shutil

        source = self._sessions_root() / session_id
        if not source.is_dir():
            return "no such session: " + session_id
        new_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-fork"
        target = self._sessions_root() / new_id
        shutil.copytree(source, target)
        self._persist_event("session_forked", {"from": session_id, "to": new_id})
        return new_id

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        """#7: Preview a past session's messages & artifacts before continuing/deleting."""
        folder = self._sessions_root() / session_id
        if not folder.is_dir():
            return {"ok": False, "error": f"no such session: {session_id}"}
        turns = self.load_session(session_id)
        arts = []
        adir = folder / "artifacts"
        if adir.is_dir():
            arts = sorted(p.name for p in adir.iterdir() if p.is_file())[:30]
        return {
            "ok": True,
            "id": session_id,
            "turns": turns[-30:],
            "turn_count": len(turns),
            "artifacts": arts,
            "current": bool(
                self._session_dir and folder.resolve() == self._session_dir.resolve()
            ),
        }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """#7: Safely delete a past session folder (refuses to delete active session)."""
        import shutil

        folder = self._sessions_root() / session_id
        if not folder.is_dir():
            return {"ok": False, "error": f"no such session: {session_id}"}
        if self._session_dir and folder.resolve() == self._session_dir.resolve():
            return {"ok": False, "error": "cannot delete the active session"}
        try:
            shutil.rmtree(folder)
            return {"ok": True, "deleted": session_id}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------- approval gate (#30/#21)
    def request_approval(self, action: str, detail: str) -> int:
        """Queue a risky action. The tool returns 'awaiting approval' and the owner
        answers from the UI or POST /api/approval; nothing risky runs unattended."""
        with self.lock:
            self._approval_seq += 1
            approval_id = self._approval_seq
            self._approvals[str(approval_id)] = {
                "id": approval_id,
                "action": action,
                "detail": str(detail)[:400],
                "status": "pending",
                "ts": datetime.now().strftime("%H:%M:%S"),
            }
        self._persist_event("approval_requested", {"id": approval_id, "action": action})
        self._think(f"approval #{approval_id} needed: {action}", "gate")
        return approval_id

    def respond_approval(self, approval_id: str, approved: bool) -> dict[str, Any]:
        """The owner's answer. Approving marks it sent (delivery is simulated: no SMTP
        credentials on this machine); denying clears it. Either way the queue drains."""
        with self.lock:
            entry = self._approvals.get(str(approval_id))
            if entry is None:
                return {"ok": False, "error": "no such approval"}
            if entry["status"] != "pending":
                return {"ok": False, "error": "already " + str(entry["status"])}
            entry["status"] = "approved" if approved else "denied"
            action = str(entry["action"])
        self._persist_event(
            "approval_answered", {"id": approval_id, "approved": approved}
        )
        self._think(f"approval #{approval_id} {entry['status']}: {action}", "gate")
        return {"ok": True, "status": entry["status"], "action": action}

    def _role_fallback_chain(self) -> list[str]:
        """W1.4: the fallback models for the answering role, primary last.

        Reads agents.json (routing is data). The chain is [fallback1, fallback2,
        ...]; the role's primary is appended at the end so a fully failed chain
        still gets one last try with the configured model before the turn dies.
        """
        roles = load_agents_config()
        front = roles.get("front")
        entry = front if isinstance(front, dict) else {}
        paid_on = bool(getattr(self.settings, "paid_models_enabled", False))
        raw_chain = [str(f) for f in (entry.get("fallbacks") or []) if str(f).strip()]
        primary = str(entry.get("model", "")).strip()
        if primary:
            raw_chain.append(primary)
        return [
            m
            for m in raw_chain
            if paid_on
            or m.startswith("github-copilot/")
            or m.endswith(":free")
            or ("/" not in m)
        ]

    def fetch_free_models(self) -> dict[str, Any]:
        """W1.3: the free-model picker's source: OpenRouter :free, tool-capable."""
        try:
            models = list_openrouter_models(
                free_only=not bool(self.settings.paid_models_enabled)
            )
            return {"ok": True, "models": models}
        except LlmError as exc:
            return {"ok": False, "error": str(exc)[:140], "models": []}

    def roles_config(self) -> dict[str, Any]:
        """W1.2: the per-role roster as data (agents.json)."""
        return load_agents_config()

    def close(self) -> None:
        """Stop audio, listening, background tasks, and all PiPane child processes."""
        self._closing = True
        self.stop_listening()
        with self._mcp_lock:
            for _client in self._mcp_clients.values():
                with contextlib.suppress(Exception):
                    _client.close()
            self._mcp_clients.clear()
        with contextlib.suppress(Exception):
            if self._fillers is not None:
                self._fillers.stop()
        with contextlib.suppress(Exception):
            self.interrupt()
        with contextlib.suppress(Exception):
            if self._tasks is not None:
                self._tasks.cancel_all()
        with self._pi_lock:
            panes = list(self._pi_panes.values())
            self._pi_panes.clear()

        def _stop_one(pane: Any) -> None:
            with contextlib.suppress(Exception):
                pane.stop()

        # Stop panes in parallel: each stop can take seconds (taskkill + wait),
        # and serial stops outlasted the Windows console-close budget, leaving
        # pi children alive after the terminal closed.
        stops = [
            threading.Thread(
                target=_stop_one,
                args=(pane,),
                daemon=True,
                name=f"pi-stop-{getattr(pane, 'name', 'pane')}",
            )
            for pane in panes
        ]
        for worker in stops:
            worker.start()
        deadline = time.perf_counter() + 8.0
        for worker in stops:
            worker.join(timeout=max(0.05, deadline - time.perf_counter()))

    def set_role_model(self, role: str, model: str) -> str:
        """W1.2: write one role's model + fallbacks to agents.json (routing is data).

        The next turn that role runs uses the new model; the global model dropdown
        is untouched, so the conversation model and the role models stay separate.
        """
        wanted = str(role).strip().lower()
        model_id = str(model).strip()
        if not wanted or not model_id:
            return "pick a role and a model first"
        with self.lock:
            roster = load_agents_config()
            if wanted not in roster:
                roster[wanted] = {}
            entry = roster[wanted] if isinstance(roster.get(wanted), dict) else {}
            previous = str(entry.get("model", ""))
            entry["model"] = model_id
            fallbacks = [f for f in (entry.get("fallbacks") or []) if f != model_id]
            if previous and previous != model_id:
                fallbacks.insert(0, previous)
            entry["fallbacks"] = fallbacks[:3]
            roster[wanted] = entry
            try:
                AGENTS_PATH.write_text(json.dumps(roster, indent=2), encoding="utf-8")
            except OSError as exc:
                return f"could not write agents.json: {exc}"
        if wanted == "front":
            with self.lock:
                self.settings.model = model_id
                if model_id.startswith("github-copilot/"):
                    self.settings.provider = "github-copilot"
                elif (
                    ":" in model_id
                    and "/" not in model_id
                    and not model_id.endswith(":free")
                ):
                    self.settings.provider = "ollama"
                elif "/" in model_id:
                    self.settings.provider = "openrouter"
                self._agent = None
        with self._pi_lock:
            pane = self._pi_panes.get(wanted)
        if pane is not None:
            with contextlib.suppress(Exception):
                pane.configure(model=model_id)
        self._think(f"{wanted} model set to {model_id}", "roster")
        log_event(f"role {wanted} model -> {model_id}")
        return f"{wanted} will use {model_id} from its next turn"

    def _summarize_span(self, span: str) -> str:
        """M4.2: summarise a dropped history span with the local router model.

        Called from a background thread by Agent._compact_history. Any local
        failure returns "" and the plain trim stands - compaction must never
        break a turn.
        """
        try:
            return self.router().summarize(span)
        except Exception as exc:  # noqa: BLE001
            log_event(f"compaction summarise failed: {exc}")
            return ""

    def ab_compare(self, text: str, role_a: str, role_b: str) -> dict[str, Any]:
        """M1.4 (D-5): send the SAME text to two roles, so the owner can watch
        how each mind answers. Two normal background tasks, each on its role's
        model; the Thinking pane shows both when they finish (cost: $0 when the
        models are :free)."""
        clean = " ".join(str(text).split())[:800]
        if not clean:
            return {"ok": False, "error": "nothing to compare"}
        roster = load_agents_config()

        def model_of(role: str) -> str:
            entry = roster.get(role)
            entry = entry if isinstance(entry, dict) else {}
            return str(entry.get("model") or self.settings.model)

        picked = [r for r in (str(role_a), str(role_b)) if r and r in roster]
        if len(picked) < 2:
            return {
                "ok": False,
                "error": "pick two roles from: " + ", ".join(sorted(roster)),
            }
        started: list[dict[str, Any]] = []
        for role in picked:
            try:
                task = self.tasks_manager().start(
                    f"[A/B via {role}] {clean}\nAnswer directly and briefly.",
                    model=model_of(role),
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            started.append({"role": role, "model": model_of(role), "id": task.id})
        self._think(
            "A/B started: "
            + " vs ".join(f"{s['role']} ({s['model'][:24]})" for s in started),
            "ab",
        )
        return {"ok": True, "runs": started}

    def pending_approvals(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                dict(a) for a in self._approvals.values() if a["status"] == "pending"
            ]

    # ------------------------------------------------------- memory facts (#62)
    def memory_facts(self) -> list[dict[str, str]]:
        """MEMORY.md rows as {stamp, fact} for the inspector view."""
        mem = BASE / "MEMORY.md"
        rows: list[dict[str, str]] = []
        if not mem.is_file():
            return rows
        for line in mem.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("- ["):
                continue
            close = line.find("]")
            stamp_raw = line[3:close] if close > 3 else ""
            fact = line[close + 1 :].strip() if close > 0 else line
            # M4.1 provenance: "2026-11-16 22:33 | 20261116-223344" -> when + which session.
            stamp, _, session = stamp_raw.partition("|")
            if fact:
                rows.append(
                    {"stamp": stamp.strip(), "session": session.strip(), "fact": fact}
                )
        return rows

    def forget_fact(self, fact: str) -> str:
        """#47/#62: drop matching facts through the same tool the agent uses."""
        result = self._toolbox.call("forget", {"fact": str(fact)})
        return str(result.content)

    # ------------------------------------------------------- two-voice playback (#24)
    def speak_script(self, script: str) -> int:
        """Speak a NOVA:/PIATTOS:-tagged script with each persona's own voice.

        This is what makes the Audio Overview a two-voice podcast instead of one
        reader; voice switching is a settings change (no pipeline rebuild).
        """
        spoken = 0
        # Mark the session as speaking so the echo defences (idle VAD during
        # playback, barge gates, deaf tail) cover script playback too.
        with self.lock:
            self._speaking = True
            self._speech_started_at = time.perf_counter()
        try:
            for raw in script.splitlines():
                line = raw.strip()
                if not line:
                    continue
                upper = line.upper()
                persona = ""
                if upper.startswith("NOVA:"):
                    persona, line = "nova", line[5:].strip()
                elif upper.startswith("PIATTOS:"):
                    persona, line = "piattos", line[8:].strip()
                if not line:
                    continue
                if persona and persona != getattr(self, "_persona", "nova"):
                    self.set_persona(persona)
                try:
                    with self.lock:
                        self._last_spoken = line
                    voice = self.tts()
                    voice.say(line)
                    voice.wait(timeout_s=90)
                    spoken += 1
                except TtsError:
                    return spoken  # a failed voice must not kill the whole script
        finally:
            with self.lock:
                self._speaking = False
                self._last_spoken_at = time.perf_counter()
                self._deaf_until = time.perf_counter() + self._barge_gates()[2]
                # Script personas are temporary: conversation returns to the
                # configured conversation voice (fast lessac) afterwards.
                self._persona_voice = ""
                self._tts_wanted = ""
        return spoken

    def greet(self) -> str:
        """#57: a cached startup greeting, spoken as soon as the engine is up."""
        if not self.settings.tts_enabled:
            return "tts off"
        line = "Nova here. Ready when you are."
        try:
            with self.lock:
                self._speaking = True
                self._speech_started_at = time.perf_counter()
                self._last_spoken = line
            voice = self.tts()
            voice.say(line)
            voice.wait(timeout_s=30)
            self._persist_event("greeting", {})
            return "greeted"
        except TtsError as exc:
            return "greeting failed: " + str(exc)[:80]
        finally:
            with self.lock:
                self._speaking = False
                self._last_spoken_at = time.perf_counter()
                self._deaf_until = time.perf_counter() + self._barge_gates()[2]

    # ------------------------------------------------------- honest transcript (#41)
    def _speech_adjusted(self, text: str) -> str:
        """After a barge-in, the transcript says the reply was cut off instead of
        pretending the whole answer was spoken."""
        with self.lock:
            cut, self._speech_cut = self._speech_cut, False
        return (str(text) + " (cut off)") if cut else str(text)

    def _wake_ok(self, text: str) -> bool:
        """#58: from idle the wake gate wants 'nova'/'piattos' in the utterance.

        Normalize away spaces/punctuation so ASR near-misses ("No va", "Nova!",
        "Piat tos") still wake. The bare-wake ack already collapses these; the gate
        must agree with it or it holds a real wake word.
        """
        low = str(text).lower()
        if ("nova" in low) or ("piattos" in low):
            return True
        collapsed = re.sub(r"[^a-z]+", "", low)
        return (
            ("nova" in collapsed) or ("piattos" in collapsed) or ("piatos" in collapsed)
        )

    def _thought_complete(self, text: str) -> bool:
        """M2.5: does this utterance read as a finished thought? Fail-open - any
        router problem answers True, which is the pre-M2.5 behaviour."""
        try:
            return self.router().thought_complete(text)
        except Exception as exc:  # noqa: BLE001
            log_event(f"smart-turn judge failed ({exc}); ending the turn")
            return True

    def request_map(self) -> dict[str, Any]:
        """What is being asked, as a small mind map the UI can draw.

        Restored 2026-11-16 from the audited backup (server.py.broken-20261108:604):
        root is the latest user utterance; branches are the router's read of it,
        the tools that ran, and any background tasks it started. A projection of
        recorded state - it never invents structure that did not happen.
        """
        with self.lock:
            turns = list(self.turns)
            recent = next(
                (t for t in reversed(turns) if t.get("source") in {"mic", "text"}), None
            )
        if recent is None:
            return {"root": "(nothing asked yet)", "branches": []}
        branches: list[dict[str, Any]] = []
        tools: list[str] = []
        for tool in recent.get("tools") or []:
            name = tool.get("name") if isinstance(tool, dict) else str(tool)
            if name and name not in tools:
                tools.append(str(name))
        if tools:
            branches.append(
                {
                    "label": "tools used",
                    "kind": "tools",
                    "children": [{"label": name, "kind": "tool"} for name in tools],
                }
            )
        try:
            tasks = self.tasks_manager().list()
        except Exception:  # noqa: BLE001 - the map must never break the UI
            tasks = []
        live = [t for t in tasks if t.state in {"queued", "running", "waiting"}]
        if live or tasks:
            chosen = live or tasks[-3:]
            branches.append(
                {
                    "label": "background work",
                    "kind": "tasks",
                    "children": [
                        {
                            "label": f"{t.id}: {t.brief[:34]}",
                            "kind": "task",
                            "detail": t.state,
                        }
                        for t in chosen
                    ],
                }
            )
        return {
            "root": str(recent.get("user_text", ""))[:80],
            "branches": branches,
            "reply": str(recent.get("assistant_text", ""))[:80],
        }

    # ------------------------------------------------------------- pi panes (RPC)
    def pi_pane(self, name: str) -> PiPane:
        """Return (creating on first use) the pi terminal pane for a role."""
        raw = str(name).strip().lower() or "front"
        key = self._pick_back_worker() if raw in BACK_ALIASES else raw
        with self._pi_lock:
            pane = self._pi_panes.get(key)
            if pane is None:
                roster = load_agents_config()
                entry = roster.get(key) if isinstance(roster.get(key), dict) else {}
                model = str((entry or {}).get("model") or "github-copilot/gpt-6-luna")
                thinking = str((entry or {}).get("thinking") or "medium").lower()
                ext_path = BASE / ".pi" / "extensions" / "sonj_bridge.ts"
                extra = [
                    "--no-extensions",
                    "--no-context-files",
                    "--no-skills",
                    "--approve",
                ]
                bridge_mode = "front"
                if key == "front":
                    if ext_path.is_file():
                        extra += ["-e", str(ext_path), "--no-builtin-tools"]
                elif key in BACK_POOL or key.startswith("back"):
                    bridge_mode = "back"
                    if ext_path.is_file():
                        extra += ["-e", str(ext_path)]
                    extra += [
                        "-t",
                        "read,bash,powershell,edit,write,grep,find,ls",
                    ]
                elif key == "doctor":
                    # Doctor is read-only + doctor_app_control + peer community posts (no self-coding).
                    bridge_mode = "doctor"
                    if ext_path.is_file():
                        extra += ["-e", str(ext_path)]
                    extra += ["-t", "read,grep,find,ls"]
                else:
                    extra += ["--no-tools"]
                pane = PiPane(
                    key,
                    cwd=str(BASE.parent),
                    model=model,
                    thinking=thinking,
                    system_prompt=self._role_system_prompt(key),
                    env={
                        "ME3_ENGINE_URL": self.engine_url,
                        "ME3_BRIDGE_MODE": bridge_mode,
                        "SONJ_ROLE": key,
                    },
                    extra_args=extra,
                )
                self._pi_panes[key] = pane
            return pane

    def pi_panes_snapshot(self) -> dict[str, Any]:
        """The LLM panes the UI renders: Front, Doctor, and all 5 Back pool workers."""
        with self._pi_lock:
            panes = dict(self._pi_panes)
        ordered = ["front", "doctor", *BACK_POOL]
        for k in panes:
            if k not in ordered:
                ordered.append(k)
        out: dict[str, Any] = {}
        for name in ordered:
            pane = panes.get(name)
            if pane is None:
                out[name] = {
                    "status": "warming",
                    "lines": [],
                    "tokens": 0,
                    "model": "",
                }
            else:
                out[name] = {
                    "status": pane.status,
                    "lines": pane.snapshot(120),
                    "tokens": int(getattr(pane, "last_tokens", 0) or 0),
                    "model": (
                        f"{getattr(pane.rpc, 'provider', '')}/"
                        f"{getattr(pane.rpc, 'model', '')}"
                    ).strip("/"),
                }
        return out

    def pi_prompt(self, role: str, message: str) -> str:
        """Send a prompt to a role's pi pane and stream it in the background."""
        text = str(message or "").strip()
        if not text:
            return "the prompt was empty"
        pane = self.pi_pane(role)
        self._think(f"pi pane {pane.name}: {text[:60]}", "pi")

        def run() -> None:
            try:
                if not pane.rpc.alive:
                    pane.start()
                pane.ask(text)
            except Exception as exc:  # noqa: BLE001 - a pane failure is shown, not raised
                pane.note(f"error: {type(exc).__name__}: {exc}")
                log_event(f"pi pane {pane.name} failed: {exc}")

        threading.Thread(target=run, daemon=True, name=f"pi-pane-{pane.name}").start()
        return f"{pane.name} is working on it"

    def pi_abort(self, role: str) -> str:
        """Stop whatever the named pane is running (every pane when role is blank)."""
        wanted = str(role or "").strip().lower()
        with self._pi_lock:
            keys = [wanted] if wanted else list(self._pi_panes)
            panes = [self._pi_panes[k] for k in keys if k in self._pi_panes]
        if not panes:
            return "no pane is running"
        for pane in panes:
            with contextlib.suppress(Exception):
                pane.rpc.abort()
        return "stopping " + ", ".join(p.name for p in panes)

    def latency_summary(self, window: int = 50) -> dict[str, Any]:
        """p50, p95 and max per stage over the last N turns.

        Read from the turn records, so it needs no log parsing. A stage a turn did
        not record (no filler, no streamed token) is simply absent.
        """
        try:
            size = max(1, int(window))
        except (TypeError, ValueError):
            size = 50
        with self.lock:
            rows = list(self.turns[-size:])
        stages = (
            "asr_ms",
            "llm_ms",
            "tts_ms",
            "first_token_ms",
            "filler_ms",
            "reply_audio_ms",
            "total_ms",
            "ttfa_ms",
            "e2e_ms",
        )
        out: dict[str, Any] = {"turns": len(rows)}
        for key in stages:
            vals = sorted(
                float(t["timings"].get(key) or 0.0)
                for t in rows
                if isinstance(t.get("timings"), dict) and key in t["timings"]
            )
            if not vals:
                continue
            out[key] = {
                "p50": round(vals[len(vals) // 2], 1),
                "p95": round(vals[min(len(vals) - 1, int(len(vals) * 0.95))], 1),
                "max": round(vals[-1], 1),
                "n": len(vals),
            }
        return out

    def state(self) -> dict[str, Any]:
        # Slow I/O (model catalogues, pane snapshots) runs OUTSIDE the session
        # lock: a 20 s model query under the lock wedged turns, speech and
        # shutdown, which all need it. Each helper takes its own short locks.
        models = self.available_models()
        panes = self.pi_panes_snapshot()
        statuses = self.active_statuses()
        active = self.agents_active()
        task_summaries = self.tasks_manager().summaries()
        workflow = self.tasks_manager().graph()
        loaded: list[dict[str, Any]] = []
        if self.settings.provider == "ollama":
            try:
                loaded = loaded_ollama_models()
            except LlmError as exc:
                loaded = []
                log_event(f"could not list loaded models: {exc}")
        with self.lock:
            settings = self.settings.persistable()
            settings["openrouter_key_set"] = bool(self.settings.openrouter_key)
            voices = list(self._voices_cache or [])  # loaded once by _prewarm_voices
            return {
                "status": self.status,
                "error": self.error,
                "mode": self.mode,
                "level": float(self.level),
                "recording": bool(self.recording),
                "barge_ins": int(self.barge_ins),
                "echo_drops": int(self.echo_drops),
                "ignored_while_speaking": int(self.ignored_while_speaking),
                "memory_mb": process_memory_mb(),
                "loaded_models": [str(m.get("name", "")) for m in loaded],
                "settings": settings,
                "persona": getattr(self, "_persona", "nova"),
                "turns": list(self.turns),
                "speaking": bool(self._speaking),
                "thinking": list(getattr(self, "_thoughts", [])[-80:]),
                "events": self.recent_events(),
                "tasks": task_summaries,
                "workflow": workflow,
                "approvals": self.pending_approvals(),
                "spend_tokens": int(self._spend_tokens),
                "spend_usd": round(self.total_spend_summary()["total_usd"], 6),
                "spend_breakdown": self.total_spend_summary(),
                "system_health": system_health(),
                "voices": self._voiceprint.dashboard(),
                "gate_stats": dict(self._gate_stats),
                "classifier": {
                    "stats": dict(self.classifier().stats),
                    "last": self._last_decision,
                },
                "doctor_pending": self._doctor_gate.pending(),
                "agents": load_agents_config(),
                "agents_active": active,
                "statuses": statuses,
                "live_turn": self._live_turn,
                "incidents": self.incidents.snapshot(40),
                "incident_count": self.incidents.count(),
                "doctor": self._doctor.status(),
                "session_dir": str(self._session_dir) if self._session_dir else "",
                "session_id": self._session_dir.name if self._session_dir else "",
                "session_title": self._session_title or "New Chat",
                "models": models,
                "pi_panes": panes,
                "asr_backends": sorted(discover_models(MODELS_DIR)),
                "tts_voices": voices,
                "devices": self.device_summary(),
                "echo_engine": getattr(self._echo, "name", "off"),
                "echo_score": round(float(getattr(self._echo, "last_score", 0.0)), 3),
                "echo_erle_db": round(
                    float(getattr(self._echo, "last_erle_db", 0.0)), 1
                ),
                "ns_engine": getattr(self._ns, "name", "off"),
                "ns_reduction_db": round(
                    float(getattr(self._ns, "last_reduction_db", 0.0)), 1
                ),
                "quality": self.quality_stats(),
                "startup_check": getattr(self, "startup_check", {}),
                "voiceprint": {
                    "enrolled": self._voiceprint.enrolled,
                    "model_present": self._voiceprint.model_present,
                },
                "session_profile": list(self._session_profile),
                "cwd": str(BASE.parent),
                "app_dir": str(BASE),
                "tools": self.tool_names(),
            }

    def _load_session_profile(self) -> None:
        path = OUT_DIR / "session_profile.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            facts = data.get("facts") if isinstance(data, dict) else None
            if isinstance(facts, list):
                self._session_profile = [str(f) for f in facts][-30:]
        except (OSError, ValueError):
            self._session_profile = []

    def _save_session_profile(self) -> None:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "session_profile.json").write_text(
                json.dumps({"facts": self._session_profile}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log_event(f"could not save session profile: {exc}")

    def _maybe_learn(self, turn: dict[str, Any]) -> None:
        """In-session learning: extract durable owner facts after each turn.

        Runs async on a cheap free model so it never adds turn latency; the
        profile is injected into Front's prompt and shown in Settings.
        """
        if not getattr(self.settings, "session_learning", True):
            return
        user_text = str(turn.get("user_text") or "")
        if len(user_text) < 15 or turn.get("source") not in {"mic", "text"}:
            return

        def run() -> None:
            try:
                client = create_llm(
                    "openrouter",
                    "nvidia/nemotron-3.5-lightning:free",
                    self.settings.openrouter_key,
                )
                reply = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Extract durable owner preferences, facts, or corrections "
                                "from this exchange. Reply with at most 3 short standalone "
                                "lines about the owner. If nothing durable, reply NONE."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Owner: {user_text[:300]}\n"
                                f"Assistant: {str(turn.get('assistant_text') or '')[:300]}"
                            ),
                        },
                    ]
                )
                text = str(reply.content or "").strip()
                if not text or text.upper().startswith("NONE"):
                    return
                lines = [
                    ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()
                ][:3]
                with self.lock:
                    profile = list(self._session_profile)
                    for line in lines:
                        if line and line not in profile:
                            profile.append(line)
                    self._session_profile = profile[-30:]
                    self._save_session_profile()
                log_event(f"session learning: +{len(lines)} fact(s)")
            except (LlmError, ValueError, OSError) as exc:
                log_event(f"session learning skipped: {exc}")

        threading.Thread(target=run, daemon=True, name="session-learn").start()

    def write_latency_report(self) -> str:
        """Export the live latency + quality stats as a markdown report."""
        st = self.latency_summary()
        q = self.quality_stats()
        lines = [
            "# SONJ latency report — " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            f"Session turns analyzed: {st.get('turns', 0)}",
            "",
            "| stage | p50 ms | p95 ms | max ms | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for key in (
            "asr_ms",
            "first_token_ms",
            "ttfa_ms",
            "e2e_ms",
            "filler_ms",
            "llm_ms",
            "tts_ms",
            "total_ms",
        ):
            r = st.get(key)
            if r:
                lines.append(
                    f"| {key} | {r.get('p50', 0):.0f} | {r.get('p95', 0):.0f} "
                    f"| {r.get('max', 0):.0f} | {r.get('n', 0)} |"
                )
        lines += [
            "",
            (
                "Targets: ack <300 ms · chat first audio <1500 ms · "
                "industry TTFA p95 <800 ms (premium <500 ms)."
            ),
            "",
            "## Quality (this session)",
            "",
        ]
        for k, v in q.items():
            lines.append(f"- {k}: {v}")
        lines += [
            "",
            "## Configuration",
            "",
            f"- model: {self.settings.provider}/{self.settings.model}",
            f"- chat thinking: {self.settings.chat_thinking}",
            f"- ASR: {self.settings.asr_backend}",
            f"- TTS: {self.settings.tts_voice}",
            f"- barge strictness: {self.settings.barge_strictness}",
            f"- echo cancellation: {self.settings.echo_cancellation}",
            "",
        ]
        path = OUT_DIR / "latency_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def enroll_voiceprint(self, seconds: int = 10) -> dict[str, Any]:
        """Record the owner's voice and store the embedding (mic released first)."""
        if not self._voiceprint.model_present:
            return {"ok": False, "error": "speaker model missing"}
        was_mode = self.mode
        self.stop_listening()
        time.sleep(0.3)
        mic = None
        try:
            mic = MicStream(device=self.input_device)
            mic.start()
            chunks: list[np.ndarray] = []
            deadline = time.perf_counter() + max(5, min(20, int(seconds)))
            self._persist_event(
                "voiceprint", {"msg": f"enrolling: recording {int(seconds)}s"}
            )
            while time.perf_counter() < deadline:
                block = mic.read(timeout=0.3)
                if block is not None:
                    chunks.append(block)
            if not chunks:
                return {"ok": False, "error": "no audio captured"}
            samples = np.concatenate(chunks)
            self._voiceprint.enroll(samples)
            log_event(f"voiceprint enrolled ({len(samples) / 16000:.1f}s)")
            return {
                "ok": True,
                "enrolled": True,
                "seconds": round(len(samples) / 16000, 1),
            }
        except AudioError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            if mic is not None:
                with contextlib.suppress(AudioError):
                    mic.stop()
            if was_mode in {"continuous", "ptt"}:
                self.start_listening(was_mode)

    def quality_stats(self) -> dict[str, Any]:
        """Live conversation-quality counters for the chat strip / Performance view."""
        try:
            tasks = self.tasks_manager().list()
            incomplete = sum(1 for t in tasks if t.state in {"blocked", "cancelled"})
        except Exception:  # noqa: BLE001 - counters must never break state
            incomplete = 0
        return {
            "turns": len(self.turns),
            "interruptions": int(self.barge_ins),
            "incomplete_tasks": incomplete,
            "errors": int(self.incidents.count()),
            "clarifications": int(self._clarifications),
            "echo_drops": int(self.echo_drops),
        }

    def pipeline_status(self) -> dict[str, Any]:
        """The interactive Pipeline view: each stage's backend, options, latency."""
        st = self.latency_summary()

        def p50(key: str) -> float:
            row = st.get(key) or {}
            return float(row.get("p50") or 0.0)

        return {
            "stages": {
                "vad": {
                    "backend": "silero (local)",
                    "note": "local VAD is sub-millisecond; cloud VAD adds a round-trip",
                    "options": {
                        "threshold": self.settings.vad_threshold,
                        "min_silence": self.settings.vad_min_silence,
                        "smart_turn": self.settings.smart_turn,
                    },
                },
                "asr": {
                    "backend": (
                        f"{self.settings.asr_cloud} (cloud)"
                        if self.settings.asr_cloud
                        else f"{self.settings.asr_backend} (local)"
                    ),
                    "p50_ms": p50("asr_ms"),
                    "local_backends": sorted(discover_models(MODELS_DIR)),
                    "active_local": self.settings.asr_backend,
                    "active_cloud": self.settings.asr_cloud,
                    "cloud": [
                        {
                            "id": "deepgram",
                            "key_set": bool(self.settings.deepgram_api_key),
                        },
                        {"id": "openai", "key_set": bool(self.settings.openai_api_key)},
                    ],
                },
                "classifier": {
                    "backend": f"{self.classifier().stats.get('stage', 'rules-v1')} (non-LLM, +10% Front bias)",
                    "p50_ms": float(
                        self.classifier().stats.get("last_latency_ms") or 0.0
                    ),
                    "last": self._last_decision,
                },
                "llm": {
                    "backend": f"{self.settings.provider}/{self.settings.model}",
                    "p50_ms": p50("first_token_ms"),
                    "chat_thinking": self.settings.chat_thinking,
                },
                "tts": {
                    "backend": (
                        f"{self.settings.tts_cloud} (cloud)"
                        if self.settings.tts_cloud
                        else (self.settings.tts_voice or "sapi")
                    ),
                    "p50_ms": p50("ttfa_ms"),
                    "voices": list(self._voices_cache or []),
                    "active_voice": self.settings.tts_voice,
                    "active_cloud": self.settings.tts_cloud,
                    "cloud": [
                        {"id": "openai", "key_set": bool(self.settings.openai_api_key)},
                        {
                            "id": "elevenlabs",
                            "key_set": bool(self.settings.elevenlabs_api_key),
                        },
                    ],
                },
            },
            "e2e_p50_ms": p50("e2e_ms"),
            "ack_p50_ms": p50("filler_ms"),
        }

    def active_statuses(self) -> list[dict[str, Any]]:
        """In-progress work ONLY, for the status pane.

        Running/queued tasks and synchronous consults appear here while they work;
        when they finish they leave this list and land in the engine event log.
        """
        rows: list[dict[str, Any]] = []
        now = time.time()
        try:
            for task in self.tasks_manager().list():
                if task.state not in {"queued", "running", "waiting"}:
                    continue
                label = re.sub(r"^\[crew:[^\]]*\]\s*", "", str(task.brief))
                roles = [
                    str(s.get("role")) for s in (getattr(task, "steps_plan", []) or [])
                ]
                started = float(task.started or task.created or now)
                rows.append(
                    {
                        "id": task.id,
                        "kind": "task",
                        "label": label[:90],
                        "role": ", ".join(roles) or "back",
                        "state": task.state,
                        "seconds": round(max(0.0, now - started), 1),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - the status pane must never break state
            log_event(f"status pane: task listing failed ({exc})")
        with self.lock:
            consults = list(self._active_consults.values())
        for consult in consults:
            rows.append(
                {
                    "id": str(consult.get("id", "")),
                    "kind": "consult",
                    "label": str(consult.get("label", ""))[:90],
                    "role": str(consult.get("role", "")),
                    "state": "running",
                    "seconds": round(
                        max(0.0, now - float(consult.get("started", now))), 1
                    ),
                }
            )
        return rows

    def docs_payload(self) -> dict[str, Any]:
        """Documentation view: the whole manual plus a live roster summary."""
        full_path = BASE / "docs" / "SONJ_DOCUMENTATION.md"
        try:
            full = full_path.read_text(encoding="utf-8")
        except OSError:
            full = (
                "# SONJ4 documentation\n\nThe documentation file is missing "
                "(sonj4/docs/SONJ_DOCUMENTATION.md)."
            )
        return {"full": full, "summary": self._docs_summary()}

    def _docs_summary(self) -> str:
        """Summary built from agents.json + live settings, so it never drifts."""
        roster = load_agents_config()
        toolsets = {
            "front": "classify_decision, ask_agent, delegate_task, start_crew, list_agents, device_tool (<2s only)",
            "doctor": "read, grep, find, ls, doctor_app_control, post_community, ask_agent (no self-coding)",
            "classifier": "non-LLM deterministic + ONNX slot (<100ms, +10% Front bias; zero pane tools)",
        }
        for slot in BACK_POOL:
            toolsets[slot] = (
                "read, bash, powershell, edit, write, grep, find, ls, device_tool, mcp_call, ask_agent, post_community"
            )
        lines = [
            "# SONJ4 — summary",
            "",
            "## Pipeline",
            "",
            (
                "mic → Adaptive NLMS AEC (TTS ref) → Telecon Wiener NS → VAD (Silero) → "
                "ASR (sherpa-onnx) → Decision Classifier (<100ms, +10% Front bias) → "
                "FRONT (1) → Back Pool (5 always-warm: back-1..back-5) / Doctor (1) → TTS → speaker."
            ),
            (
                "FRONT is the only voice the owner hears; specialists work in their own pi.dev "
                "terminal panes. Long work is delegated to background tasks and reports into chat."
            ),
            "",
            "## Roles",
            "",
        ]
        for role, raw_entry in roster.items():
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            lines.append(f"### {role}")
            lines.append(f"- **Does:** {entry.get('description') or 'specialist role'}")
            lines.append(
                f"- **Model:** `{entry.get('model') or '?'}` "
                f"(thinking: {entry.get('thinking') or 'medium'})"
            )
            lines.append(f"- **Tools:** {toolsets.get(role, 'pi.dev tools')}")
            fallbacks = entry.get("fallbacks") or []
            if fallbacks:
                lines.append(
                    "- **Fallbacks:** " + ", ".join(f"`{m}`" for m in fallbacks)
                )
            lines.append("")
        settings = self.settings
        lines += [
            "## Current configuration",
            "",
            f"- Conversation model: `{settings.model}` ({settings.provider})",
            f"- ASR backend: `{settings.asr_backend}`",
            (
                f"- TTS: {'on' if settings.tts_enabled else 'off'} — voice "
                f"`{settings.tts_voice or settings.nova_voice}`"
            ),
            (
                f"- Audio front-end: TTS Echo Cancel ({getattr(self._echo, 'name', 'nlms-aec')}): "
                f"{'ON' if settings.echo_cancellation else 'off'} · "
                f"Telecon Noise Suppress ({getattr(self._ns, 'name', 'wiener-ns')}): "
                f"{'ON' if getattr(settings, 'noise_suppression', True) else 'off'}"
            ),
            (
                f"- Controls: Auto-Interrupt: {'ON' if getattr(settings, 'instant_barge', False) else 'off (default)'} · "
                f"Front Only: {'ON' if getattr(settings, 'front_only', False) else 'off (default)'} · "
                f"Streamed voice: {'on' if settings.stream_voice else 'off'}"
            ),
            f"- Paid models: {'enabled' if settings.paid_models_enabled else 'disabled (free + Copilot only)'}",
            "",
        ]
        return "\n".join(lines)

    def tool_names(self) -> list[str]:
        """The tool names the agent can call, for the dashboard's resource panel.

        Schemas are OpenAI-shaped ({type: function, function: {name, ...}}), so the
        name sits one level down; reading the top level returned an empty list and
        the dashboard fell back to a hardcoded tool list.
        """
        box = getattr(self, "_toolbox", None)
        if box is None:
            return []
        try:
            names = []
            for schema in box.schemas():
                fn = schema.get("function") if isinstance(schema, dict) else None
                name = (fn or schema).get("name") if isinstance(schema, dict) else None
                if name:
                    names.append(str(name))
            return names
        except Exception as exc:  # noqa: BLE001 - a catalogue failure is not fatal
            log_event(f"could not list tools: {exc}")
            return []

    def full_catalog(self) -> list[dict[str, Any]]:
        """Every usable model, tagged so the picker can group and search it.

        OpenRouter is asked for ALL tool-capable models (not just the free ones),
        because the owner wants DeepSeek-class models on tap. Free variants
        (":free") and local Ollama models are flagged so the UI keeps them first.
        """
        with self.lock:
            if (
                self._catalog_cache is not None
                and (time.perf_counter() - self._catalog_at) < MODELS_CACHE_S
            ):
                return self._catalog_cache

        catalog: list[dict[str, Any]] = []
        for mid in self.available_models().get("github-copilot", []):
            catalog.append(
                {
                    "id": str(mid),
                    "provider": "GitHub Copilot",
                    "group": "Copilot (Pi.dev)",
                    "free": True,
                    "tools": True,
                }
            )
        for mid in self.available_models().get("ollama", []):
            catalog.append(
                {"id": str(mid), "provider": "Ollama", "group": "Ollama", "free": True}
            )
        # The owner's full collection: every OpenRouter model, not only the
        # tool-capable subset, so the picker is complete and searchable.
        try:
            openrouter = list_openrouter_models(free_only=False, require_tools=False)
        except (LlmError, ValueError):
            openrouter = list(OPENROUTER_MODELS)
        for preferred in OPENROUTER_MODELS:
            if preferred not in openrouter:
                openrouter.append(preferred)
        tool_capable: set[str] = set()
        try:
            tool_capable = set(list_openrouter_models(free_only=False))
        except (LlmError, ValueError):
            pass
        for mid in openrouter:
            text = str(mid)
            family = text.split("/")[0] if "/" in text else "other"
            catalog.append(
                {
                    "id": text,
                    "provider": "OpenRouter",
                    "group": family.replace("-", " ").replace("_", " ").title(),
                    "free": text.endswith(":free"),
                    "tools": text in tool_capable,
                }
            )
        catalog.sort(
            key=lambda m: (0 if m["free"] else 1, m["provider"], m["group"], m["id"])
        )
        with self.lock:
            self._catalog_cache = catalog
            self._catalog_at = time.perf_counter()
        return catalog

    def file_tree(self, limit: int = 300) -> dict[str, Any]:
        """Allowed folders and readable files for the Files view (/api/files)."""
        import filereader

        folders = [str(f) for f in filereader.allowed_folders()]
        files: list[dict[str, Any]] = []
        for folder in filereader.allowed_folders():
            try:
                rows = filereader.list_folder(folder, "*", limit=limit)
            except filereader.FileError:
                continue
            for row in rows:
                files.append(
                    {
                        "name": row["name"],
                        "path": str(folder / str(row["name"])),
                        "kind": row["kind"],
                        "mb": row["mb"],
                    }
                )
                if len(files) >= limit:
                    break
            if len(files) >= limit:
                break
        return {"folders": folders, "files": files}

    def dir_context(self, limit: int = 80) -> dict[str, Any]:
        """The main directory: its path and a bounded listing, for the UI card."""
        root = BASE.parent
        entries: list[dict[str, Any]] = []
        try:
            children = sorted(
                root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
            for child in children:
                if child.name.startswith("."):
                    continue
                try:
                    size = child.stat().st_size if child.is_file() else 0
                except OSError:
                    size = 0
                entries.append(
                    {"name": child.name, "dir": child.is_dir(), "size": size}
                )
                if len(entries) >= limit:
                    break
        except OSError as exc:
            log_event(f"could not list {root}: {exc}")
        return {
            "ok": True,
            "path": str(root),
            "entries": entries,
            "count": len(entries),
        }

    def set_role_thinking(self, role: str, level: str) -> str:
        """How hard a role thinks before it answers, stored in agents.json.

        thinking is a routing knob, not a code change: off / low / medium / high.
        """
        wanted = str(role).strip().lower()
        chosen = str(level).strip().lower()
        if chosen not in {"off", "low", "medium", "high"}:
            return "thinking must be off, low, medium or high"
        roster = load_agents_config()
        if wanted not in roster:
            return f"no such role: {wanted}"
        with self.lock:
            entry = roster[wanted] if isinstance(roster.get(wanted), dict) else {}
            entry["thinking"] = chosen
            roster[wanted] = entry
            try:
                AGENTS_PATH.write_text(json.dumps(roster, indent=2), encoding="utf-8")
            except OSError as exc:
                return f"could not write agents.json: {exc}"
        with self._pi_lock:
            pane = self._pi_panes.get(wanted)
        if pane is not None:
            with contextlib.suppress(Exception):
                pane.configure(thinking=chosen)
        self._think(f"{wanted} thinking set to {chosen}", "roster")
        log_event(f"role {wanted} thinking -> {chosen}")
        return f"{wanted} will think {chosen} from its next turn"

    def update_settings(self, patch: dict[str, Any]) -> None:
        allowed = set(Settings().__dataclass_fields__)
        with self.lock:
            for key, value in patch.items():
                if key in allowed and value is not None:
                    setattr(self.settings, key, value)
            if "paid_models_enabled" in patch:
                self._models_cache = None
                self._catalog_cache = None
            if "vad_threshold" in patch or "vad_min_silence" in patch:
                self._vad = None
            if "model_memory" in patch:
                self._agent = None
            if any(
                k in patch
                for k in ("deepgram_api_key", "openai_api_key", "elevenlabs_api_key")
            ):
                # Cloud engines cache their key at build time: drop them so the
                # next call rebuilds with the new key.
                self._tts_engines = {
                    k: v
                    for k, v in self._tts_engines.items()
                    if not k.startswith("cloud:")
                }
                self._tts_wanted = ""
                self._asr = None
            if "tts_voice" in patch or "tts_enabled" in patch or "tts_cloud" in patch:
                # Invalidate FIRST, then reload: prewarm only fills an empty cache, so
                # wiping afterwards left tts_voices empty until the next reboot.
                self._voices_cache = None
                self._prewarm_voices()  # reload the list with the new voice
            if "model" in patch:
                chosen = str(self.settings.model or "")
                if chosen.startswith("github-copilot/"):
                    self.settings.provider = "github-copilot"
                elif (
                    ":" in chosen and "/" not in chosen and not chosen.endswith(":free")
                ):
                    self.settings.provider = "ollama"
                elif "/" in chosen and "provider" not in patch:
                    self.settings.provider = "openrouter"
                pane = self._pi_panes.get("front")
                if pane is not None and chosen:
                    with contextlib.suppress(Exception):
                        pane.configure(model=chosen)
            if "provider" in patch or "model" in patch or "front_only" in patch:
                self._agent = None
            if "asr_backend" in patch:
                self._asr = None
        if "front_only" in patch:
            with self._pi_lock:
                stale = self._pi_panes.pop("front", None)
            if stale is not None:
                threading.Thread(target=lambda p=stale: p.stop(), daemon=True).start()
        self.save_config()

    def save_config(self) -> None:
        """Write settings, keeping the previous file as config.json.bak.

        R8 restores from that backup when the live file is corrupt, so the backup
        is written BEFORE the new file, not after.

        A process running with ME3_NO_PERSIST set (the checks, which flip TTS off
        so they stay silent) never writes: that is how the owner's tts_enabled
        kept getting clobbered back to False in the live config.
        """
        if os.environ.get("ME3_NO_PERSIST"):
            return
        try:
            if CONFIG_PATH.exists():
                backup = CONFIG_PATH.with_suffix(".json.bak")
                backup.write_text(
                    CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
                )
            CONFIG_PATH.write_text(
                json.dumps(self.settings.persistable(), indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._incident(
                "config", f"could not write config: {exc}", where="save_config"
            )

    def reset(self) -> None:
        with self.lock:
            self.turns.clear()
            if self._agent is not None:
                self._agent.reset()
            self.error = ""


def _handler_factory(session: Session):
    class Handler(BaseHTTPRequestHandler):
        server_version = "me3-laptop/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            # format is the message template; args[0] is its first substitution, which is
            # an int for send_error - testing it as text raised on every 404.
            if "api/state" in str(format):
                return
            print(f"[http] {format % args}")

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404, "not found")
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError:
                return {}

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route in {"/", "/index.html"}:
                self._send_file(STATIC_DIR / "index.html")
                return
            if route == "/api/state":
                self._send_json(session.state())
                return
            if route == "/api/elevenlabs/usage":
                self._send_json(elevenlabs_usage(session.settings.elevenlabs_api_key))
                return
            if route == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "status": session.status,
                        "boot_id": getattr(session, "_boot_id", ""),
                    }
                )
                return
            if route == "/api/latency":
                # Per-stage p50/p95 over recent turns: the latency checkpoints.
                self._send_json({"ok": True, "stages": session.latency_summary()})
                return
            if route == "/api/system_health":
                self._send_json(system_health())
                return
            if route == "/api/performance/distribution":
                self._send_json(feature_distribution(session))
                return
            if route == "/api/performance/review":
                self._send_json(performance_review(session))
                return
            if route == "/api/introspection":
                self._send_json(app_introspection(session))
                return
            if route == "/api/voices":
                if "active=1" in (urlparse(self.path).query or ""):
                    session._voices_view_at = time.monotonic()
                self._send_json(session._voiceprint.dashboard())
                return
            if route == "/api/pi/panes":
                self._send_json({"ok": True, "panes": session.pi_panes_snapshot()})
                return
            if route == "/api/incidents":
                # S1.3: the incident log the Doctor pane renders (text only).
                self._send_json(
                    {
                        "incidents": session.incidents.snapshot(80),
                        "count": session.incidents.count(),
                        "report": session.incidents.report(60),
                    }
                )
                return
            if route == "/api/doctor":
                self._send_json(
                    {
                        "status": session.doctor_status(),
                        "scopes": list(DOCTOR_SCOPES),
                        "agents": list(DOCTOR_AGENTS),
                        "repairs": [
                            {"id": key, "title": entry.title, "action": entry.action}
                            for key, entry in sorted(DOCTOR_REPAIRS.items())
                        ],
                    }
                )
                return
            if route == "/api/agents_active":
                # S4: the activity indicators (which roles are working right now).
                self._send_json({"agents": session.agents_active()})
                return
            if route == "/api/skills":
                self._send_json({"skills": session.list_skills()})
                return
            if route == "/api/mcp":
                self._send_json(session.mcp_status())
                return
            if route == "/api/map":
                mapper = getattr(session, "request_map", None)
                # The map is a projection of recorded state; a build without it must
                # answer rather than drop the connection.
                self._send_json(
                    mapper()
                    if mapper
                    else {
                        "intent": "",
                        "tools": [],
                        "note": "map unavailable in this build",
                    }
                )
                return

            if route == "/api/sessions":
                lister = getattr(session, "list_sessions", None)
                self._send_json({"sessions": lister() if lister else []})
                return
            if route == "/api/files":
                tree = getattr(session, "file_tree", None)
                self._send_json(tree() if tree else {"folders": [], "files": []})
                return
            if route == "/api/registry":
                self._send_json(session.registry())
                return
            if route == "/api/approvals":
                # #30: the pending queue the approval UI polls.
                self._send_json({"approvals": session.pending_approvals()})
                return
            if route == "/api/free_models":
                # W1.3: the free-model picker feeds the per-role dropdowns.
                try:
                    models = list_openrouter_models(
                        free_only=not bool(session.settings.paid_models_enabled)
                    )
                    self._send_json({"ok": True, "models": models})
                except LlmError as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc)[:120], "models": []}
                    )
                return
            if route == "/api/voiceprint/status":
                self._send_json(
                    {
                        "enrolled": session._voiceprint.enrolled,
                        "model_present": session._voiceprint.model_present,
                    }
                )
                return
            if route == "/api/pipeline":
                self._send_json(session.pipeline_status())
                return
            if route == "/api/docs":
                # Documentation view: whole manual + live roster summary.
                self._send_json(session.docs_payload())
                return
            if route == "/api/roles":
                # W1.2: the per-role roster with each role's model + fallbacks.
                self._send_json({"ok": True, "roles": load_agents_config()})
                return
            if route == "/api/agents":
                # The dashboard reads the roster (model + thinking) as data.
                self._send_json({"ok": True, "config": load_agents_config()})
                return
            if route == "/api/model_catalog":
                # The grouped, searchable model picker fetches this once (not polled).
                self._send_json({"ok": True, "catalog": session.full_catalog()})
                return
            if route == "/api/dir":
                # The main directory: path + listing, so the UI can show and copy it.
                self._send_json(session.dir_context())
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            body = self._body()
            if route == "/api/diagnose":
                # S2.2: the owner picks who investigates and what scope; dry by design.
                result = session.diagnose(
                    str(body.get("scope") or "all"), str(body.get("agent") or "doctor")
                )
                self._send_json(result)
                return
            if route == "/api/repair":
                # S3: one catalogue entry, verified, rolled back if it made things worse.
                result = session.repair(
                    str(body.get("id") or ""), bool(body.get("dry_run", True))
                )
                self._send_json(result)
                return
            if route == "/api/verify":
                # S3: the verifier can be asked on its own.
                self._send_json(session.verify())
                return
            if route == "/api/escalate":
                # S3.5: propose a fix in code; nothing runs until it is approved.
                self._send_json(
                    session.escalate(
                        str(body.get("question") or ""), str(body.get("evidence") or "")
                    )
                )
                return
            if route == "/api/fix_now":
                # S4.1: repair the most severe finding that has a safe repair.
                self._send_json(
                    session.fix_top_finding(bool(body.get("dry_run", False)))
                )
                return
            if route == "/api/session_report":
                # S4.2: the same report that is written at close, on demand.
                path = session.write_incident_report()
                self._send_json({"ok": bool(path), "path": path})
                return
            if route == "/api/incidents":
                # S1.2: a GUI window reports its own crash here. TEXT only - the
                # incident store never holds audio (the owner's rule: transcripts
                # only, never a voice recording).
                raw_context = body.get("context")
                context: dict[str, Any] = (
                    raw_context if isinstance(raw_context, dict) else {}
                )
                traceback_text = str(body.get("traceback") or "")[:1200]
                if traceback_text:
                    context["traceback"] = traceback_text
                incident = session.incidents.record(
                    str(body.get("kind") or "gui"),
                    str(body.get("message") or "unknown error"),
                    str(body.get("severity") or "error"),
                    where=str(body.get("where") or "gui"),
                    context=context or None,
                    traceback_id=str(body.get("traceback_id") or ""),
                )
                self._send_json({"ok": True, "incident": incident})
                return
            if route == "/api/doctor":
                if "enabled" in body:
                    session.update_settings(
                        {"doctor_enabled": bool(body.get("enabled"))}
                    )
                if "kill_switch" in body:
                    session.update_settings(
                        {"doctor_kill_switch": bool(body.get("kill_switch"))}
                    )
                self._send_json({"ok": True, "status": session._doctor.status()})
                return
            if route == "/api/persona":
                name = str(body.get("name") or "")
                persona = session.set_persona(name)
                self._send_json({"ok": True, "persona": persona})
                return

            if route == "/api/crew":
                # M3.1: the Agents tab starts a crew without going through chat.
                roles = body.get("roles")
                message = session.start_crew(
                    str(body.get("brief") or ""),
                    roles=[str(r) for r in roles] if isinstance(roles, list) else None,
                    process=str(body.get("process") or "sequential"),
                )
                self._send_json({"ok": "crew" in message.lower(), "message": message})
                return
            if route == "/api/mcp/call":
                self._send_json(
                    session.mcp_call(
                        str(body.get("server") or ""),
                        str(body.get("tool") or ""),
                        body.get("arguments")
                        if isinstance(body.get("arguments"), dict)
                        else {},
                    )
                )
                return
            if route == "/api/tool":
                name = str(body.get("name") or "").strip()
                args = body.get("arguments")
                if not name:
                    self._send_json(
                        {"ok": False, "error": "name is required"}, status=400
                    )
                    return
                if not isinstance(args, dict):
                    args = {}
                try:
                    res = session._toolbox.call(name, args)
                    self._send_json({"ok": bool(res.ok), "result": res.content})
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if route == "/api/latency_report":
                path = session.write_latency_report()
                self._send_json({"ok": bool(path), "path": path})
                return
            if route == "/api/voices/unlock":
                session._voiceprint.unlock_master()
                self._send_json({"ok": True})
                return
            if route == "/api/voices/rename":
                self._send_json(
                    session._voiceprint.rename_speaker(
                        str(body.get("id") or ""), str(body.get("label") or "")
                    )
                )
                return
            if route == "/api/voices/ignore":
                self._send_json(
                    session._voiceprint.set_ignored(
                        str(body.get("id") or ""), bool(body.get("ignored", True))
                    )
                )
                return
            if route == "/api/voices/master":
                self._send_json(
                    session._voiceprint.make_master(str(body.get("id") or ""))
                )
                return
            if route == "/api/voices/promote":
                self._send_json(
                    session._voiceprint.promote_owner(str(body.get("id") or ""))
                )
                return
            if route == "/api/voices/reset":
                self._send_json(session._voiceprint.reset_roster())
                return
            if route == "/api/voices/merge":
                self._send_json(
                    session._voiceprint.merge_speakers(
                        str(body.get("source_id") or ""),
                        str(body.get("target_id") or ""),
                    )
                )
                return
            if route == "/api/voices/threshold":
                self._send_json(
                    session._voiceprint.set_match_threshold(
                        float(body.get("threshold") or 0.28)
                    )
                )
                return
            if route == "/api/session/inspect":
                self._send_json(session.inspect_session(str(body.get("id") or "")))
                return
            if route == "/api/session/delete":
                self._send_json(session.delete_session(str(body.get("id") or "")))
                return
            if route == "/api/doctor/control":
                self._send_json(session.doctor_control(body))
                return
            if route == "/api/doctor/confirm":
                phrase = str(body.get("phrase") or "")
                force = bool(body.get("confirm"))
                res = session._doctor_gate.confirm_and_apply(
                    phrase, session, force=force
                )
                if res.get("ok"):
                    session._notify_doctor_outcome(res, source="user")
                self._send_json(res)
                return
            if route == "/api/doctor/cancel":
                prev = session._doctor_gate.cancel()
                if prev:
                    session._notify_doctor_outcome(
                        {
                            "ok": True,
                            "cancelled": True,
                            "summary": prev.get("summary") or prev.get("action"),
                        },
                        source="user-cancel",
                    )
                self._send_json({"ok": True})
                return
            if route == "/api/session_profile/clear":
                with session.lock:
                    session._session_profile = []
                    session._save_session_profile()
                self._send_json({"ok": True})
                return
            if route == "/api/voiceprint/enroll":
                self._send_json(
                    session.enroll_voiceprint(int(body.get("seconds") or 10))
                )
                return
            if route == "/api/voiceprint/clear":
                session._voiceprint.clear()
                self._send_json({"ok": True, "enrolled": False})
                return
            if route == "/api/classify":
                dec = session.classify_decision(str(body.get("text") or ""))
                self._send_json({"ok": True, **dec.to_dict()})
                return
            if route == "/api/community":
                res = session.post_community(
                    str(body.get("from_role") or "agent"),
                    str(body.get("message") or ""),
                    str(body.get("to_role") or "front"),
                )
                self._send_json(res, status=200 if res.get("ok") else 400)
                return
            if route == "/api/delegate":
                # Background delegation: Front stays available while a Back pool worker works.
                message = session.delegate_task(
                    str(body.get("brief") or ""), str(body.get("role") or "back")
                )
                self._send_json({"ok": "asked" in message, "message": message})
                return
            if route == "/api/ask_agent":
                # A2A: one roster role answers one question, inside the request.
                q = (
                    body.get("question")
                    or body.get("prompt")
                    or body.get("message")
                    or body.get("task")
                    or ""
                )
                reply = session.ask_agent_turn(
                    str(body.get("role") or ""),
                    str(q),
                    str(body.get("context") or ""),
                    from_role=str(body.get("from_role") or "front"),
                )
                self._send_json({"ok": True, "reply": reply})
                return

            if route == "/api/ab":
                # M1.4: the same question to two roles, side by side.
                result = session.ab_compare(
                    str(body.get("text") or ""),
                    str(body.get("role_a") or ""),
                    str(body.get("role_b") or ""),
                )
                self._send_json(result)
                return

            if route == "/api/attach":
                # Attachments (images pasted in chat) are saved to disk so the agent can
                # read them with its own tools; the event makes them visible in the UI.
                import base64

                name = str(body.get("name") or "attachment.png")[:80]
                data64 = str(body.get("data64") or "")
                # Pin the stored name to the stamp: the user suffix keeps only the
                # extension, so nothing the client sends can escape the attachments
                # folder (no ../, no absolute paths, no double extensions games).
                suffix = (
                    "".join(c for c in name.rsplit(".", 1)[-1] if c.isalnum())[:8]
                    or "png"
                )
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_dir = BASE / "out" / "attachments"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / (stamp + "_attachment." + suffix.lower())
                try:
                    blob = base64.b64decode(data64, validate=True)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        {"ok": False, "error": "bad base64: " + str(exc)[:60]},
                        status=400,
                    )
                    return
                path.write_bytes(blob)
                session._persist_event(
                    "attach",
                    {"msg": str(path.name), "path": str(path), "bytes": len(blob)},
                )
                self._send_json({"ok": True, "path": str(path), "bytes": len(blob)})
                return

            if route == "/api/folder":
                import filereader

                folder = str(body.get("path") or "").strip()
                if not folder:
                    self._send_json(
                        {"ok": False, "error": "path is required"}, status=400
                    )
                    return
                try:
                    granted = filereader.allow_folder(folder)
                except filereader.FileError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                self._send_json({"ok": True, "folder": str(granted)})
                return
            if route == "/api/settings":
                session.update_settings(body)
                self._send_json({"ok": True})
                return
            if route == "/api/send":
                text = str(body.get("text") or "")
                if not text.strip():
                    self._send_json({"ok": False, "error": "empty text"}, status=400)
                    return
                session.submit_text(text)
                self._send_json({"ok": True})
                return
            if route == "/api/listen":
                mode = str(body.get("mode") or "continuous")
                session.start_listening(mode)
                self._send_json({"ok": True, "mode": mode})
                return
            if route == "/api/shutdown":
                # Cut active speech/listening right away, then exit cleanly.
                with contextlib.suppress(Exception):
                    session.interrupt()
                    session.stop_listening()
                self._send_json({"ok": True, "stopping": True})
                threading.Thread(
                    target=_request_shutdown, args=(self.server,), daemon=True
                ).start()
                return
            if route == "/api/stop":
                session.stop_listening()
                self._send_json({"ok": True})
                return
            if route == "/api/barge":
                session.interrupt()
                self._send_json({"ok": True})
                return
            if route == "/api/device":
                idx = body.get("input_index")
                if idx is None:
                    self._send_json(
                        {"ok": False, "error": "input_index required"}, status=400
                    )
                    return
                self._send_json(session.set_input_device(int(idx)))
                return
            if route == "/api/pi/prompt":
                # Drive one role's pi terminal pane (the bottom-half panes).
                role = str(body.get("role") or "front")
                text = str(body.get("text") or "")
                if not text.strip():
                    self._send_json({"ok": False, "error": "empty text"}, status=400)
                    return
                self._send_json({"ok": True, "message": session.pi_prompt(role, text)})
                return
            if route == "/api/pi/abort":
                self._send_json(
                    {
                        "ok": True,
                        "message": session.pi_abort(str(body.get("role") or "")),
                    }
                )
                return
            if route == "/api/reset":
                session.reset()
                self._send_json({"ok": True})
                return
            if route == "/api/session/new":
                info = session.new_session()
                self._send_json(info)
                return
            if route == "/api/restart":
                with contextlib.suppress(Exception):
                    session.interrupt()
                    session.stop_listening()
                    session.pi_abort("")
                self._send_json({"ok": True, "restarting": True})
                session.schedule_restart(self.server)
                return
            if route == "/api/session/load":
                # #60/#9: load a past conversation into the active session so chat continues cleanly.
                res = session.activate_session(str(body.get("id") or ""))
                self._send_json(res)
                return
            if route == "/api/session/resume":
                # #26: make a past session current again; new turns append there.
                session_id = session.resume_session(str(body.get("id") or ""))
                self._send_json(
                    {"ok": not session_id.startswith("no such"), "id": session_id}
                )
                return
            if route == "/api/session/fork":
                # #9 fork: same past, independent future.
                new_id = session.fork_session(str(body.get("id") or ""))
                self._send_json({"ok": not new_id.startswith("no such"), "id": new_id})
                return
            if route == "/api/approval":
                # #30: the owner's answer to a queued risky action.
                result = session.respond_approval(
                    str(body.get("id") or ""), bool(body.get("approved"))
                )
                self._send_json(result)
                return
            if route == "/api/set_role_model":
                # W1.2: per-role model switch (routing is data in agents.json).
                message = session.set_role_model(
                    str(body.get("role") or ""), str(body.get("model") or "")
                )
                self._send_json({"ok": "will use" in message, "message": message})
                return
            if route == "/api/set_role_thinking":
                # The dashboard writes a role's thinking level without a restart.
                message = session.set_role_thinking(
                    str(body.get("role") or ""), str(body.get("level") or "")
                )
                self._send_json({"ok": "will think" in message, "message": message})
                return
            if route == "/api/agents":
                # Save the whole roster (model + fallbacks + thinking) at once.
                config = body.get("config")
                if not isinstance(config, dict):
                    self._send_json(
                        {"ok": False, "error": "config must be an object"}, status=400
                    )
                    return
                path = session.apply_agent_config(config)
                self._send_json({"ok": True, "path": str(path)})
                return
            if route == "/api/greet":
                # #57: pre-warmed startup greeting.
                self._send_json({"ok": True, "greeting": session.greet()})
                return
            if route == "/api/speak_script":
                # #24: speak a NOVA:/PIATTOS: script with each persona's voice.
                spoken = session.speak_script(str(body.get("script") or ""))
                self._send_json({"ok": spoken > 0, "lines": spoken})
                return
            self.send_error(404, "not found")

    return Handler


_EL_USAGE_CACHE: dict[str, Any] = {"at": 0.0, "data": {}}


def elevenlabs_usage(api_key: str) -> dict[str, Any]:
    """Credit analytics for ElevenLabs: subscription balance + session usage.

    Cached 60s so the live UI does not hammer the API. Never returns the key.
    """
    import urllib.request

    from tts import cloud_chars

    if not api_key:
        return {"ok": False, "error": "no ElevenLabs key configured"}
    now = time.monotonic()
    if now - float(_EL_USAGE_CACHE["at"]) < 60 and _EL_USAGE_CACHE["data"]:
        data = dict(_EL_USAGE_CACHE["data"])
    else:
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": api_key},
            )
            sub = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            used = int(sub.get("character_count", 0))
            limit = int(sub.get("character_limit", 0))
            data = {
                "ok": True,
                "tier": sub.get("tier", ""),
                "status": sub.get("status", ""),
                "character_count": used,
                "character_limit": limit,
                "remaining": max(0, limit - used),
                "percent_used": round(100 * used / limit, 1) if limit else 0,
                "reset_unix": sub.get("next_character_count_reset_unix", 0),
            }
            _EL_USAGE_CACHE["at"] = now
            _EL_USAGE_CACHE["data"] = data
        except Exception as exc:  # noqa: BLE001 - analytics must not break a turn
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    data = dict(data)
    data["session_chars"] = cloud_chars().get("elevenlabs", 0)
    data["credit_score"] = round(100 - float(data.get("percent_used", 0)), 1)
    data["est_overage_usd"] = round(
        max(
            0, int(data.get("character_count", 0)) - int(data.get("character_limit", 0))
        )
        / 1000
        * 0.30,
        4,
    )
    return data


def apply_pi_profile(settings: Settings) -> Settings:
    """Overlay pi_profile.json when SONJ_PROFILE=pi (light models for a 4 GB Pi)."""
    path = BASE / "pi_profile.json"
    try:
        prof = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    asr = prof.get("asr") or {}
    if asr.get("backend"):
        settings.asr_backend = str(asr["backend"])
    tts = prof.get("tts") or {}
    if tts.get("voice"):
        voice = "Neural: " + str(tts["voice"])
        settings.tts_voice = voice
        settings.nova_voice = voice
        settings.piattos_voice = voice
    if prof.get("idle_unload_s"):
        settings.idle_unload_s = float(prof["idle_unload_s"])
    return settings


def load_settings() -> Settings:
    """Config file wins over defaults; CLI flags win over both."""
    settings = Settings()
    if CONFIG_PATH.exists():
        try:
            # utf-8-sig: a BOM written by an external editor must not make the
            # whole config silently unreadable (it once reverted the provider).
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
            for key, value in stored.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
        except (OSError, ValueError) as exc:
            print(f"[config] ignoring unreadable {CONFIG_PATH.name}: {exc}")
    if os.environ.get("SONJ_PROFILE", "").strip().lower() == "pi":
        settings = apply_pi_profile(settings)
    if not settings.openrouter_key:
        found = find_openrouter_key()
        if found:
            settings.openrouter_key = found
            log_event("OpenRouter key found on this machine (memory only, never saved)")
    if not settings.elevenlabs_api_key:
        found_el = find_elevenlabs_key()
        if found_el:
            settings.elevenlabs_api_key = found_el
            log_event("ElevenLabs key found on this machine (memory only, never saved)")
    # ME3_ASR_BACKEND is the documented knob for the MCP server, and it has to
    # apply to every entry point too - otherwise a machine that only downloaded
    # one model fails with a confusing "backend unavailable" further down.
    override_backend = os.environ.get("ME3_ASR_BACKEND", "").strip()
    if override_backend:
        settings.asr_backend = override_backend
    settings = repair_provider_model(settings)
    settings.barge_rms = clamp_barge_rms(settings.barge_rms)
    return settings


def _selftest() -> int:
    """Check the tuning helpers: barge-in decisions and config safety."""
    cases: list[tuple[str, tuple[float, float, float, float], bool]] = [
        ("echo inside the grace window", (0.50, 800.0, 100.0, 0.06), False),
        ("short blip mid-reply", (0.50, 120.0, 3000.0, 0.06), False),
        ("quiet noise mid-reply", (0.01, 800.0, 3000.0, 0.06), False),
        ("loud sustained interruption", (0.25, 800.0, 3000.0, 0.06), True),
        ("your voice at the floor", (0.021, 800.0, 3000.0, 0.02), True),
    ]
    failures = 0
    print("barge-in decisions:")
    for title, args, want in cases:
        got = should_interrupt(*args)
        ok = got == want
        failures += 0 if ok else 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {title:30s} interrupt={got} (want {want})"
        )

    print("echo detection (the assistant must not answer itself):")
    echo_cases = [
        (
            "verbatim own reply",
            (
                "I could not reach the language model",
                "I could not reach the language model",
                True,
            ),
        ),
        (
            "own reply, asr variant",
            (
                "could not reach the language model olive",
                "I could not reach the language model: Ollama chat failed",
                True,
            ),
        ),
        (
            "genuine new question",
            (
                "what is the weather in manila",
                "The current time is Monday 02 November",
                False,
            ),
        ),
        ("too short to judge", ("hi", "The current time is Monday", False)),
        (
            "robotic mis-hearing just after speech (the Pi's espeak)",
            (
                "The memory light is now about 40% right",
                "The bedroom light is now on at forty percent brightness",
                True,
                True,
            ),
        ),
        (
            "similar words, long after speaking, is real speech",
            (
                "The memory light is now about 40% right",
                "The bedroom light is now on at forty percent brightness",
                False,
                False,
            ),
        ),
    ]
    for title, case in echo_cases:
        transcript, spoken_text = str(case[0]), str(case[1])
        recent = bool(case[2]) if len(case) == 4 else True
        want = bool(case[-1])
        got = is_echo(transcript, spoken_text, recent=recent)
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {title:24s} echo={got} (want {want})")

    print("config safety:")
    for stored, want in ((0.005, MIN_BARGE_RMS), (0.5, 0.5), ("junk", 0.06)):
        got = clamp_barge_rms(stored)
        ok = abs(got - want) < 1e-9
        failures += 0 if ok else 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] stored={stored!r:8s} -> {got} (want {want})"
        )

    live = load_settings()
    if live.provider == "ollama":
        try:
            installed = list_ollama_models()
        except LlmError as exc:
            installed = []
            print(f"  [SKIP] cannot list Ollama models: {exc}")
        if installed:
            ok = live.model in installed
            failures += 0 if ok else 1
            print(
                f"  [{'PASS' if ok else 'FAIL'}] stored model '{live.model}' exists in Ollama"
            )
    ok = live.barge_rms >= MIN_BARGE_RMS
    failures += 0 if ok else 1
    print(
        f"  [{'PASS' if ok else 'FAIL'}] loaded config gives barge_rms={live.barge_rms} "
        f"(floor {MIN_BARGE_RMS})"
    )
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


def _pid_alive(pid: int) -> bool:
    """True when this pid is a live process (psutil when available)."""
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001 - without psutil, fall back to os.kill(0)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _claim_engine(
    port: int = 8766, host: str = "127.0.0.1", force: bool = False
) -> tuple[bool, str]:
    """S0: one engine, one voice.

    A second engine means a second audio stack - two voices fighting for the
    speakers. This claims the single-engine lock BEFORE anything opens a device:
    an already-running engine makes this process exit instead of competing.

    Returns (ok, reason). `force` replaces the running engine instead of refusing.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pid = 0
    try:
        pid = int(ENGINE_PID.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        pid = 0
    alive = _pid_alive(pid) and pid != os.getpid()
    healthy = engine_health(port, host)
    if alive or healthy:
        if not force:
            who = f"pid {pid}" if alive else "an engine without a pid file"
            return (
                False,
                f"another engine is already running ({who}); refusing to start a second voice",
            )
        # force: stop only older engines belonging to THIS installation (leave sibling versions untouched).
        for other in engine_processes():
            other_pid = int(other.get("pid") or 0)
            cmd = str(other.get("cmdline") or "")
            if (
                other_pid
                and other_pid != os.getpid()
                and (BASE.name in cmd or str(BASE) in cmd)
            ):
                with contextlib.suppress(OSError):
                    os.kill(other_pid, 15)
        time.sleep(1.0)
    try:
        ENGINE_PID.write_text(str(os.getpid()), encoding="utf-8")
        ENGINE_LOCK.write_text(
            json.dumps(
                {"pid": os.getpid(), "port": port, "host": host, "at": time.time()}
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"could not write the engine lock: {exc}"
    return True, "engine claimed"


def _release_engine() -> None:
    """Drop the lock only when this process owns it.

    A failure here is not interesting - the process is exiting - but swallowing
    it must still be visible as a decision, so it is an explicit suppression
    rather than a bare pass.
    """
    with contextlib.suppress(OSError, ValueError):
        if int(ENGINE_PID.read_text(encoding="utf-8").strip() or "0") == os.getpid():
            ENGINE_PID.unlink()
        ENGINE_LOCK.unlink()


def _install_exit_handlers(server: ThreadingHTTPServer) -> None:
    """Return the engine to its default state on every exit path.

    Ctrl+C already worked. A closed terminal window and a CLI stop did not: on
    Windows those terminate the process without running Python cleanup, leaving
    the audio device held and the engine lock behind. Both now route through the
    same graceful shutdown the /api/shutdown route uses.
    """

    def _graceful(*_args: object) -> None:
        threading.Thread(target=_request_shutdown, args=(server,), daemon=True).start()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(Exception):
                signal.signal(sig, _graceful)

    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _on_console(event: int) -> bool:
            if event in (2, 5, 6):  # CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
                _graceful()
                # Wait for finally teardown (max 4.5s under Windows console-close budget).
                _SHUTDOWN_DONE.wait(timeout=4.5)
                return True
            return False

        keep = handler_type(_on_console)  # the callback must outlive registration
        ctypes.windll.kernel32.SetConsoleCtrlHandler(keep, True)
        globals()["_CONSOLE_HANDLER"] = keep
    except Exception:  # noqa: BLE001 - no console (pythonw) is not a failure
        return


_CONSOLE_HANDLER: Any = None


def _request_shutdown(server: ThreadingHTTPServer) -> None:
    """Ask serve_forever() to return, so the normal teardown in main() runs.

    Called from a separate thread: shutdown() blocks until the loop exits, so
    calling it from the loop thread (or from inside a request handler) would
    deadlock. This is the one exit path every caller shares - the CLI's stop
    command, a signal, and a closed terminal window all land here.
    """
    with contextlib.suppress(Exception):
        server.shutdown()


# Public names: the desktop app and the CLI need the same single-engine gate, so
# they import these instead of reaching into the private helpers.
claim_engine = _claim_engine
release_engine = _release_engine
request_shutdown = _request_shutdown


def _install_console_close_handler(on_close: Callable[[], None]) -> None:
    """Windows: closing the terminal window must also return the engine to default.

    A console close is not a Python exception. Without this the process is
    terminated outright and the audio device and the engine lock are left behind,
    so the next start finds a lock nobody owns. The OS grants about five seconds
    before it kills the process, so the handler waits for the teardown.
    """
    global _CONSOLE_HANDLER
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(event: int) -> bool:
            if event in (2, 5, 6):  # CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
                with contextlib.suppress(Exception):
                    on_close()
                time.sleep(4.0)
                return True
            return False

        _CONSOLE_HANDLER = handler_type(_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_CONSOLE_HANDLER, True)
    except Exception:  # noqa: BLE001 - a windowed process has no console to hook
        return


def _sweep_orphan_panes() -> int:
    """Kill pi RPC pane trees whose owning engine is gone (no dead processes).

    A pane is owned when some live engine (python server.py) is its ancestor.
    Interactive pi sessions have no --mode rpc flag and are never touched.
    """
    if os.name != "nt":
        return 0
    try:
        import psutil
    except ImportError:
        return 0
    engine_pids = {os.getpid()}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            name = str(proc.info.get("name") or "").lower()
            if "python" in name and "server.py" in cmd:
                engine_pids.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "--mode rpc" not in cmd:
                continue
            owned = False
            cur = proc
            for _ in range(8):
                parent = cur.parent()
                if parent is None:
                    break
                if parent.pid in engine_pids:
                    owned = True
                    break
                cur = parent
            if not owned:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                killed += 1
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            OSError,
            subprocess.SubprocessError,
        ):
            continue
    if killed:
        log_event(f"swept {killed} orphan pi pane process tree(s)")
    return killed


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()

    try:
        import faulthandler

        OUT_DIR.mkdir(exist_ok=True)
        _crash_fh = (OUT_DIR / "crash_dump.log").open("a", encoding="utf-8")
        faulthandler.enable(_crash_fh)
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="SONJ4 voice agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--provider", choices=["ollama", "openrouter", "github-copilot"]
    )
    parser.add_argument("--model")
    parser.add_argument("--asr", choices=["whisper", "moonshine", "parakeet"])
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--openrouter-key", default="")
    # RULE-1/#35-37: the browser console is opt-in now, not the default face.
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    # S0: replace a running engine instead of refusing to start a second one.
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ok, why = _claim_engine(args.port, args.host, force=bool(args.force))
    if not ok:
        print(f"[engine] {why}")
        print("[engine] stop the running engine first, or pass --force to replace it")
        return 2

    _sweep_orphan_panes()
    settings = load_settings()
    if args.provider:
        settings.provider = args.provider
    if args.model:
        settings.model = args.model
    if args.asr:
        settings.asr_backend = args.asr
    if args.no_tts:
        settings.tts_enabled = False
    if args.openrouter_key:
        settings.openrouter_key = args.openrouter_key

    session = Session(settings)
    session._listen_port = int(args.port)
    session.engine_url = f"http://{args.host}:{args.port}"
    OUT_DIR.mkdir(exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), _handler_factory(session))
    url = f"http://{args.host}:{args.port}/"
    print("=" * 68)
    print("SONJ4 voice agent (1 Front + 1 Classifier + 1 Doctor + 5 Back pool)")
    print(f"  UI        {url}")
    print(f"  provider  {settings.provider}/{settings.model}")
    print(f"  ASR       {settings.asr_backend} (offline, sherpa-onnx)")
    print(f"  TTS       {'Windows SAPI' if settings.tts_enabled else 'disabled'}")
    print(f"  devices   in={session.input_device} out={session.output_device}")
    print("  Ctrl+C to stop")
    print("=" * 68)
    if args.browser and not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    # S1: serve IMMEDIATELY so the window connects in <1s; warm everything in the
    # background. The mic waits on _prewarm_done before streaming (sherpa-onnx
    # hot-init while the mic streams can segfault, so listening gates on it).
    def _prewarm() -> None:
        t0 = time.monotonic()
        print("[startup] loading voices...")
        with contextlib.suppress(Exception):
            session.prewarm_tts()
        with contextlib.suppress(Exception):
            session._voiceprint.warm()
        session._prewarm_done.set()
        print(f"[startup] voices ready in {time.monotonic() - t0:.1f}s")
        with contextlib.suppress(Exception):
            session.greet()
        with contextlib.suppress(Exception):
            if session._should_use_front_pi():
                session.warm_all_panes()
        with contextlib.suppress(Exception):
            session.warm_extras()
        # S4: Doctor fast boot scan — detect and log problems present at opening.
        with contextlib.suppress(Exception):
            from doctor import fast_boot_scan

            session.startup_check = fast_boot_scan()
            log_event(
                f"startup check: {session.startup_check.get('status')} "
                f"({session.startup_check.get('recent_errors', 0)} recent errors)"
            )
        print(f"[startup] ready in {time.monotonic() - t0:.1f}s")

    threading.Thread(target=_prewarm, daemon=True, name="prewarm").start()

    # Exit-to-default: every way out of this process runs the same teardown.
    _install_exit_handlers(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        try:
            with contextlib.suppress(Exception):
                session._pending.put(("shutdown", None))
            session.close()
            with contextlib.suppress(Exception):
                session.write_incident_report()
            session.save_config()
            server.server_close()
            _release_engine()
        finally:
            _SHUTDOWN_DONE.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
