"""A local, fast intent router.

The cloud model is good but it is 1.6-2.2 s away per call, and a turn needs two calls.
A local 7B model answers a bounded classification in ~200 ms (measured: 118 ms to first
token, 209 ms total, returning the bare label), so the router decides *what kind of turn
this is* before anything slow starts:

    command  - one of the fixed device commands: run the tool and speak its result
    chat     - conversational: the cloud model handles it
    complex  - open-ended: hand it to the harness in the background

It also picks the acknowledgement to speak while the real answer is being prepared,
which is how human conversation covers thinking time.

Fail-safe: any error, timeout, or missing model returns "chat" - exactly the behaviour
the assistant had before, so adding the router can never make a turn worse.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_ROUTER_MODEL = "qwen2.5-coder:7b"
DEFAULT_TIMEOUT_MS = 700.0
LABELS = ("command", "chat", "complex")

# Spoken while the real answer is still prepared. Keys match tts.FILLER_PHRASES, which
# owns the wording and renders the audio.
FILLER_FOR = {"command": "okay", "chat": "hmm", "complex": "one_moment"}
DEFAULT_FILLER = "hmm"

# When the router is unavailable, this keyword test still routes the common case.
COMMAND_HINTS = (
    "light",
    "lamp",
    "timer",
    "alarm",
    "remind",
    "time",
    "clock",
    "weather",
    "temperature",
    "thermostat",
    "volume",
    "music",
    "song",
    "track",
    "play",
    "pause",
    "skip",
    "next",
    "previous",
    "call",
    "dial",
    "silence",
    "quiet",
    "turn on",
    "turn off",
    "dim",
    "brighten",
    "calculate",
    "plus",
    "minus",
)
COMPLEX_HINTS = (
    "research",
    "find out",
    "look up",
    "search",
    "more of these",
    "headline",
    "news",
    "top ",
    "latest",
    "price",
    "prices",
    "write a",
    "write me",
    "draft",
    "plan",
    "build",
    "create",
    "game",
    "website",
    "webpage",
    "bio link",
    "linktree",
    "make a",
    "compare",
    "analyse",
    "analyze",
    "summarise",
    "summarize",
    "summary",
    "condense",
    "verify",
    "audit",
    "doctor",
    "diagnos",
    "assess",
    "troubleshoot",
    "error",
    "errors",
    "log",
    "logs",
    "setting",
    "settings",
    "harness",
    "delegate",
    "code",
    "script",
    "automate",
)

PROMPT = (
    "You route a single utterance for a voice assistant that runs on a small device.\n"
    "Answer with exactly one word, no punctuation, no explanation.\n"
    "command = one of the fixed device commands (lights, timer, alarm, reminder, clock,\n"
    "  weather, temperature, volume, music/media, call, silence, arithmetic).\n"
    "complex = open-ended work needing research, writing, planning, building or code.\n"
    "chat = everything else, including questions and small talk.\n"
    "Utterance: {text}\n"
    "Answer:"
)


@dataclass
class RouterDecision:
    """What the router decided, and how it knows."""

    label: str
    ms: float = 0.0
    source: str = "fallback"


class Router:
    """Classifies a turn locally, with a hard timeout and a safe default."""

    def __init__(
        self,
        model: str = DEFAULT_ROUTER_MODEL,
        url: str = OLLAMA_URL,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        enabled: bool = True,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        try:
            requested = float(timeout_ms)
        except (TypeError, ValueError):
            requested = DEFAULT_TIMEOUT_MS
        self.timeout_ms = max(50.0, requested)
        self.enabled = enabled
        self.keep_alive = keep_alive
        self._available: bool | None = None
        self._checked_at = 0.0
        self.last_ms = 0.0

    # ------------------------------------------------------------------ probing
    def available(self, force: bool = False) -> bool:
        """Is the local model reachable? Cached for 30 s so turns stay cheap."""
        if not self.enabled:
            return False
        now = time.perf_counter()
        if self._available is not None and not force and now - self._checked_at < 30.0:
            return self._available
        import requests

        try:
            response = requests.get(f"{self.url}/api/tags", timeout=1.5)
            response.raise_for_status()
            names = [str(m.get("name", "")) for m in response.json().get("models", [])]
            self._available = any(
                n == self.model or n.startswith(self.model.split(":")[0]) for n in names
            )
        except Exception:  # noqa: BLE001 - an unreachable router is not an error
            self._available = False
        self._checked_at = now
        return self._available

    def warm(self) -> bool:
        """Load the model once so the first real turn does not pay the load cost.

        An unloaded 7B model takes ~11 s to page in; the router would time out and fall
        back to keywords for the first turn of every session. One throwaway call here
        fixes that, and keep_alive holds it for the session.
        """
        if not self.available(force=True):
            return False
        import requests

        try:
            requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 1, "temperature": 0},
                },
                timeout=30.0,
            )
            return True
        except Exception:  # noqa: BLE001 - warming is best effort
            return False

    # ------------------------------------------------------------------ routing
    def classify(self, text: str) -> RouterDecision:
        """Label one utterance. Never raises and never blocks longer than timeout_ms."""
        stripped = text.strip()
        if not stripped:
            return RouterDecision("chat", 0.0, "empty")
        if not self.available():
            return self._heuristic(stripped, "keywords")

        import requests

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": PROMPT.format(text=stripped),
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 4, "temperature": 0},
                },
                timeout=self.timeout_ms / 1000.0,
            )
            response.raise_for_status()
            raw = str(response.json().get("response", ""))
        except Exception:  # noqa: BLE001 - fall back rather than fail a turn
            elapsed = (time.perf_counter() - started) * 1000.0
            return self._heuristic(stripped, f"timeout {elapsed:.0f}ms")

        elapsed = (time.perf_counter() - started) * 1000.0
        self.last_ms = elapsed
        label = self._parse(raw)
        if label is None:
            return self._heuristic(stripped, f"unparsed {raw[:16]!r}")
        return RouterDecision(label, elapsed, "local")

    @staticmethod
    def _parse(raw: str) -> str | None:
        lowered = raw.lower()
        for label in LABELS:
            if re.search(rf"\b{label}\b", lowered):
                return label
        return None

    def _heuristic(self, text: str, source: str) -> RouterDecision:
        lowered = text.lower()
        if any(hint in lowered for hint in COMPLEX_HINTS):
            return RouterDecision("complex", 0.0, source)
        if any(hint in lowered for hint in COMMAND_HINTS):
            return RouterDecision("command", 0.0, source)
        return RouterDecision("chat", 0.0, source)

    def classify_cheap(self, text: str) -> RouterDecision:
        """Keyword-only routing: microseconds, and safe on a turn's critical path.

        The local model costs 450-930 ms; the acknowledgement must not wait for it, so
        the filler is chosen from keywords and the model is only consulted when a turn
        actually needs a routing hint.
        """
        stripped = text.strip()
        if not stripped:
            return RouterDecision("chat", 0.0, "empty")
        return self._heuristic(stripped, "keywords")

    def filler_for(self, decision: RouterDecision, text: str = "") -> str:
        """Which pre-rendered acknowledgement suits this turn."""
        label = decision.label
        if label == "chat" and decision.source in {"keywords", "empty"}:
            label = self._heuristic(text, "keywords").label
        return FILLER_FOR.get(label, DEFAULT_FILLER)

    # ------------------------------------------------------------- smart turn (M2.5)
    # Words that end an utterance in mid-air. The list is deliberately short and
    # syntactic: "call me tomorrow at" is unfinished, "call me at six" is not.
    INCOMPLETE_TAILS = frozenset(
        {
            "a",
            "an",
            "the",
            "at",
            "for",
            "to",
            "with",
            "and",
            "or",
            "of",
            "in",
            "on",
            "my",
            "your",
            "is",
            "this",
            "that",
            "next",
            "about",
            "by",
            "from",
            # W2: fillers & continuation signals — the speaker is still holding the floor.
            "uh",
            "um",
            "erm",
            "like",
            "so",
            "because",
            "but",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "please",
            "if",
            "then",
            "than",
            "as",
            "into",
            "onto",
            "upon",
        }
    )

    COMPLETE_PROMPT = (
        "You judge one spoken utterance for a voice assistant. The user may have "
        "paused mid-sentence. Answer with exactly one word: COMPLETE if the thought "
        "is finished, INCOMPLETE if it visibly trails off (a dangling preposition, "
        "conjunction, or article, or clearly half a request).\n"
        "Utterance: {text}\nAnswer:"
    )

    def thought_complete(self, text: str) -> bool:
        """M2.5: is this utterance a finished thought? Fail-open by design.

        The cheap syntactic check runs first (microseconds). Only when it is
        inconclusive is the local model asked - and any error, timeout or
        missing model answers True, which is exactly the previous behaviour:
        the turn ends. So this can only ever keep a turn open when the evidence
        is positive, never break one.
        """
        stripped = str(text).strip()
        if not stripped:
            return True
        words = re.findall(r"[a-z']+", stripped.lower())
        if words and words[-1] in self.INCOMPLETE_TAILS:
            return False
        if not self.available():
            return True
        import requests

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.COMPLETE_PROMPT.format(text=stripped),
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 3, "temperature": 0},
                },
                timeout=self.timeout_ms / 1000.0,
            )
            response.raise_for_status()
            raw = str(response.json().get("response", "")).lower()
        except Exception:  # noqa: BLE001 - fail-open, never break a turn
            return True
        return "incomplete" not in raw

    def summarize(self, text: str, max_words: int = 60) -> str:
        """M4.2: a local, cheap summary of a dropped conversation span.

        Runs on the same local model as routing; returns "" on any failure so
        the caller falls back to a plain trim.
        """
        stripped = str(text).strip()
        if not stripped or not self.available():
            return ""
        import requests

        prompt = (
            "Summarise this conversation span in at most "
            f"{max_words} words. Keep names, decisions and open questions. "
            "Reply with the summary only, no preamble.\n\n" + stripped[:3800]
        )
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_predict": 220, "temperature": 0.2},
                },
                timeout=6.0,
            )
            response.raise_for_status()
            return " ".join(str(response.json().get("response", "")).split())[:1200]
        except Exception:  # noqa: BLE001 - compaction falls back to the plain trim
            return ""
