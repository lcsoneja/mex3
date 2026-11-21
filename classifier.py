"""SONJ4 Decision Classifier — local ONNX + rules default (<100 ms) + optional cloud classifiers.

Sits between the user utterance/message and Front. Takes a structured
DecisionRequest (text + compact session context) and returns a fixed-choice
Decision (`front` | `back` | `doctor`) with zero LLM prose or question
generation.

Backends:
    local-onnx   Default when models/classifier/intent.onnx + onnxruntime exist.
                 Runs Stage-1 rules + tiny local ONNX MLP (<1 ms, offline).
    local-rules  Pure Stage-1 deterministic rule/keyword feature scorer (<0.2 ms).
    openrouter/<model_id>
                 OpenRouter TypeSafe System-One classifier (/api/v1/systemone),
                 falling back cleanly to local-onnx/local-rules on timeout/error.
    openrouter-chat:<model_id>
                 Fast OpenRouter chat model used as a bounded JSON/label router.
    ollama:<model_id>
                 Local Ollama router (http://127.0.0.1:11434), with local fallback.

Bias: +0.10 added to score_front in the Front-vs-Back decision; Back must beat
Front by >0.10 to dispatch a pool worker. Low-confidence ties abstain to `front`.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import re
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

RouteChoice = Literal["front", "back", "doctor"]
CHOICES: tuple[RouteChoice, ...] = ("front", "back", "doctor")

FRONT_BIAS = 0.10
DEADLINE_MS = 90.0
CLOUD_DEADLINE_MS = 2500.0
ABSTAIN_FLOOR = 0.50
DOCTOR_THRESHOLD = 0.65

BASE = Path(__file__).resolve().parent
ONNX_DIR = BASE / "models" / "classifier"
OPENROUTER_SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Stage 2 callable signature: req -> (score_front, score_back, score_doctor, reason_code)
Stage2Fn = Callable[["DecisionRequest"], tuple[float, float, float, str]]

DEFAULT_CLASSIFIER_BACKEND = "local-onnx"

CLASSIFIER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "local-onnx",
        "label": "Local ONNX (intent.onnx, <1ms, offline)",
        "kind": "local",
        "paid": False,
        "notes": "Default tiny local MLP ONNX model + Stage-1 rules (+10% Front bias)",
    },
    {
        "id": "local-rules",
        "label": "Local Rules (deterministic keywords, <0.1ms, offline)",
        "kind": "local",
        "paid": False,
        "notes": "Zero-model deterministic keyword scorer",
    },
    {
        "id": "openrouter/perplexity/pplx-decider-flash-v1.3:free",
        "label": "OpenRouter · Perplexity Decider Flash v1.3 (free)",
        "kind": "openrouter-systemone",
        "paid": False,
        "notes": "System-One classifier over OpenRouter (~550ms RTT, free)",
    },
    {
        "id": "openrouter/perplexity/pplx-decider-v1.3:free",
        "label": "OpenRouter · Perplexity Decider v1.3 (free)",
        "kind": "openrouter-systemone",
        "paid": False,
        "notes": "System-One classifier over OpenRouter (~1.1s RTT, free)",
    },
    {
        "id": "openrouter/parallel/decisions-fast",
        "label": "OpenRouter · Parallel Decisions Fast ($0)",
        "kind": "openrouter-systemone",
        "paid": False,
        "notes": "System-One classifier over OpenRouter (~1.1s RTT, $0 input)",
    },
    {
        "id": "openrouter-chat:liquid/lfm-3-1.7b:free",
        "label": "OpenRouter · Liquid LFM-3 1.7B Router (free)",
        "kind": "openrouter-chat",
        "paid": False,
        "notes": "Fast compact chat model prompted for front/back/doctor choice (free)",
    },
    {
        "id": "openrouter/openai/gpt-6-luna-decisions",
        "label": "OpenRouter · OpenAI GPT-6 Luna Decisions (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "1.05M ctx System-One decision model ($0.10/1M in)",
    },
    {
        "id": "openrouter/~typesafe/jev-latest",
        "label": "OpenRouter · TypeSafe Jev Latest (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "System-One decision model ($0.042/1M in)",
    },
    {
        "id": "openrouter/liquid/d1.2",
        "label": "OpenRouter · LiquidAI d1.2 (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "Compact fast System-One decision model ($0.02/1M in)",
    },
    {
        "id": "openrouter/google/gemini-decision-1",
        "label": "OpenRouter · Google Gemini Decision 1 (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "System-One decision model ($0.10/1M in)",
    },
    {
        "id": "openrouter/cloudflare/clef-flash",
        "label": "OpenRouter · Cloudflare Clef Flash (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "System-One decision model ($0.019/1M in)",
    },
    {
        "id": "openrouter/microsoft/microsoft-decision-1.1",
        "label": "OpenRouter · Microsoft Decision 1.1 (paid)",
        "kind": "openrouter-systemone",
        "paid": True,
        "notes": "System-One decision model ($0.042/1M in)",
    },
    {
        "id": "openrouter-chat:openai/gpt-5.4-nano",
        "label": "OpenRouter · OpenAI GPT-5.4 Nano Router (paid)",
        "kind": "openrouter-chat",
        "paid": True,
        "notes": "Ultra-fast nano chat model prompted for front/back/doctor",
    },
]


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
    "github",
    "public repo",
    "github repo",
    "repository",
    "repositories",
    "pull request",
    "create repo",
    "create a repo",
    "github issue",
    "git push",
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
_WORD_RE = re.compile(r"[a-z0-9_']+")
_NEG_WORDS = frozenset(
    {"not", "dont", "don't", "never", "no", "without", "instead", "skip", "stop"}
)


def extract_text_features(text: str) -> Counter[str]:
    """Extract word, bigram, negation-context, and char-ngram features in <0.15 ms."""
    low = " ".join(str(text or "").lower().split())
    counts: Counter[str] = Counter()
    if not low:
        counts["__empty__"] = 1
        return counts

    words = _WORD_RE.findall(low)
    if _MATH_RE.search(low):
        counts["__math__"] = 1
    if "?" in low:
        counts["__question__"] = 1
    if len(words) <= 6:
        counts["__short__"] = 1
    elif len(words) >= 14:
        counts["__long__"] = 1

    neg_ttl = 0
    prev = ""
    for w in words:
        counts[f"w:{w}"] += 1
        if prev:
            counts[f"b:{prev}_{w}"] += 1
        if neg_ttl > 0:
            counts[f"neg:{w}"] += 1
            neg_ttl -= 1
        if w in _NEG_WORDS or w.endswith("n't"):
            counts["__neg__"] += 1
            neg_ttl = 4
        # Character 3- and 4-grams inside padded word for morphology/typos
        if len(w) >= 3:
            padded = f"^{w}$"
            n = len(padded)
            for i in range(n - 2):
                counts[f"c3:{padded[i : i + 3]}"] += 1
            for i in range(n - 3):
                counts[f"c4:{padded[i : i + 4]}"] += 1
        prev = w
    return counts


class _OnnxIntentEngine:
    """Tiny local ONNX MLP intent classifier (<0.5 ms, offline)."""

    def __init__(self, onnx_dir: Path) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.model_path = self.onnx_dir / "intent.onnx"
        self.vocab_path = self.onnx_dir / "vocab.json"
        self.ready = False
        self._session: Any = None
        self._input_name = "features"
        self._index: dict[str, int] = {}
        self._idf = np.empty(0, dtype=np.float32)
        self._dim = 0
        self.load_error = ""
        self._load()

    def _load(self) -> None:
        if not (self.model_path.is_file() and self.vocab_path.is_file()):
            self.load_error = "intent.onnx or vocab.json missing"
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            self.load_error = f"onnxruntime missing: {exc}"
            return
        try:
            meta = json.loads(self.vocab_path.read_text(encoding="utf-8"))
            tokens = meta.get("tokens") or []
            idf = meta.get("idf") or []
            if not tokens or len(tokens) != len(idf):
                self.load_error = "invalid vocab.json"
                return
            self._index = {str(tok): idx for idx, tok in enumerate(tokens)}
            self._idf = np.asarray(idf, dtype=np.float32)
            self._dim = len(tokens)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            # Warm once so the very first user turn stays <1 ms
            zero = np.zeros((1, self._dim), dtype=np.float32)
            self._session.run(None, {self._input_name: zero})
            self.ready = True
        except Exception as exc:  # noqa: BLE001 - graceful fallback to Stage 1 rules
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False

    def vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        feats = extract_text_features(text)
        for feat, cnt in feats.items():
            idx = self._index.get(feat)
            if idx is not None:
                vec[idx] = (1.0 + math.log(cnt)) * float(self._idf[idx])
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec.reshape(1, self._dim)

    def predict(self, text: str) -> tuple[float, float, float]:
        vec = self.vectorize(text)
        out = self._session.run(None, {self._input_name: vec})[0][0]
        return float(out[0]), float(out[1]), float(out[2])


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


def _infer_mode_hint(text: str, route_hint: str = "") -> str:
    low = " ".join(str(text or "").lower().split())
    if route_hint == "doctor":
        return "diagnose"
    if any(k in low for k in BACK_VERIFY):
        return "verify"
    if any(k in low for k in BACK_RESEARCH):
        return "research"
    if any(k in low for k in BACK_SUMMARIZE):
        return "summarize"
    if any(k in low for k in BACK_BUILD):
        return "build"
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
    ):
        return "chat"
    if any(k in low for k in FRONT_DEVICE) or _MATH_RE.search(low):
        return "device"
    return "general" if route_hint == "back" else "chat"


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

    if back_total > 0 and front_hits > 0:
        sf = min(0.82, 0.55 + 0.10 * front_hits)
        sb = min(0.90, 0.62 + 0.10 * back_total)
        mode = (
            "build" if back_build_hits else ("research" if back_res_hits else "general")
        )
        return (sf, sb, 0.05, mode, "mixed_bias_check")

    words = len(low.split())
    if words <= 14:
        return (0.62, 0.25, 0.05 + 0.10 * doc_weak, "chat", "short_chat_front")
    return (0.42, 0.48, 0.05 + 0.10 * doc_weak, "general", "uncertain_borderline")


def _classify_openrouter_systemone(
    model_id: str,
    req: DecisionRequest,
    api_key: str,
    timeout_s: float = 2.5,
) -> tuple[float, float, float, str]:
    """Call OpenRouter's TypeSafe System-One classifier endpoint (/api/v1/systemone)."""
    import requests

    clean_id = model_id.removeprefix("openrouter/")
    payload = {
        "model": clean_id,
        "state": {
            "user_message": req.text,
            "session_state": req.session_state,
            "active_backs": req.active_backs,
            "recent_route": req.recent_route,
        },
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose which SONJ4 agent should handle the user message.",
                "criteria": {
                    "front": "Quick conversational turn, Q&A, memory recall, math, or single device action (<2s)",
                    "back": "Long build/code/script/file creation, web research, fact verification, drafting, or multi-step summary",
                    "doctor": "App diagnostics, audio/mic/TTS troubleshooting, health check, incident logs, failed tasks, or settings inspection",
                },
            }
        },
    }
    resp = requests.post(
        OPENROUTER_SYSTEMONE_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    ans = (data.get("answers") or {}).get("route") or {}
    probs = ans.get("probabilities") or {}
    sf = float(probs.get("front") or 0.0)
    sb = float(probs.get("back") or 0.0)
    sd = float(probs.get("doctor") or 0.0)
    if sf + sb + sd <= 0.0:
        choice = str(ans.get("choice") or "front").lower()
        conf = float(ans.get("confidence") or 0.75)
        sf = conf if choice == "front" else 0.1
        sb = conf if choice == "back" else 0.1
        sd = conf if choice == "doctor" else 0.05
    return sf, sb, sd, f"or_sys1:{clean_id}"


def _classify_openrouter_chat(
    model_id: str,
    req: DecisionRequest,
    api_key: str,
    timeout_s: float = 2.5,
) -> tuple[float, float, float, str]:
    """Use a fast OpenRouter chat model as a bounded 1-word route classifier."""
    import requests

    clean_id = model_id.removeprefix("openrouter-chat:")
    prompt = (
        "Classify this voice-assistant request into exactly one label: front, back, or doctor.\n"
        "- front: quick answer, small talk, math, memory recall, or single device tool (lights/timer/alarm/music/volume)\n"
        "- back: build code/file/game/webpage, web research, verify/audit claims, write/draft/summarize artifacts\n"
        "- doctor: app health, audio/mic/TTS problems, incident logs, failed tasks, settings/architecture\n"
        f"Request: {req.text}\nReply with ONLY one word: front, back, or doctor."
    )
    resp = requests.post(
        OPENROUTER_CHAT_URL,
        json={
            "model": clean_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (
        (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        .strip()
        .lower()
    )
    if "doctor" in text:
        return 0.10, 0.10, 0.85, f"or_chat:{clean_id}"
    if "back" in text:
        return 0.12, 0.85, 0.03, f"or_chat:{clean_id}"
    return 0.82, 0.12, 0.03, f"or_chat:{clean_id}"


def _classify_ollama(
    model_id: str, req: DecisionRequest, timeout_s: float = 1.5
) -> tuple[float, float, float, str]:
    """Classify via local Ollama if available."""
    import requests

    clean_id = model_id.removeprefix("ollama:")
    prompt = (
        "Reply with ONLY one word: front, back, or doctor.\n"
        "front=quick chat/device action, back=code/research/write/verify, doctor=app health/audio/logs/settings.\n"
        f"User: {req.text}\nLabel:"
    )
    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": clean_id, "prompt": prompt, "stream": False, "keep_alive": "5m"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    text = str(resp.json().get("response") or "").strip().lower()
    if "doctor" in text:
        return 0.10, 0.10, 0.85, f"ollama:{clean_id}"
    if "back" in text:
        return 0.12, 0.85, 0.03, f"ollama:{clean_id}"
    return 0.82, 0.12, 0.03, f"ollama:{clean_id}"


def list_classifier_options(
    onnx_dir: Path | None = None, *, openrouter_key: str = ""
) -> list[dict[str, Any]]:
    """Return all selectable classifier backends with live readiness flags."""
    onnx_ready = ((onnx_dir or ONNX_DIR) / "intent.onnx").is_file()
    if not openrouter_key:
        try:
            from llm import find_openrouter_key

            openrouter_key = find_openrouter_key()
        except Exception:  # noqa: BLE001
            openrouter_key = ""
    has_or = bool(openrouter_key)
    out: list[dict[str, Any]] = []
    for entry in CLASSIFIER_CATALOG:
        item = dict(entry)
        if item["id"] == "local-onnx":
            item["ready"] = onnx_ready
        elif item["id"] == "local-rules":
            item["ready"] = True
        else:
            item["ready"] = has_or
        out.append(item)
    return out


class DecisionClassifier:
    """Decision Classifier sitting between user and Front.

    Defaults to the tiny local ONNX model (`local-onnx`, <1 ms) + Stage-1 rules,
    while allowing selection of pure `local-rules`, OpenRouter System-One
    decision classifiers, OpenRouter fast chat routers, or local Ollama.
    """

    def __init__(
        self,
        deadline_ms: float = DEADLINE_MS,
        stage2_fn: Stage2Fn | None = None,
        onnx_dir: Path | None = None,
        backend: str = DEFAULT_CLASSIFIER_BACKEND,
        openrouter_key: str = "",
        cloud_deadline_ms: float = CLOUD_DEADLINE_MS,
    ) -> None:
        self.deadline_ms = max(10.0, float(deadline_ms))
        self.cloud_deadline_ms = max(250.0, float(cloud_deadline_ms))
        self.stage2_fn = stage2_fn
        self.onnx_dir = onnx_dir or ONNX_DIR
        self.openrouter_key = str(openrouter_key or "")
        self._onnx = _OnnxIntentEngine(self.onnx_dir)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sonj4-clf"
        )
        self._latencies: deque[float] = deque(maxlen=60)
        self._total_latency_ms: float = 0.0
        self.backend = str(backend or DEFAULT_CLASSIFIER_BACKEND).strip()
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
            "stage": self._active_stage_label(),
            "backend": self.backend,
        }

    def _active_stage_label(self) -> str:
        if self.backend == "local-onnx":
            return "local-onnx" if self._onnx.ready else "rules-v1"
        return self.backend

    def set_backend(self, backend: str, openrouter_key: str = "") -> None:
        self.backend = str(backend or DEFAULT_CLASSIFIER_BACKEND).strip()
        if openrouter_key:
            self.openrouter_key = str(openrouter_key)
        self.stats["backend"] = self.backend
        self.stats["stage"] = self._active_stage_label()

    def has_onnx(self) -> bool:
        return self._onnx.ready

    def _resolve_openrouter_key(self) -> str:
        if self.openrouter_key:
            return self.openrouter_key
        try:
            from llm import find_openrouter_key

            return find_openrouter_key()
        except Exception:  # noqa: BLE001
            return ""

    def _local_classify_scores(
        self,
        request: DecisionRequest,
        sf1: float,
        sb1: float,
        sd1: float,
        mode1: str,
        reason1: str,
    ) -> tuple[float, float, float, str, str]:
        if self.backend != "local-rules" and self._onnx.ready:
            pf, pb, pd = self._onnx.predict(request.text)
            top = "doctor" if pd >= max(pf, pb) else ("back" if pb > pf else "front")
            mode = _infer_mode_hint(request.text, top)
            if top == "front" and mode == "device":
                return pf, pb, pd, "device", "onnx_intent"
            return pf, pb, pd, mode, "onnx_intent"
        return sf1, sb1, sd1, mode1, reason1

    def classify(self, req: DecisionRequest | str) -> Decision:
        request = (
            req if isinstance(req, DecisionRequest) else DecisionRequest(text=str(req))
        )
        t0 = time.perf_counter()
        sf, sb, sd, mode, reason = _stage1_scores(request.text)

        # Explicit Stage-2 callable hook (used by deterministic deadline & custom tests)
        if self.stage2_fn is not None and reason in {
            "uncertain_borderline",
            "short_chat_front",
        }:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
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
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[classifier] stage2 failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        elif self.backend in {"local-onnx", "local-rules"}:
            sf, sb, sd, mode, reason = self._local_classify_scores(
                request, sf, sb, sd, mode, reason
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
        else:
            # Cloud OpenRouter / Ollama classifier with bounded timeout & local ONNX fallback
            fallback_used = False
            try:
                timeout_s = self.cloud_deadline_ms / 1000.0
                if self.backend.startswith("openrouter-chat:"):
                    key = self._resolve_openrouter_key()
                    if not key:
                        raise RuntimeError("missing openrouter key")
                    sf, sb, sd, reason = _classify_openrouter_chat(
                        self.backend, request, key, timeout_s=timeout_s
                    )
                elif self.backend.startswith("openrouter/"):
                    key = self._resolve_openrouter_key()
                    if not key:
                        raise RuntimeError("missing openrouter key")
                    sf, sb, sd, reason = _classify_openrouter_systemone(
                        self.backend, request, key, timeout_s=timeout_s
                    )
                elif self.backend.startswith("ollama:"):
                    sf, sb, sd, reason = _classify_ollama(
                        self.backend, request, timeout_s=timeout_s
                    )
                else:
                    raise RuntimeError(f"unknown classifier backend: {self.backend}")
                top = (
                    "doctor"
                    if sd >= max(pf := sf, pb := sb)
                    else ("back" if pb > pf else "front")
                )
                mode = _infer_mode_hint(request.text, top)
            except Exception as exc:  # noqa: BLE001 - cloud errors degrade to local ONNX
                fallback_used = True
                sf, sb, sd, mode, r_loc = self._local_classify_scores(
                    request, sf, sb, sd, mode, reason
                )
                reason = f"cloud_fallback_{r_loc}:{type(exc).__name__}"

            ms = (time.perf_counter() - t0) * 1000.0
            dec = apply_front_bias(
                sf,
                sb,
                sd,
                reason_code=reason,
                suggested_mode=mode,
                latency_ms=ms,
                fallback=fallback_used,
            )
            self._record(dec)
            return dec

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
