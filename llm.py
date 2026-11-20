"""LLM clients for the ME3 laptop voice app: local Ollama and OpenRouter.

Both clients return the same LlmReply shape so the agent loop never needs to
know which provider answered. Tool-call formatting differs slightly between
the two APIs, so each client owns its own tool-result message shape.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


def _log_stderr(message: str) -> None:
    """Diagnostics go to stderr, never stdout (the CLI parses stdout)."""
    print(message, file=sys.stderr)


OLLAMA_HOST = "http://127.0.0.1:11434"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_S = 180
# Streaming replies need absolute deadlines. requests' read timeout resets every time a
# byte arrives, so a provider that dribbles can hang far longer than the timeout: one
# benchmark run recorded a 557 s stall. These are wall-clock limits on the stream itself.
STREAM_CONNECT_TIMEOUT_S = 10
STREAM_READ_TIMEOUT_S = 45
FIRST_TOKEN_DEADLINE_S = 10.0
STREAM_TOTAL_DEADLINE_S = 45.0
# How long Ollama keeps a model in RAM after a request. "0s" unloads it as soon
# as the reply finishes (lowest memory, slowest next turn); "5m" keeps it warm
# for an active conversation. Exposed in the UI as "Model memory".
OLLAMA_KEEP_ALIVE = "5m"


class LlmError(RuntimeError):
    """Raised when a provider cannot be reached or returns an unusable reply."""


@dataclass
class ToolCall:
    """One function call requested by the model."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class StreamReply:
    """A streamed reply: what the latency benchmark needs to report TTFT."""

    text: str = ""
    first_token_s: float = 0.0
    total_s: float = 0.0
    model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LlmReply:
    """Normalised model output."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    model: str = ""
    # Usage as the provider reported it, so the session cost meter is real rather than guessed.
    tokens: int = 0
    cost_usd: float = 0.0
    reasoning: str = ""


EMBEDDING_HINTS = ("embed", "nomic-embed", "bge-", "e5-", "minilm", "gte-")


def unload_ollama_model(
    model: str, host: str = OLLAMA_HOST, timeout: float = 20.0
) -> bool:
    """Ask Ollama to evict a model from RAM right now.

    Setting keep_alive to 0 on a trivial request is the documented way to drop a
    model, and the runner process exits with it - which is what makes the app
    clean up after itself when it goes idle.
    """
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LlmError(f"could not unload {model}: {exc}") from exc
    return True


def loaded_ollama_models(
    host: str = OLLAMA_HOST, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Models currently resident in memory, per Ollama itself."""
    try:
        response = requests.get(f"{host}/api/ps", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LlmError(f"could not list loaded models: {exc}") from exc
    models = response.json().get("models")
    return list(models) if isinstance(models, list) else []


KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9\-_]{12,}")
KEY_FILE = Path(__file__).resolve().parent / "openrouter.key"
PI_MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"


def find_elevenlabs_key() -> str:
    """Locate an ElevenLabs key (env ELEVENLABS_API_KEY or sonj3/elevenlabs.key).

    Memory only — Settings.persistable() drops it, so it never reaches config.json
    or a log. Used for the optional cloud TTS stage and its credit analytics.
    """
    from_env = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if from_env:
        return from_env
    key_file = Path(__file__).resolve().parent / "elevenlabs.key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8-sig", errors="replace").strip()
    return ""


def find_openrouter_key() -> str:
    """Locate an OpenRouter key the machine already has.

    Order: the OPENROUTER_API_KEY environment variable, a local
    me3_laptop/openrouter.key file, then the Pi CLI's own config so the app can
    reuse the key that is already on this machine instead of duplicating it.
    The value is only ever held in memory - Settings.persistable() drops it, so
    it never reaches config.json or a log.
    """
    from_env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if from_env:
        return from_env

    if KEY_FILE.exists():
        # utf-8-sig: an editor or PowerShell may write a BOM, which would defeat
        # the fullmatch below and silently lose the key.
        candidate = KEY_FILE.read_text(encoding="utf-8-sig", errors="replace").strip()
        if KEY_PATTERN.fullmatch(candidate):
            return candidate

    if PI_MODELS_JSON.exists():
        text = PI_MODELS_JSON.read_text(encoding="utf-8", errors="replace")
        match = KEY_PATTERN.search(text)
        if match:
            return match.group(0)
    return ""


def list_openrouter_models(
    free_only: bool = True, timeout: float = 40.0, require_tools: bool = True
) -> list[str]:
    """Tool-capable OpenRouter model ids, free ones first when requested.

    The catalogue is public, so this needs no key. Free variants carry a ":free"
    suffix and cost nothing, which matters when the account has a small cap.
    """
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=timeout)
        response.raise_for_status()
        rows = response.json().get("data", [])
    except (requests.RequestException, ValueError) as exc:
        raise LlmError(f"could not list OpenRouter models: {exc}") from exc

    usable: list[str] = []
    for model in rows:
        params = model.get("supported_parameters") or []
        if require_tools and "tools" not in params:
            continue
        model_id = str(model.get("id", ""))
        if not model_id:
            continue
        if free_only:
            pricing = model.get("pricing") or {}
            try:
                free = (
                    float(pricing.get("prompt", "1")) == 0
                    and float(pricing.get("completion", "1")) == 0
                )
            except (TypeError, ValueError):
                free = False
            if not free:
                continue
        usable.append(model_id)
    # Nemotron variants are the ones verified here: fast and tool-reliable.
    usable.sort(key=lambda name: (0 if "nemotron" in name else 1, name))
    return usable


def model_capabilities(
    model: str, host: str = OLLAMA_HOST, timeout: float = 15.0
) -> list[str]:
    """What a model can do, according to Ollama itself.

    Embedding models report ["embedding"] and reject /api/chat with HTTP 400, so
    this is what keeps them out of the chat model dropdown.
    """
    try:
        response = requests.post(
            f"{host}/api/show", json={"model": model}, timeout=timeout
        )
        response.raise_for_status()
        capabilities = response.json().get("capabilities")
    except (requests.RequestException, ValueError) as exc:
        raise LlmError(f"could not inspect {model}: {exc}") from exc
    if not isinstance(capabilities, list):
        return []
    return [str(item) for item in capabilities]


def list_ollama_models(
    host: str = OLLAMA_HOST, timeout: float = 5.0, chat_only: bool = True
) -> list[str]:
    """Names of models on the local Ollama daemon that can hold a conversation."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LlmError(f"Ollama not reachable at {host}: {exc}") from exc

    names = sorted(
        str(model.get("name", "")) for model in response.json().get("models", [])
    )
    if not chat_only:
        return names

    usable: list[str] = []
    for name in names:
        try:
            capabilities = model_capabilities(name, host=host)
        except LlmError:
            capabilities = []
        if capabilities:
            if "completion" in capabilities or "tools" in capabilities:
                usable.append(name)
        elif not any(hint in name.lower() for hint in EMBEDDING_HINTS):
            usable.append(name)  # older Ollama: fall back to a name heuristic
    return usable


class OllamaClient:
    """Local inference through the Ollama HTTP API."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str = OLLAMA_HOST,
        temperature: float = 0.4,
        keep_alive: str = OLLAMA_KEEP_ALIVE,
    ) -> None:
        self.model = model
        self.keep_alive = keep_alive
        self.host = host.rstrip("/")
        self.temperature = temperature

    @property
    def label(self) -> str:
        return f"ollama/{self.model}"

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LlmReply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if tools:
            body["tools"] = tools

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.host}/api/chat", json=body, timeout=REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LlmError(f"Ollama chat failed: {exc}") from exc
        elapsed = time.perf_counter() - started

        payload = response.json()
        message = payload.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = __import__("json").loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            calls.append(
                ToolCall(
                    name=str(function.get("name", "")), arguments=dict(arguments or {})
                )
            )

        # Ollama reports token counts rather than a price; cost stays unknown.
        try:
            counts = response.json()
        except ValueError:
            counts = {}
        tokens = int(counts.get("prompt_eval_count", 0) or 0) + int(
            counts.get("eval_count", 0) or 0
        )
        return LlmReply(
            content=str(message.get("content") or "").strip(),
            tool_calls=calls,
            latency_s=elapsed,
            model=self.label,
            tokens=tokens,
        )

    def chat_stream(self, messages: list[dict[str, Any]]) -> StreamReply:
        """Stream a reply so first-token latency can be measured."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        started = time.perf_counter()
        first = 0.0
        chunks: list[str] = []
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=body,
                timeout=(STREAM_CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    now = time.perf_counter()
                    if not first and now - started > FIRST_TOKEN_DEADLINE_S:
                        raise LlmError(
                            f"no first token within {FIRST_TOKEN_DEADLINE_S:.0f}s - local model stalled"
                        )
                    if now - started > STREAM_TOTAL_DEADLINE_S:
                        raise LlmError(
                            f"stream exceeded {STREAM_TOTAL_DEADLINE_S:.0f}s - local model stalled"
                        )
                    if not line:
                        continue
                    try:
                        event = __import__("json").loads(line)
                    except ValueError:
                        continue
                    piece = str((event.get("message") or {}).get("content") or "")
                    if piece and not first:
                        first = time.perf_counter() - started
                    chunks.append(piece)
        except requests.RequestException as exc:
            raise LlmError(f"Ollama stream failed: {exc}") from exc
        return StreamReply(
            text="".join(chunks).strip(),
            first_token_s=first or (time.perf_counter() - started),
            total_s=time.perf_counter() - started,
            model=self.label,
        )

    def chat_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> LlmReply:
        """M2.2: stream a chat turn, assembling text AND tool calls.

        chat_stream only measures first-token latency for the benchmark. A real
        turn also needs tool calls, so this sibling streams content pieces to
        on_delta as they arrive (that is what makes sentence-chunked speech
        possible) and assembles any tool calls the model requests. The reply is
        a normal LlmReply, so the agent loop is unchanged downstream.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        if tools:
            body["tools"] = tools
        started = time.perf_counter()
        chunks: list[str] = []
        calls: list[ToolCall] = []
        tokens = 0
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=body,
                timeout=(STREAM_CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    now = time.perf_counter()
                    if (
                        not chunks
                        and not calls
                        and now - started > FIRST_TOKEN_DEADLINE_S
                    ):
                        raise LlmError(
                            f"no first token within {FIRST_TOKEN_DEADLINE_S:.0f}s - local model stalled"
                        )
                    if now - started > STREAM_TOTAL_DEADLINE_S:
                        raise LlmError(
                            f"stream exceeded {STREAM_TOTAL_DEADLINE_S:.0f}s - local model stalled"
                        )
                    if not line:
                        continue
                    try:
                        event = __import__("json").loads(line)
                    except ValueError:
                        continue
                    message = event.get("message") or {}
                    piece = str(message.get("content") or "")
                    if piece:
                        chunks.append(piece)
                        if on_delta is not None:
                            try:
                                on_delta(piece)
                            except Exception as exc:  # noqa: BLE001 - speech must not kill the stream
                                _log_stderr(
                                    f"on_delta raised ({exc}); continuing without speech"
                                )
                    for raw in message.get("tool_calls") or []:
                        function = raw.get("function") or {}
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = __import__("json").loads(arguments)
                            except ValueError:
                                arguments = {"raw": arguments}
                        calls.append(
                            ToolCall(
                                name=str(function.get("name", "")),
                                arguments=dict(arguments or {}),
                            )
                        )
                    if event.get("done"):
                        tokens = int(event.get("prompt_eval_count", 0) or 0) + int(
                            event.get("eval_count", 0) or 0
                        )
        except requests.RequestException as exc:
            raise LlmError(f"Ollama stream failed: {exc}") from exc
        return LlmReply(
            content="".join(chunks).strip(),
            tool_calls=calls,
            latency_s=time.perf_counter() - started,
            model=self.label,
            tokens=tokens,
        )

    def tool_result_message(self, call: ToolCall, result: str) -> dict[str, Any]:
        """Ollama expects the tool name, not an OpenAI-style call id."""
        return {"role": "tool", "tool_name": call.name, "content": result}

    def assistant_tool_message(self, reply: LlmReply) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in reply.tool_calls
            ],
        }


class OpenRouterClient:
    """Any OpenAI-compatible model through OpenRouter (needs an API key)."""

    provider = "openrouter"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.4,
        provider_order: str = "",
    ) -> None:
        if not api_key:
            raise LlmError("OpenRouter needs an API key")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        # OpenRouter routes one model to different upstream providers with very
        # different speeds (measured here: 1.4 s vs 24 s for the same prompt).
        # Preferring a fast provider keeps replies conversational; fallbacks stay
        # enabled so an upstream outage cannot break the app.
        self.provider_order = [
            p.strip() for p in provider_order.split(",") if p.strip()
        ]

    @property
    def label(self) -> str:
        return f"openrouter/{self.model}"

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LlmReply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        if self.provider_order:
            body["provider"] = {"order": self.provider_order, "allow_fallbacks": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ME3 voice agent (laptop dev)",
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                OPENROUTER_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
            )
            if not response.ok:
                detail = response.text.strip()[:220]
                raise LlmError(f"OpenRouter HTTP {response.status_code}: {detail}")
        except requests.RequestException as exc:
            raise LlmError(f"OpenRouter chat failed: {exc}") from exc
        elapsed = time.perf_counter() - started

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise LlmError(f"OpenRouter returned no choices: {str(payload)[:200]}")
        message = choices[0].get("message") or {}

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = __import__("json").loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            calls.append(
                ToolCall(
                    name=str(function.get("name", "")),
                    arguments=dict(arguments or {}),
                    id=str(raw.get("id", "")),
                )
            )

        # Usage is what makes the session cost meter real instead of invented.
        usage = payload.get("usage") or {}
        reasoning = str(
            message.get("reasoning") or message.get("reasoning_content") or ""
        ).strip()
        return LlmReply(
            content=str(message.get("content") or "").strip(),
            tool_calls=calls,
            latency_s=elapsed,
            model=str(payload.get("model") or self.label),
            tokens=int(usage.get("total_tokens") or 0),
            cost_usd=float(usage.get("cost") or 0.0),
            reasoning=reasoning,
        )

    def chat_stream(self, messages: list[dict[str, Any]]) -> StreamReply:
        """Stream a reply (SSE) and time the first content token."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        provider: dict[str, Any] = {"allow_fallbacks": True}
        if self.provider_order:
            provider["order"] = self.provider_order
        else:
            # OpenRouter's own fastest-provider sort. Without it a free model can land on a
            # provider that answers in minutes (measured: one 557 s stall in 9 runs).
            provider["sort"] = "throughput"
        body["provider"] = provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ME3 latency benchmark",
        }
        started = time.perf_counter()
        first = 0.0
        chunks: list[str] = []
        try:
            with requests.post(
                OPENROUTER_URL,
                json=body,
                headers=headers,
                timeout=(STREAM_CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw in response.iter_lines(decode_unicode=True):
                    now = time.perf_counter()
                    if not first and now - started > FIRST_TOKEN_DEADLINE_S:
                        raise LlmError(
                            f"no first token within {FIRST_TOKEN_DEADLINE_S:.0f}s - provider stalled"
                        )
                    if now - started > STREAM_TOTAL_DEADLINE_S:
                        raise LlmError(
                            f"stream exceeded {STREAM_TOTAL_DEADLINE_S:.0f}s - provider stalled"
                        )
                    if not raw:
                        continue
                    # iter_lines is typed as a buffer; be explicit about text.
                    line = (
                        raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                    )
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = __import__("json").loads(payload)
                    except ValueError:
                        continue
                    choices = event.get("choices") or [{}]
                    piece = str((choices[0].get("delta") or {}).get("content") or "")
                    if piece and not first:
                        first = time.perf_counter() - started
                    chunks.append(piece)
        except requests.RequestException as exc:
            raise LlmError(f"OpenRouter stream failed: {exc}") from exc
        return StreamReply(
            text="".join(chunks).strip(),
            first_token_s=first or (time.perf_counter() - started),
            total_s=time.perf_counter() - started,
            model=self.label,
        )

    def chat_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> LlmReply:
        """M2.2: stream a chat turn (SSE), assembling text AND tool calls.

        Tool calls arrive as indexed deltas (id/name on the first piece,
        arguments as JSON fragments), so they are accumulated per index and
        JSON-decoded once the stream ends.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            # Ask for usage in the final chunk so the cost meter stays real.
            "usage": {"include": True},
        }
        if tools:
            body["tools"] = tools
        provider: dict[str, Any] = {"allow_fallbacks": True}
        if self.provider_order:
            provider["order"] = self.provider_order
        else:
            provider["sort"] = "throughput"
        body["provider"] = provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ME3 voice agent (streamed turn)",
        }
        started = time.perf_counter()
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        partial: dict[int, dict[str, Any]] = {}
        saw_activity = False
        model = self.label
        tokens = 0
        cost = 0.0
        try:
            with requests.post(
                OPENROUTER_URL,
                json=body,
                headers=headers,
                timeout=(STREAM_CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
                stream=True,
            ) as response:
                if not response.ok:
                    detail = response.text.strip()[:220]
                    raise LlmError(
                        f"OpenRouter stream HTTP {response.status_code}: {detail}"
                    )
                for raw in response.iter_lines(decode_unicode=True):
                    now = time.perf_counter()
                    if not saw_activity and now - started > FIRST_TOKEN_DEADLINE_S:
                        raise LlmError(
                            f"no first token within {FIRST_TOKEN_DEADLINE_S:.0f}s - provider stalled"
                        )
                    if now - started > STREAM_TOTAL_DEADLINE_S:
                        raise LlmError(
                            f"stream exceeded {STREAM_TOTAL_DEADLINE_S:.0f}s - provider stalled"
                        )
                    if not raw:
                        continue
                    line = (
                        raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                    )
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = __import__("json").loads(payload)
                    except ValueError:
                        continue
                    if event.get("model"):
                        model = str(event["model"])
                    usage = event.get("usage") or {}
                    if usage:
                        tokens = int(usage.get("total_tokens") or tokens or 0)
                        cost = float(usage.get("cost") or cost or 0.0)
                    choices = event.get("choices") or [{}]
                    delta = choices[0].get("delta") or {}
                    r_piece = str(
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    if r_piece:
                        reasoning_chunks.append(r_piece)
                        saw_activity = True
                    if delta.get("tool_calls"):
                        saw_activity = True
                    piece = str(delta.get("content") or "")
                    if piece:
                        saw_activity = True
                    if piece:
                        chunks.append(piece)
                        if on_delta is not None:
                            try:
                                on_delta(piece)
                            except Exception as exc:  # noqa: BLE001 - speech must not kill the stream
                                _log_stderr(
                                    f"on_delta raised ({exc}); continuing without speech"
                                )
                    for raw_call in delta.get("tool_calls") or []:
                        index = int(raw_call.get("index") or 0)
                        slot = partial.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if raw_call.get("id"):
                            slot["id"] = str(raw_call["id"])
                        function = raw_call.get("function") or {}
                        if function.get("name"):
                            slot["name"] += str(function["name"])
                        if function.get("arguments"):
                            slot["arguments"] += str(function["arguments"])
        except requests.RequestException as exc:
            raise LlmError(f"OpenRouter stream failed: {exc}") from exc
        calls: list[ToolCall] = []
        for index in sorted(partial):
            slot = partial[index]
            arguments: Any = slot["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = (
                        __import__("json").loads(arguments) if arguments.strip() else {}
                    )
                except ValueError:
                    arguments = {"raw": arguments}
            calls.append(
                ToolCall(
                    name=str(slot["name"]),
                    arguments=dict(arguments or {}),
                    id=str(slot["id"]),
                )
            )
        return LlmReply(
            content="".join(chunks).strip(),
            tool_calls=[c for c in calls if c.name],
            latency_s=time.perf_counter() - started,
            model=model,
            tokens=tokens,
            cost_usd=cost,
            reasoning="".join(reasoning_chunks).strip(),
        )

    def tool_result_message(self, call: ToolCall, result: str) -> dict[str, Any]:
        """OpenAI-compatible APIs require the originating tool_call_id."""
        return {"role": "tool", "tool_call_id": call.id, "content": result}

    def assistant_tool_message(self, reply: LlmReply) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": __import__("json").dumps(call.arguments),
                    },
                }
                for index, call in enumerate(reply.tool_calls)
            ],
        }


def create_llm(
    provider: str,
    model: str,
    api_key: str = "",
    keep_alive: str = OLLAMA_KEEP_ALIVE,
    openrouter_order: str = "",
):
    """Factory used by the server so the UI can switch providers."""
    if provider == "ollama":
        return OllamaClient(model, keep_alive=keep_alive)
    if provider == "openrouter":
        return OpenRouterClient(model, api_key, provider_order=openrouter_order)
    raise LlmError(f"unknown provider '{provider}'")
