"""Pi RPC bridge: embed the installed pi harness (pi.dev) as a live agent.

Spawns `pi --mode rpc --no-session` and speaks the JSONL protocol: one JSON object
per line on stdin (commands) and stdout (responses and session events). One PiRpc
instance owns one long-lived child process; PiPane adds the UI-facing ring buffer
that classifies thinking, tool calls, tool output and text as they stream.

Why RPC instead of `pi --print`: print mode returns only the final text, so the
chain of thought and the tool activity are invisible. RPC streams
thinking_delta / toolcall_* / tool_execution_* / compaction_* / auto_retry_*, which
is what the panes need. Context overflow is handled by pi itself (auto-compaction
plus overflow recovery), so this module never truncates history.

Protocol references: the installed docs/rpc.md, rpc-commands.md, json.md and
rpc-extension-ui.md. Run the self-test with:  python pi_rpc.py --selftest
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PI_EXE_ENV = "PI_RPC_EXE"
START_TIMEOUT_S = 25.0
DIALOG_METHODS = frozenset({"select", "confirm", "input", "editor"})
KNOWN_PI_PROVIDERS = frozenset(
    {
        "github-copilot",
        "openrouter",
        "ollama",
        "aiand",
        "aiand-thinkmax",
        "anthropic",
        "openai",
        "google",
    }
)


def log_stderr(message: str) -> None:
    """Diagnostics go to stderr so a piping caller still reads clean stdout."""
    print(message, file=sys.stderr, flush=True)


def find_pi() -> str | None:
    """Locate the pi CLI. PI_RPC_EXE overrides PATH (used by the installer)."""
    override = os.environ.get(PI_EXE_ENV, "").strip()
    if override:
        return override
    for name in ("pi", "pi.cmd", "pi.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def split_provider_model(spec: str, default_provider: str = "") -> tuple[str, str]:
    """Split 'github-copilot/gpt-6-luna' -> ('github-copilot', 'gpt-6-luna').

    OpenRouter ids look like 'deepseek/deepseek-v4.1-flash' or
    'nvidia/nemotron-3.5-super-vl:free'. When the first segment is a known Pi
    provider, treat it as (provider, model_id); otherwise if it has '/' keep
    default_provider (e.g. 'openrouter') and the full slug as model_id.
    """
    raw = str(spec or "").strip()
    if not raw:
        return default_provider.strip(), ""
    if "/" in raw:
        head, tail = raw.split("/", 1)
        if head in KNOWN_PI_PROVIDERS:
            return head, tail
        return (default_provider.strip() or "openrouter"), raw
    return default_provider.strip(), raw


def pi_argv(
    exe: str,
    no_session: bool = True,
    model: str = "",
    provider: str = "",
    thinking: str = "",
    system_prompt: str = "",
    extra: list[str] | None = None,
) -> list[str]:
    """Build the child argv, wrapping the Windows .cmd shim the way CreateProcess needs.

    npm installs pi as a batch shim on Windows; CreateProcess cannot start a .cmd
    directly, so the command goes through cmd.exe /c. This mirrors the pattern the
    toolbox already uses for `pi --print`.
    """
    base = [exe, "--mode", "rpc"]
    if no_session:
        base.append("--no-session")
    prov, mod = split_provider_model(model, default_provider=provider)
    if mod:
        base += ["--model", mod]
        if thinking:
            # pi accepts the thinking level as a :suffix on --model, e.g. id:high
            base[-1] = f"{base[-1]}:{thinking}"
    if prov:
        base += ["--provider", prov]
    if thinking and not mod:
        base += ["--thinking", thinking]
    if system_prompt.strip():
        base += ["--append-system-prompt", system_prompt.strip()]
    if extra:
        base += [str(x) for x in extra]
    if os.name == "nt" and base[0].lower().endswith((".cmd", ".bat")):
        base = ["cmd.exe", "/c", *base]
    return base


@dataclass
class PiEvent:
    """One parsed session event from pi."""

    type: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def assistant_event(self) -> dict[str, Any]:
        """The nested assistant-message event, when this is a message_update."""
        ev = self.raw.get("assistantMessageEvent")
        return ev if isinstance(ev, dict) else {}

    @property
    def delta(self) -> str:
        """Text or thinking delta carried by a message_update, else ""."""
        ev = self.assistant_event
        if ev.get("type") in ("text_delta", "thinking_delta"):
            return str(ev.get("delta") or "")
        return ""

    @property
    def kind(self) -> str:
        """Coarse class used by the panes: think / text / tool / compact / retry / meta."""
        ev = self.assistant_event
        et = str(ev.get("type") or "")
        if self.type == "message_update":
            if et.startswith("thinking"):
                return "think"
            if et.startswith("text"):
                return "text"
            if et.startswith("toolcall"):
                return "tool"
            return "meta"
        if (
            self.type.startswith("tool_execution")
            or self.type == "bash_execution_update"
        ):
            return "tool"
        if self.type.startswith("compaction") or self.type.startswith("summarization"):
            return "compact"
        if self.type.startswith("auto_retry") or self.type.startswith("auto_retry_end"):
            return "retry"
        if self.type in ("agent_end", "agent_settled", "turn_end", "message_end"):
            return "meta"
        return "meta"


class PiRpcError(RuntimeError):
    """Raised when pi cannot start or a command fails."""


class PiRpc:
    """One long-lived pi RPC child process; thread-safe command round trips.

    Commands are written as one JSON object per line. Responses carry the command's
    `id`; session events do not. A reader thread routes both, so callers never
    touch the pipes directly.
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: str = "",
        provider: str = "",
        thinking: str = "",
        system_prompt: str = "",
        no_session: bool = True,
        on_event: Callable[[PiEvent], None] | None = None,
        on_dialog: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.cwd = cwd or str(Path.cwd())
        prov, mod = split_provider_model(model, default_provider=provider)
        self.model = mod
        self.provider = prov
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.no_session = no_session
        self.on_event = on_event
        self.on_dialog = on_dialog
        self._env = env
        self._extra_args = list(extra_args or [])
        self._proc: subprocess.Popen[str] | None = None
        self._write_lock = threading.Lock()
        self._waiters: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._settled = threading.Event()
        self._running = threading.Event()
        self._seq = 0
        self.stderr_tail: list[str] = []

    # ---------------------------------------------------------------- lifecycle
    def start(self, timeout_s: float = START_TIMEOUT_S) -> None:
        """Spawn pi and start the reader threads. Idempotent while running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        exe = find_pi()
        if not exe:
            raise PiRpcError("pi not found on PATH - install pi, or set " + PI_EXE_ENV)
        argv = pi_argv(
            exe,
            no_session=self.no_session,
            model=self.model,
            provider=self.provider,
            thinking=self.thinking,
            system_prompt=self.system_prompt,
            extra=self._extra_args or None,
        )
        env = dict(os.environ)
        if self._env:
            env.update(self._env)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            # Prevent child pi/cmd processes from attaching to the console or hanging on terminal close.
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            ) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                **kwargs,
            )
        except OSError as exc:
            raise PiRpcError(f"could not start pi ({argv[0]}): {exc}") from exc
        self._settled.clear()
        self._running.set()
        self._reader = threading.Thread(
            target=self._read_stdout, daemon=True, name="pi-rpc-stdout"
        )
        self._reader.start()
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True, name="pi-rpc-stderr"
        )
        self._stderr_thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Terminate the child process tree (cmd.exe -> node pi) cleanly."""
        proc, self._proc = self._proc, None
        self._running.clear()
        self._settled.set()
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        if os.name == "nt":
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        try:
            proc.terminate()
            proc.wait(timeout=min(2.0, timeout_s))
        except (OSError, subprocess.TimeoutExpired):
            with contextlib.suppress(OSError):
                proc.kill()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------- I/O
    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                log_stderr(f"[pi-rpc] non-JSON line: {line[:160]}")
                continue
            if not isinstance(rec, dict):
                continue
            self._dispatch(rec)
        self._running.clear()
        self._settled.set()

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.rstrip()
            if text:
                self.stderr_tail.append(text)
                del self.stderr_tail[:-40]

    def _dispatch(self, rec: dict[str, Any]) -> None:
        rtype = str(rec.get("type") or "")
        if rtype == "response":
            rid = str(rec.get("id") or "")
            waiter = self._waiters.pop(rid, None)
            self._responses[rid] = rec
            if waiter is not None:
                waiter.set()
            return
        if rtype == "extension_ui_request":
            self._handle_ui_request(rec)
            return
        event = PiEvent(type=rtype, raw=rec)
        if rtype == "agent_settled":
            self._settled.set()
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 - a pane must not kill the stream
                log_stderr(f"[pi-rpc] on_event raised: {exc}")

    def _handle_ui_request(self, rec: dict[str, Any]) -> None:
        """Answer extension dialogs so a blocked agent cannot wedge the pane."""
        method = str(rec.get("method") or "")
        if method not in DIALOG_METHODS:
            return  # notify / setStatus / setWidget / setTitle: display or ignore
        answer: dict[str, Any] | None = None
        if self.on_dialog is not None:
            try:
                answer = self.on_dialog(rec)
            except Exception as exc:  # noqa: BLE001
                log_stderr(f"[pi-rpc] on_dialog raised: {exc}")
        payload = {"type": "extension_ui_response", "id": rec.get("id")}
        if answer:
            payload.update(answer)
        else:
            payload["cancelled"] = True
        self._write(payload)

    def _write(self, record: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise PiRpcError("pi rpc process is not running")
        line = json.dumps(record, ensure_ascii=False)
        with self._write_lock:
            proc.stdin.write(line + chr(10))
            proc.stdin.flush()

    def command(
        self, payload: dict[str, Any], timeout_s: float = 60.0
    ) -> dict[str, Any]:
        """Send one command and wait for its matching response record."""
        if not self.alive:
            self.start()
        self._seq += 1
        rid = str(payload.get("id") or f"me3-{self._seq}")
        payload = {**payload, "id": rid}
        waiter = threading.Event()
        self._waiters[rid] = waiter
        self._write(payload)
        if not waiter.wait(timeout_s):
            self._waiters.pop(rid, None)
            tail = " | ".join(self.stderr_tail[-3:])
            suffix = f" ({tail})" if tail else ""
            raise PiRpcError(
                f"pi did not answer {payload.get('type')} in {timeout_s:.0f}s{suffix}"
            )
        rec = self._responses.pop(rid, {})
        if rec.get("success") is False:
            raise PiRpcError(f"pi refused {payload.get('type')}: {rec.get('error')}")
        return rec

    # ---------------------------------------------------------------- commands
    def prompt(self, message: str, timeout_s: float = 60.0) -> dict[str, Any]:
        """Send a prompt. The response means accepted, not finished: watch events."""
        self._settled.clear()
        return self.command({"type": "prompt", "message": message}, timeout_s)

    def abort(self, timeout_s: float = 15.0) -> dict[str, Any]:
        return self.command({"type": "abort"}, timeout_s)

    def set_model(
        self, model: str, provider: str = "", timeout_s: float = 30.0
    ) -> dict[str, Any]:
        prov, mod = split_provider_model(
            model, default_provider=provider or self.provider or "openrouter"
        )
        self.provider = prov
        self.model = mod
        return self.command(
            {"type": "set_model", "provider": prov, "modelId": mod},
            timeout_s,
        )

    def set_thinking_level(self, level: str, timeout_s: float = 30.0) -> dict[str, Any]:
        self.thinking = level
        return self.command({"type": "set_thinking_level", "level": level}, timeout_s)

    def get_available_models(self, timeout_s: float = 30.0) -> list[dict[str, Any]]:
        rec = self.command({"type": "get_available_models"}, timeout_s)
        data = rec.get("data") or {}
        models = data.get("models") or []
        return [m for m in models if isinstance(m, dict)]

    def set_auto_compaction(
        self, enabled: bool = True, timeout_s: float = 30.0
    ) -> dict[str, Any]:
        return self.command(
            {"type": "set_auto_compaction", "enabled": bool(enabled)}, timeout_s
        )

    def set_auto_retry(
        self, enabled: bool = True, timeout_s: float = 30.0
    ) -> dict[str, Any]:
        return self.command(
            {"type": "set_auto_retry", "enabled": bool(enabled)}, timeout_s
        )

    def compact(self, timeout_s: float = 30.0) -> dict[str, Any]:
        return self.command({"type": "compact"}, timeout_s)

    def bash(self, command: str, timeout_s: float = 60.0) -> dict[str, Any]:
        return self.command({"type": "bash", "command": command}, timeout_s)

    def get_state(self, timeout_s: float = 30.0) -> dict[str, Any]:
        return self.command({"type": "get_state"}, timeout_s)

    def wait_settled(self, timeout_s: float = 180.0) -> bool:
        """Block until agent_settled (or timeout). True when settled."""
        return self._settled.wait(timeout_s)

    def prompt_and_wait(self, message: str, timeout_s: float = 180.0) -> str:
        """One prompt round trip: send, collect text, wait for settled, return text."""
        text: list[str] = []
        final: list[str] = []
        outer = self.on_event

        def collect(ev: PiEvent) -> None:
            if ev.kind == "text" and ev.delta:
                text.append(ev.delta)
            if ev.type == "message_end":
                msg = ev.raw.get("message") or {}
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    body = _message_text(msg)
                    if body:
                        final.append(body)
            if outer is not None:
                outer(ev)

        self.on_event = collect
        try:
            self.prompt(message)
            if not self.wait_settled(timeout_s):
                raise PiRpcError(f"pi did not settle within {timeout_s:.0f}s")
        finally:
            self.on_event = outer
        return final[-1] if final else "".join(text)


def _message_text(message: dict[str, Any]) -> str:
    """Plain text of an assistant message, whatever shape its content has."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def _message_thinking(message: dict[str, Any]) -> str:
    """Plain thinking/reasoning text from a completed assistant message."""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            txt = str(block.get("thinking") or "").strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts)


# --------------------------------------------------------------------- pane
PANE_LINE_CAP = 400


@dataclass
class PaneLine:
    """One display line in a pane."""

    kind: str
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class PiTurnResult:
    """Structured output of one PiPane turn, including captured tools & thoughts."""

    text: str
    thinking: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class PiPane:
    """A named pane: one pi session plus the last PANE_LINE_CAP classified lines.

    The pane is the UI-facing half of the bridge: it turns pi's event stream into
    coalesced readable lines and keeps a bounded history, so a long run cannot
    flood the UI with single-token lines. Line kinds map to pane styles:
    think, tool, tool_result, text, compact, retry, meta.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.lines: list[PaneLine] = []
        self.status = "idle"
        self.rpc = PiRpc(on_event=self._on_event, **kwargs)
        self._lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._stream_buf: dict[str, str] = {"think": "", "text": ""}
        self._turn_capture: dict[str, Any] | None = None
        self._pending_tools: dict[str, dict[str, Any]] = {}
        self.on_delta: Callable[[str], None] | None = None
        self.on_think: Callable[[str], None] | None = None
        # Last turn's total tokens, for the UI context-length display.
        self.last_tokens = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.rpc.start()
        self.rpc.set_auto_compaction(True)
        self.rpc.set_auto_retry(True)
        # ask_rich marks the pane "thinking" before a cold start; do not
        # overwrite it with "ready" or the UI shows an idle pill mid-turn.
        if self.status != "thinking":
            self.status = "ready"

    def stop(self) -> None:
        self.rpc.stop()
        self.status = "stopped"

    def configure(
        self, model: str = "", provider: str = "", thinking: str = ""
    ) -> None:
        """Apply model/provider/thinking to a warm pane (restarting only if needed)."""
        prov, mod = split_provider_model(
            model, default_provider=provider or self.rpc.provider
        )
        if not self.rpc.alive:
            if mod:
                self.rpc.model = mod
            if prov:
                self.rpc.provider = prov
            if thinking:
                self.rpc.thinking = thinking
            return
        if mod and (mod != self.rpc.model or (prov and prov != self.rpc.provider)):
            try:
                self.rpc.set_model(mod, provider=prov)
            except PiRpcError as exc:
                self.note(f"model switch error: {exc}", "retry")
        if thinking and thinking != self.rpc.thinking:
            with contextlib.suppress(PiRpcError):
                self.rpc.set_thinking_level(thinking)

    def ask(
        self,
        message: str,
        timeout_s: float = 180.0,
        on_delta: Callable[[str], None] | None = None,
        on_think: Callable[[str], None] | None = None,
    ) -> str:
        """Send a prompt and stream the events into the pane until it settles."""
        res = self.ask_rich(
            message, timeout_s=timeout_s, on_delta=on_delta, on_think=on_think
        )
        return res.text

    def ask_rich(
        self,
        message: str,
        timeout_s: float = 180.0,
        on_delta: Callable[[str], None] | None = None,
        on_think: Callable[[str], None] | None = None,
    ) -> PiTurnResult:
        """Run one turn and return text + captured thinking + tool calls + usage."""
        with self._turn_lock:
            self.status = "thinking"
            self._stream_buf = {"think": "", "text": ""}
            self._pending_tools.clear()
            cap: dict[str, Any] = {
                "thinking": [],
                "tools": [],
                "model": f"{self.rpc.provider}/{self.rpc.model}".strip("/"),
                "tokens": 0,
                "cost_usd": 0.0,
                "error": "",
            }
            self._turn_capture = cap
            self.on_delta = on_delta
            self.on_think = on_think
            self._append("meta", f"prompt: {message[:200]}")
            started = time.perf_counter()
            try:
                if not self.rpc.alive:
                    self.start()
                answer = self.rpc.prompt_and_wait(message, timeout_s)
                self._flush_stream("think")
                self._flush_stream("text")
                if cap["error"] and not answer.strip():
                    raise PiRpcError(cap["error"])
                return PiTurnResult(
                    text=answer,
                    thinking=list(cap["thinking"]),
                    tools=list(cap["tools"]),
                    model=str(cap["model"]),
                    tokens=int(cap["tokens"]),
                    cost_usd=float(cap["cost_usd"]),
                    seconds=time.perf_counter() - started,
                )
            except PiRpcError:
                with contextlib.suppress(Exception):
                    self.rpc.abort(timeout_s=3.0)
                if not self.rpc.alive or not self.rpc._settled.is_set():
                    with contextlib.suppress(Exception):
                        self.stop()
                raise
            finally:
                self._flush_stream("think")
                self._flush_stream("text")
                self._turn_capture = None
                self.on_delta = None
                self.on_think = None
                self.status = "idle"

    # ---------------------------------------------------------------- events
    def _flush_stream(self, kind: str) -> None:
        buf = self._stream_buf.get(kind, "")
        if not buf:
            return
        self._stream_buf[kind] = ""
        text = buf.rstrip("\n")
        if not text:
            return
        self._append(kind, text)
        cap = self._turn_capture
        if kind == "think":
            if cap is not None:
                cap["thinking"].append(text)
            if self.on_think is not None:
                with contextlib.suppress(Exception):
                    self.on_think(text)

    def _buffer_delta(self, kind: str, delta: str) -> None:
        other = "text" if kind == "think" else "think"
        self._flush_stream(other)
        buf = self._stream_buf.get(kind, "") + delta
        if "\n" in buf:
            parts = buf.split("\n")
            for complete in parts[:-1]:
                if complete.strip():
                    self._append(kind, complete)
                    cap = self._turn_capture
                    if kind == "think":
                        if cap is not None:
                            cap["thinking"].append(complete)
                        if self.on_think is not None:
                            with contextlib.suppress(Exception):
                                self.on_think(complete)
            buf = parts[-1]
        elif len(buf) >= 180:
            self._append(kind, buf)
            cap = self._turn_capture
            if kind == "think":
                if cap is not None:
                    cap["thinking"].append(buf)
                if self.on_think is not None:
                    with contextlib.suppress(Exception):
                        self.on_think(buf)
            buf = ""
        self._stream_buf[kind] = buf
        if kind == "text" and delta and self.on_delta is not None:
            with contextlib.suppress(Exception):
                self.on_delta(delta)

    def _on_event(self, ev: PiEvent) -> None:
        kind = ev.kind
        if kind in ("think", "text") and ev.delta:
            self._buffer_delta(kind, ev.delta)
            return
        if ev.type == "message_update":
            et = str(ev.assistant_event.get("type") or "")
            if et == "thinking_end":
                self._flush_stream("think")
                content = str(ev.assistant_event.get("content") or "").strip()
                cap = self._turn_capture
                if content and cap is not None and not cap["thinking"]:
                    self._append("think", content[:1200])
                    cap["thinking"].append(content[:1200])
                    if self.on_think is not None:
                        with contextlib.suppress(Exception):
                            self.on_think(content[:1200])
            elif et == "text_end":
                self._flush_stream("text")
            elif et == "toolcall_end":
                self._flush_stream("think")
                self._flush_stream("text")
                call = ev.assistant_event.get("toolCall") or {}
                name = str(call.get("name") or call.get("toolName") or "tool")
                args_obj = call.get("arguments") or {}
                args = json.dumps(args_obj, ensure_ascii=False)[:220]
                self._append("tool", f"{name} {args}")
            return
        if ev.type == "tool_execution_start":
            self._flush_stream("think")
            self._flush_stream("text")
            call_id = str(ev.raw.get("toolCallId") or "")
            name = str(ev.raw.get("toolName") or "tool")
            args = ev.raw.get("args")
            if not isinstance(args, dict):
                args = {"raw": str(args or "")[:200]}
            self._pending_tools[call_id] = {
                "name": name,
                "arguments": args,
                "started": time.perf_counter(),
            }
            return
        if ev.type == "tool_execution_end":
            call_id = str(ev.raw.get("toolCallId") or "")
            name = str(ev.raw.get("toolName") or "tool")
            result = ev.raw.get("result") or {}
            text = _result_text(result)[:600]
            failed = bool(ev.raw.get("isError"))
            ms = ev.raw.get("durationMs")
            suffix = f" ({ms:.0f} ms)" if isinstance(ms, (int, float)) else ""
            self._append("tool_result", ("FAIL " if failed else "") + text + suffix)
            started_info = self._pending_tools.pop(call_id, {})
            if self._turn_capture is not None:
                secs = (
                    float(ms) / 1000.0
                    if isinstance(ms, (int, float))
                    else max(
                        0.0,
                        time.perf_counter()
                        - float(started_info.get("started") or time.perf_counter()),
                    )
                )
                self._turn_capture["tools"].append(
                    {
                        "name": str(started_info.get("name") or name),
                        "arguments": started_info.get("arguments") or {},
                        "result": text,
                        "ok": not failed,
                        "seconds": round(secs, 3),
                    }
                )
            return
        if ev.type == "message_end":
            self._flush_stream("think")
            self._flush_stream("text")
            msg = ev.raw.get("message") or {}
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                cap = self._turn_capture
                if cap is not None:
                    prov = str(msg.get("provider") or self.rpc.provider or "")
                    mod = str(msg.get("model") or self.rpc.model or "")
                    if mod:
                        cap["model"] = f"{prov}/{mod}" if prov else mod
                    usage = msg.get("usage") or {}
                    if isinstance(usage, dict):
                        cap["tokens"] = int(
                            usage.get("totalTokens") or cap["tokens"] or 0
                        )
                        cost = usage.get("cost")
                        if isinstance(cost, dict):
                            cap["cost_usd"] = float(
                                cost.get("total") or cap["cost_usd"] or 0.0
                            )
                        elif isinstance(cost, (int, float)):
                            cap["cost_usd"] = float(cost)
                    err_msg = str(msg.get("errorMessage") or "").strip()
                    if err_msg:
                        cap["error"] = err_msg
                        self._append("retry", f"error: {err_msg[:220]}")
                    # Capture thinking if streamed only on message_end.
                    if not cap["thinking"]:
                        full_think = _message_thinking(msg)
                        if full_think:
                            cap["thinking"].append(full_think[:1200])
                            self._append("think", full_think[:1200])
                            if self.on_think is not None:
                                with contextlib.suppress(Exception):
                                    self.on_think(full_think[:1200])
            return
        if ev.type == "compaction_start":
            self._append("compact", f"compacting ({ev.raw.get('reason')})")
            return
        if ev.type == "compaction_end":
            result = ev.raw.get("result") or {}
            before = result.get("tokensBefore")
            after = result.get("estimatedTokensAfter")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                self._append("compact", f"compacted {before:.0f} -> {after:.0f} tokens")
            else:
                self._append("compact", "compacted")
            return
        if ev.type == "auto_retry_start":
            self._append(
                "retry", f"retry {ev.raw.get('attempt')}/{ev.raw.get('maxAttempts')}"
            )
            return
        if ev.type == "agent_settled":
            self._flush_stream("think")
            self._flush_stream("text")
            self.status = "idle"
            return

    # ------------------------------------------------------------------ buffer
    def note(self, text: str, kind: str = "meta") -> None:
        """Append a plain note (an error, an abort) without touching the rpc process."""
        self._append(kind, text)

    def _append(self, kind: str, text: str) -> None:
        with self._lock:
            self.lines.append(PaneLine(kind=kind, text=text))
            if len(self.lines) > PANE_LINE_CAP:
                del self.lines[: len(self.lines) - PANE_LINE_CAP]

    def snapshot(self, limit: int = PANE_LINE_CAP) -> list[dict[str, Any]]:
        """The tail of the pane for the UI: [{kind, text, ts}, ...]."""
        with self._lock:
            rows = self.lines[-limit:]
        return [{"kind": r.kind, "text": r.text, "ts": r.ts} for r in rows]


def _result_text(result: Any) -> str:
    """Plain text of a tool result, whatever shape it has."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or ""))
            return "".join(parts)
        return json.dumps(result, ensure_ascii=False)[:600]
    return str(result)


def list_pi_models(timeout_s: float = 25.0) -> list[dict[str, Any]]:
    """Query Pi's available model list once over RPC."""
    exe = find_pi()
    if not exe:
        return []
    rpc = PiRpc(extra_args=["--no-extensions", "--no-tools"])
    try:
        rpc.start()
        return rpc.get_available_models(timeout_s=timeout_s)
    except PiRpcError:
        return []
    finally:
        rpc.stop()


# ------------------------------------------------------------------ self-test
def _selftest() -> int:
    """Start pi over RPC, ask one short question, print what streamed."""
    print("pi:", find_pi() or "NOT FOUND")
    if not find_pi():
        return 2
    seen: list[str] = []
    pane = PiPane("selftest", cwd=str(Path(__file__).resolve().parent))

    def record(ev: PiEvent) -> None:
        seen.append(ev.kind)
        pane._on_event(ev)

    pane.rpc.on_event = record
    try:
        pane.start()
    except PiRpcError as exc:
        print("start failed:", exc)
        return 2
    try:
        answer = pane.ask("Reply with exactly: bridge works", timeout_s=120.0)
    except PiRpcError as exc:
        print("ask failed:", exc)
        print("stderr tail:", chr(10).join(pane.rpc.stderr_tail[-5:]))
        return 2
    finally:
        pane.stop()
    kinds = sorted(set(seen))
    print("answer:", answer[:200])
    print("event kinds seen:", kinds)
    print("pane lines:", len(pane.snapshot()))
    ok = bool(answer.strip()) and ("think" in kinds or "text" in kinds)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pi RPC bridge")
    parser.add_argument("--selftest", action="store_true", help="ask pi one question")
    parser.add_argument("--ask", default="", help="one prompt, print the answer")
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.ask:
        rpc = PiRpc(model=args.model)
        try:
            rpc.start()
            print(rpc.prompt_and_wait(args.ask))
        finally:
            rpc.stop()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
