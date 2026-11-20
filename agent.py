"""The conversation loop: user text in, tool calls executed, reply text out."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm import LlmError, LlmReply, ToolCall
from toolbox import ToolResult

MAX_TOOL_ROUNDS = 8


class ToolBoxLike(Protocol):
    """What the conversation loop needs from a toolbox: schemas and call.

    A one-shot consult (server._ConsultBox) is deliberately NOT a ToolBox - it has
    no tools at all - and this protocol is what lets both satisfy the same type
    without handing a specialist the ability to act.
    """

    def schemas(self) -> list[dict[str, Any]]:
        """The tool schemas offered to the model this turn."""
        raise NotImplementedError

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run one tool call and return its result."""
        raise NotImplementedError


def log_stderr(message: str) -> None:
    """Diagnostics go to stderr, never stdout (the CLI parses stdout)."""
    import sys

    print(message, file=sys.stderr)


# Tools whose own result is already a speakable sentence. For these the second LLM round -
# 1.6-2.2 s in the cloud, ~200-700 ms locally - buys nothing, so the result is spoken
# directly. That removes an entire round trip from the critical path.
SPEAKABLE_TOOLS = frozenset(
    {
        "get_time",
        "get_weather",
        "calculate",
        "set_timer",
        "set_alarm",
        "set_lights",
        "adjust_lights",
        "adjust_temperature",
        "adjust_volume",
        "media_control",
        "set_reminder",
        "list_reminders",
        "make_call",
        "silence",
    }
)
SPEAKABLE_MAX_CHARS = 220
# Compaction bounds (#28): trim the running history when it exceeds KEEP entries.
COMPACT_KEEP = 24
COMPACT_TRIM_TO = 12

DEFAULT_SYSTEM_PROMPT = (
    "You are SONJ4 (Front), the owner's warm, natural voice companion and sole coordinator. "
    "ARCHITECTURE: 1 Front (you), 1 non-LLM Decision Classifier between the user and Front, "
    "1 Doctor (diagnostics & settings), and a pool of 5 always-warm Back workers (back-1..back-5) "
    "handling builds, web research, verification/audits, summaries, and automation. "
    "LATENCY BOUNDARY: FRONT handles ONLY <2s quick turns (small talk, 1-2 sentence facts, "
    "and single-shot device_tool actions). Anything taking >2s, multi-step work, web research, "
    "verification, summaries, or building/code/games/websites MUST be delegated immediately to "
    "role='back' (dispatched to an idle worker in back-1..back-5). Diagnostics/errors/logs/settings "
    "go to role='doctor'. Use classify_decision(task) only if genuinely unsure; never call an LLM router. "
    "NON-REFUSAL & DEFAULTS: Rarely deny a request. Never interrogate with clarifying questions; "
    "pick sensible defaults and delegate immediately. "
    "LANGUAGE: Always think, speak, and reply in English by default — even if an audio "
    "transcription looks noisy or resembles another language. Switch ONLY if explicitly asked. "
    "SPEECH STYLE: Spoken aloud — 1-2 short natural sentences, no markdown/lists/emoji/code. "
    "Before delegating, say 'Checking {short task name}', then act."
)


@dataclass
class ToolTrace:
    """What one tool call did, for display in the UI."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    ok: bool = True
    seconds: float = 0.0


def _spoken_result(trace: ToolTrace) -> str:
    """Turn a tool's own message into something worth saying aloud."""
    text = " ".join(trace.result.split())
    if not text:
        return "Done."
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _fallback_text(traces: list[ToolTrace]) -> str:
    """Last-resort reply when the model returns nothing after a tool call."""
    if not traces:
        return "I am not sure how to answer that."
    last = traces[-1]
    if last.ok:
        return last.result
    return f"I could not do that: {last.result}"


@dataclass
class AgentReply:
    """Final assistant turn plus the bookkeeping the UI and benchmark need."""

    text: str
    tools: list[ToolTrace] = field(default_factory=list)
    llm_seconds: float = 0.0
    rounds: int = 0
    model: str = ""
    error: bool = False
    # Set when a caller asked the turn to stop (a cancelled background task). The reply is
    # incomplete by definition, so callers must treat it as cancelled, not as an answer.
    cancelled: bool = False
    # Usage summed over the rounds of this turn, for the session cost meter.
    tokens: int = 0
    cost_usd: float = 0.0
    reasoning: list[str] = field(default_factory=list)


class Agent:
    """Holds the chat history and runs the tool-calling loop."""

    def __init__(
        self,
        llm,
        toolbox: ToolBoxLike,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_rounds: int = MAX_TOOL_ROUNDS,
        prefer_tool_result: bool = True,
    ) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.prefer_tool_result = prefer_tool_result
        self.history: list[dict[str, Any]] = []
        # Compaction v2 (#28/D-9): when history is trimmed, the dropped span is
        # summarised (asynchronously, by whatever summariser the caller wires in -
        # the local router model in practice) and the summary rides along as a
        # system note. Without a summariser this degrades to the v1 plain trim.
        self.summary: str = ""
        self.compact_summarizer: Callable[[str], str] | None = None
        # M4.3: scored memories relevant to THIS turn, injected by the caller
        # (server.Session) so recall happens without a tool round-trip.
        self.extra_context: str = ""

    def reset(self) -> None:
        """Forget the conversation (used by the UI's New chat button)."""
        self.history.clear()

    def _compact_history(self) -> None:
        """M4.2 (#28/D-9): compact the history, never through a tool exchange.

        The v1 trim cut at an arbitrary index, which could separate a tool call
        from its result. The kept window now starts at a user message, so every
        surviving tool exchange is intact. The dropped span is summarised in the
        background (if a summariser is wired) and the summary is prepended to
        the context on later rounds.
        """
        if len(self.history) <= COMPACT_KEEP:
            return
        target = len(self.history) - COMPACT_TRIM_TO
        while (
            target < len(self.history)
            and str(self.history[target].get("role")) != "user"
        ):
            target += 1
        dropped = self.history[:target]
        self.history = self.history[target:]
        if not dropped or self.compact_summarizer is None:
            return
        lines = [
            f"{m.get('role')}: {str(m.get('content'))[:300]}"
            for m in dropped
            if m.get("content")
        ]
        span = "\n".join(lines)[:4000]
        if not span:
            return

        def run() -> None:
            try:
                summary = self.compact_summarizer(span)  # type: ignore[misc]
            except Exception:  # noqa: BLE001 - compaction must never break a turn
                return
            if summary:
                self.summary = str(summary)[:1200]
                log_stderr(f"history compacted: {len(dropped)} entries -> summary")

        threading.Thread(target=run, daemon=True).start()

    def respond(
        self,
        user_text: str,
        on_tool: Callable[[ToolTrace], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
        raise_errors: bool = False,
        append_user: bool = True,
    ) -> AgentReply:
        """Send user_text, run any tools the model asks for, return the reply.

        should_stop is checked before every model round and before every tool call, so a
        cancelled background task stops at the next step instead of running to completion.
        An HTTP call already in flight cannot be interrupted; it is abandoned, not undone.

        M2.2: when stream=True and on_delta is given, every model round is streamed
        and on_delta receives content pieces as they arrive - that is what lets the
        caller speak sentence by sentence instead of waiting for the whole reply.
        The assembled reply is identical to the non-streamed one, tool calls included.

        raise_errors=True makes a provider failure propagate as LlmError instead of
        returning an error reply - the caller (the engine's fallback chain, a crew
        step, the role bench) needs the exception to walk its fallback models. The
        default stays False so a plain caller still gets a reply object it can show.
        append_user=False lets a retry run without duplicating the user message that
        the failed attempt already appended.
        """
        if append_user:
            self.history.append({"role": "user", "content": user_text})
        self._compact_history()
        system_content = self.system_prompt
        if self.extra_context:
            # M4.3: scored memories for this turn, so recall needs no tool round.
            system_content = system_content + "\n\n" + self.extra_context
            self.extra_context = ""
        if self.summary:
            system_content = (
                system_content + "\n\nEarlier in this conversation: " + self.summary
            )
        messages = [{"role": "system", "content": system_content}, *self.history]
        schemas = self.toolbox.schemas()

        traces: list[ToolTrace] = []
        reasonings: list[str] = []
        total_llm_s = 0.0
        total_tokens = 0
        total_cost = 0.0
        rounds = 0

        while rounds < self.max_rounds:
            if should_stop is not None and should_stop():
                self.history.append({"role": "assistant", "content": "(stopped)"})
                return AgentReply(
                    text="",
                    tools=traces,
                    llm_seconds=total_llm_s,
                    rounds=rounds,
                    cancelled=True,
                    tokens=total_tokens,
                    cost_usd=total_cost,
                )
            rounds += 1
            try:
                if stream and on_delta is not None and hasattr(self.llm, "chat_events"):
                    reply: LlmReply = self.llm.chat_events(messages, schemas, on_delta)
                else:
                    reply = self.llm.chat(messages, schemas)
            except LlmError as exc:
                if raise_errors:
                    # The caller owns recovery (fallback chains, crew retries).
                    raise
                text = f"I could not reach the language model: {exc}"
                self.history.append({"role": "assistant", "content": text})
                # error=True keeps this out of the speaker: an error read aloud is
                # picked up by the microphone and becomes an endless feedback loop.
                return AgentReply(
                    text=text,
                    tools=traces,
                    llm_seconds=total_llm_s,
                    rounds=rounds,
                    error=True,
                    tokens=total_tokens,
                    cost_usd=total_cost,
                )

            total_llm_s += reply.latency_s
            total_tokens += int(getattr(reply, "tokens", 0) or 0)
            total_cost += float(getattr(reply, "cost_usd", 0.0) or 0.0)
            if getattr(reply, "reasoning", ""):
                reasonings.append(str(reply.reasoning).strip())

            if not reply.tool_calls:
                # Small models occasionally finish with an empty message even
                # after a successful tool call. Speak the tool result rather
                # than losing the answer the user already earned.
                text = reply.content or _fallback_text(traces)
                self.history.append({"role": "assistant", "content": text})
                return AgentReply(
                    text=text,
                    tools=traces,
                    llm_seconds=total_llm_s,
                    rounds=rounds,
                    model=reply.model,
                    tokens=total_tokens,
                    cost_usd=total_cost,
                    reasoning=reasonings,
                )

            messages.append(self.llm.assistant_tool_message(reply))
            for call in reply.tool_calls:
                if should_stop is not None and should_stop():
                    self.history.append({"role": "assistant", "content": "(stopped)"})
                    return AgentReply(
                        text="",
                        tools=traces,
                        llm_seconds=total_llm_s,
                        rounds=rounds,
                        cancelled=True,
                        tokens=total_tokens,
                        cost_usd=total_cost,
                    )
                trace = self._run_tool(call)
                traces.append(trace)
                if on_tool is not None:
                    on_tool(trace)
                messages.append(self.llm.tool_result_message(call, trace.result))

            # Fast path: one definite device command whose result already reads like an
            # answer. Speaking it now removes the second LLM round from the turn.
            if self.prefer_tool_result and len(reply.tool_calls) == 1:
                last = traces[-1]
                if (
                    last.ok
                    and last.name in SPEAKABLE_TOOLS
                    and 0 < len(last.result) <= SPEAKABLE_MAX_CHARS
                ):
                    text = _spoken_result(last)
                    self.history.append({"role": "assistant", "content": text})
                    return AgentReply(
                        text=text,
                        tools=traces,
                        llm_seconds=total_llm_s,
                        rounds=rounds,
                        model=reply.model,
                        tokens=total_tokens,
                        cost_usd=total_cost,
                        reasoning=reasonings,
                    )

        # Final synthesis round without tools so work already gathered is not lost.
        if traces:
            try:
                synth_msgs = [
                    *messages,
                    {
                        "role": "user",
                        "content": "Summarise the answer concisely using only the tool results gathered above.",
                    },
                ]
                final_reply = (
                    self.llm.chat_events(synth_msgs, [], on_delta)
                    if (
                        stream
                        and on_delta is not None
                        and hasattr(self.llm, "chat_events")
                    )
                    else self.llm.chat(synth_msgs, [])
                )
                total_llm_s += final_reply.latency_s
                total_tokens += int(getattr(final_reply, "tokens", 0) or 0)
                total_cost += float(getattr(final_reply, "cost_usd", 0.0) or 0.0)
                if getattr(final_reply, "reasoning", ""):
                    reasonings.append(str(final_reply.reasoning).strip())
                if final_reply.content:
                    self.history.append(
                        {"role": "assistant", "content": final_reply.content}
                    )
                    return AgentReply(
                        text=final_reply.content,
                        tools=traces,
                        llm_seconds=total_llm_s,
                        rounds=rounds + 1,
                        model=final_reply.model,
                        tokens=total_tokens,
                        cost_usd=total_cost,
                        reasoning=reasonings,
                    )
            except LlmError:
                pass

        fallback = (
            _fallback_text(traces)
            if traces
            else (
                "That took too many steps, so I stopped. Try asking in a simpler way."
            )
        )
        self.history.append({"role": "assistant", "content": fallback})
        return AgentReply(
            text=fallback,
            tools=traces,
            llm_seconds=total_llm_s,
            rounds=rounds,
            tokens=total_tokens,
            cost_usd=total_cost,
            reasoning=reasonings,
        )

    def _run_tool(self, call: ToolCall) -> ToolTrace:
        started = time.perf_counter()
        result: ToolResult = self.toolbox.call(call.name, call.arguments)
        return ToolTrace(
            name=call.name,
            arguments=call.arguments,
            result=result.content,
            ok=result.ok,
            seconds=time.perf_counter() - started,
        )
