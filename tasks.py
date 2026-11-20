"""Background task agents: the back layer of the architecture.

A task is the work the front agent should not do itself - reading documents, multi-step research,
anything that takes longer than a conversational pause. Each task runs on its own thread with its
own model (DeepSeek v4.1 Flash by default), emits events that drive the live workflow canvas, can
be cancelled, and reports back when it is done.

Design rules:

- The front agent stays responsive: starting a task never blocks a turn.
- Events are append-only and replayable, so the canvas survives a restart.
- Cancellation is cooperative: the flag is checked between steps and before each tool call.
- Concurrency is capped; the rest queue, so a demo cannot saturate the machine.
- Models are injected, so the whole layer is testable without a network.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm import LlmError

QUEUED = "queued"
RUNNING = "running"
WAITING = "waiting"
BLOCKED = "blocked"
DONE = "done"
CANCELLED = "cancelled"
FINISHED = {DONE, CANCELLED}

MAX_BRIEF_CHARS = 2000
REPORT_CHARS = 1200


def log(message: str) -> None:
    """Diagnostics go to stderr, never stdout."""
    print(message, file=sys.stderr)


AGENTS_PATH = Path(__file__).resolve().parent / "agents.json"


def _role_models(role: str) -> list[str]:
    """M5.1: a role's fallback chain from agents.json (routing is data).

    Returns [fallback1, fallback2, ..., primary] so a fully failed chain still
    tries the configured model once. Any read problem returns [] - the caller
    then behaves exactly as before this feature existed.
    """
    try:
        raw = json.loads(AGENTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entry = raw.get(role) if isinstance(raw, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    chain = [str(m) for m in (entry.get("fallbacks") or []) if str(m).strip()]
    primary = str(entry.get("model") or "").strip()
    if primary:
        chain.append(primary)
    return chain


@dataclass
class TaskEvent:
    """One thing that happened inside a task. The canvas is a projection of these."""

    ts: float
    task: str
    kind: str  # queued | start | step | artifact | done | error | cancelled
    state: str = RUNNING
    parent: str = ""
    agent: str = "back"
    model: str = ""
    step: int = 0
    of: int = 0
    label: str = ""
    artifact: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    """A unit of background work."""

    id: str
    brief: str
    state: str = QUEUED
    agent: str = "back"
    model: str = ""
    parent: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    steps: list[TaskEvent] = field(default_factory=list)
    result: str = ""
    error: str = ""
    artifacts: list[str] = field(default_factory=list)
    tokens: int = 0
    cost_usd: float = 0.0
    cancel_requested: bool = False
    # W2.1: crew bookkeeping (empty for plain tasks).
    steps_plan: list[dict[str, Any]] = field(default_factory=list)
    crew_process: str = ""

    @property
    def seconds(self) -> float:
        end = self.finished or time.time()
        return max(0.0, end - (self.started or self.created))

    def summary(self) -> dict[str, Any]:
        """What the UI needs per row, without the event list."""
        return {
            "id": self.id,
            "brief": self.brief,
            "state": self.state,
            "model": self.model,
            "steps": len(self.steps),
            "seconds": round(self.seconds, 1),
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "result": self.result[:400],
            "error": self.error[:200],
            "artifacts": list(self.artifacts),
        }


class TaskManager:
    """Runs task agents, owns their events, and reports results back."""

    def __init__(
        self,
        session_dir: Path | None = None,
        agents: dict[str, Any] | None = None,
        on_event: Callable[[TaskEvent], None] | None = None,
        on_report: Callable[[Task, str], None] | None = None,
        llm_factory: Callable[..., Any] | None = None,
        toolbox_factory: Callable[[], Any] | None = None,
        deliver: Callable[[str, Task], None] | None = None,
        role_runner: Callable[[str, str, str], tuple[str, int, float]] | None = None,
    ) -> None:
        settings = (
            ((agents or {}).get("back-1") or (agents or {}).get("back") or {})
            if agents
            else {}
        )
        self.model = str(settings.get("model") or "github-copilot/gpt-6-luna")
        self.fallbacks = [str(m) for m in settings.get("fallbacks", []) or []]
        self.max_steps = int(settings.get("max_steps") or 12)
        self.max_parallel = max(1, int(settings.get("max_parallel") or 5))
        self.session_dir = Path(session_dir) if session_dir else None
        self.on_event = on_event
        self.on_report = on_report
        self._llm_factory = llm_factory
        self._toolbox_factory = toolbox_factory
        self._deliver = deliver
        self._role_runner = role_runner
        self._tasks: dict[str, Task] = {}
        self._queue: list[str] = []
        self._running: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        if self.session_dir:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "artifacts").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ public
    def start(self, brief: str, model: str = "", parent: str = "") -> Task:
        """Accept a task and return immediately; the work happens on its own thread."""
        clean = " ".join(str(brief).split())[:MAX_BRIEF_CHARS]
        if not clean:
            raise ValueError("a task needs a brief")
        task = Task(
            id="t-" + uuid.uuid4().hex[:6],
            brief=clean,
            model=model or self.model,
            parent=parent,
        )
        with self._lock:
            self._tasks[task.id] = task
            self._queue.append(task.id)
        self._emit(task, "queued", state=QUEUED, label=clean[:80])
        self._pump()
        return task

    # ------------------------------------------------------------------ crews (W2.1)
    def start_crew(
        self,
        task_brief: str,
        roles: list[str] | None = None,
        process: str = "sequential",
        agents: dict[str, Any] | None = None,
    ) -> Task:
        """W2.1: a CrewAI-pattern crew as real background work.

        The crew is one Task whose steps are real Agent.respond turns, each with
        its own role's model from agents.json (routing is data, all :free-capable).

        process="sequential": role[i] works on step[i]; each step's output feeds
        the next step's prompt, exactly like CrewAI's sequential process.
        process="hierarchical": the first role is the manager - it writes a plan,
        each later role executes one plan step, and the manager writes the final
        review. Token budget is bounded by max_steps.
        """
        roster = agents or {}
        wanted = [r for r in (roles or []) if str(r) in roster] or (
            ["back-1"] if "back-1" in roster else ["back"]
        )
        clean = " ".join(str(task_brief).split())[:MAX_BRIEF_CHARS]
        if not clean:
            raise ValueError("a crew needs a brief")
        task = Task(
            id="crew-" + uuid.uuid4().hex[:6],
            brief=f"[crew:{process}] " + clean,
            model=str((roster.get(wanted[0]) or {}).get("model") or self.model),
            parent="",
        )
        task.steps_plan = [  # the plan the crew will execute, per role
            {
                "role": role,
                "model": str((roster.get(role) or {}).get("model") or self.model),
            }
            for role in wanted
        ]
        task.crew_process = process
        with self._lock:
            self._tasks[task.id] = task
            self._queue.append(task.id)
        self._emit(
            task,
            "queued",
            state=QUEUED,
            label=f"crew of {len(wanted)}: {', '.join(wanted)}",
        )
        self._pump()
        return task

    def _crew_worker(self, task: Task) -> None:
        """Run the crew plan step by step, each step on its role's model."""
        try:
            task.started = time.time()
            self._emit(task, "start", label=f"crew process: {task.crew_process}")
            context = ""
            final = ""
            total_tokens = 0
            total_cost = 0.0
            plan = list(getattr(task, "steps_plan", []))
            if task.crew_process == "hierarchical" and len(plan) >= 3:
                # Manager plans first: role[0] writes numbered steps for the rest.
                manager = plan[0]
                plan_prompt = (
                    "You are the crew manager. Break this task into at most "
                    f"{len(plan) - 1} numbered steps for these workers: "
                    + ", ".join(p["role"] for p in plan[1:])
                    + ". Task: "
                    + task.brief
                )
                plan_text, p_tokens, p_cost = self._crew_step(
                    task, manager, plan_prompt, 0
                )
                total_tokens += p_tokens
                total_cost += p_cost
                context = plan_text
            for index, step in enumerate(plan, start=1):
                if task.cancel_requested:
                    task.state = CANCELLED
                    task.finished = time.time()
                    self._emit(
                        task, "cancelled", state=CANCELLED, label="cancelled mid-crew"
                    )
                    if self._deliver is not None:
                        try:
                            self._deliver("task cancelled", task)
                        except Exception as exc:  # noqa: BLE001
                            log(
                                f"[tasks] deliver callback failed: {type(exc).__name__}: {exc}"
                            )
                    return
                role = step["role"]
                carry = (
                    f"\n\nPrevious work to build on:\n{context[:1200]}"
                    if context
                    else ""
                )
                step_prompt = (
                    f"You are the crew's {role}. Work on this task: {task.brief}"
                    + carry
                    + "\nFinish with your concrete contribution."
                )
                text, tokens, cost = self._crew_step(task, step, step_prompt, index)
                total_tokens += tokens
                total_cost += cost
                context = text
                final = text
            with self._lock:
                task.result = str(final)[:4000]
                task.tokens = total_tokens
                task.cost_usd = round(total_cost, 6)
                task.state = DONE
                task.finished = time.time()
            artifact = self._save_artifact(task)
            if artifact:
                task.artifacts.append(artifact)
            self._emit(
                task,
                "done",
                state=DONE,
                label="crew finished",
                artifact=artifact,
                tokens=total_tokens,
                cost_usd=total_cost,
            )
            if self._deliver is not None:
                try:
                    self._deliver(task.result, task)
                except Exception as exc:  # noqa: BLE001
                    log(f"[tasks] deliver callback failed: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            task.state = BLOCKED
            task.error = f"{type(exc).__name__}: {exc}"[:400]
            task.finished = time.time()
            self._emit(task, "error", state=BLOCKED, label=task.error)
            if self._deliver is not None:
                try:
                    self._deliver(f"task failed: {task.error}", task)
                except Exception as exc:  # noqa: BLE001
                    log(f"[tasks] deliver callback failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._running.pop(task.id, None)
            self._pump()

    def _crew_step(
        self, task: Task, step: dict, prompt: str, index: int
    ) -> tuple[str, int, float]:
        """One crew step: a real agent turn on that role's model, streamed into the graph."""
        from agent import Agent

        role = str(step.get("role", "back"))
        model = str(step.get("model", "")) or self.model
        if self._role_runner is not None:
            self._emit(
                task,
                "step",
                label=f"{role} starts via pi.dev ({model[:40]})",
                seconds=0.0,
                state=RUNNING,
            )
            text, tokens, cost = self._role_runner(role, prompt, model)
            text = str(text or "").strip()[:4000]
            self._emit(
                task,
                "step",
                label=f"{role} done ({len(text)} chars)",
                seconds=0.0,
                state=DONE,
            )
            return text, tokens, cost
        toolbox = self._toolbox()

        def on_tool(trace: Any) -> None:
            self._emit(
                task,
                "step",
                label=f"{role}: {trace.name} ({'ok' if trace.ok else 'failed'})",
                seconds=float(getattr(trace, "seconds", 0.0)),
                state=RUNNING,
            )

        # M5.1: a dead model must not kill the crew. Each step walks its role's
        # fallback chain (agents.json) and only gives up when all of it fails.
        attempts = [model] + [m for m in _role_models(role) if m != model]
        reply = None
        last_error: Exception | None = None
        for attempt_model in attempts:
            self._emit(
                task,
                "step",
                label=f"{role} starts ({attempt_model[:40]})",
                seconds=0.0,
                state=RUNNING,
            )
            agent = Agent(
                self._client(attempt_model),
                toolbox,
                max_rounds=self.max_steps,
                prefer_tool_result=True,
            )
            try:
                # raise_errors=True is what makes this loop real: a provider
                # failure must reach the except below to walk the chain.
                reply = agent.respond(
                    prompt,
                    on_tool=on_tool,
                    should_stop=lambda: task.cancel_requested,
                    raise_errors=True,
                )
                break
            except LlmError as exc:
                last_error = exc
                self._emit(
                    task,
                    "step",
                    label=f"{role}: {attempt_model[:30]} failed ({str(exc)[:60]})",
                    seconds=0.0,
                    state=RUNNING,
                )
        if reply is None:
            raise last_error or LlmError(f"every model failed for role {role}")
        text = str(reply.text or "").strip()[:4000]
        tokens = int(getattr(reply, "tokens", 0) or 0)
        cost = float(getattr(reply, "cost_usd", 0.0) or 0.0)
        self._emit(
            task,
            "step",
            label=f"{role} done ({len(text)} chars)",
            seconds=0.0,
            state=DONE,
        )
        return text, tokens, cost

    def cancel(self, task_id: str) -> bool:
        """Ask a task to stop; a queued task stops immediately."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state in FINISHED:
                return False
            task.cancel_requested = True
            if task.state == QUEUED:
                task.state = CANCELLED
                task.finished = time.time()
                if task_id in self._queue:
                    self._queue.remove(task_id)
        if task.state == CANCELLED:
            self._emit(
                task, "cancelled", state=CANCELLED, label="cancelled before it started"
            )
            return True
        return True

    def cancel_all(self) -> int:
        with self._lock:
            ids = [t.id for t in self._tasks.values() if t.state not in FINISHED]
        return sum(1 for task_id in ids if self.cancel(task_id))

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created)

    def summaries(self) -> list[dict[str, Any]]:
        return [task.summary() for task in self.list()]

    def graph(self) -> dict[str, Any]:
        """Nodes and edges for the live workflow canvas."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for task in self.list():
            nodes.append(
                {
                    "id": task.id,
                    "label": task.brief[:60],
                    "state": task.state,
                    "kind": "task",
                    "model": task.model,
                    "seconds": round(task.seconds, 1),
                }
            )
            if task.parent:
                edges.append({"from": task.parent, "to": task.id})
            for index, event in enumerate(task.steps, start=1):
                child = f"{task.id}.{index}"
                nodes.append(
                    {
                        "id": child,
                        "label": event.label[:50],
                        "state": event.state,
                        "kind": "step",
                        "model": event.model,
                        "seconds": event.seconds,
                    }
                )
                edges.append({"from": task.id, "to": child})
        return {"nodes": nodes, "edges": edges}

    def events(self) -> list[dict[str, Any]]:
        return [event.as_dict() for task in self.list() for event in task.steps]

    # ------------------------------------------------------------------ internal
    def _emit(self, task: Task, kind: str, **fields: Any) -> None:
        event = TaskEvent(
            ts=time.time(),
            task=task.id,
            kind=kind,
            model=task.model,
            parent=task.parent,
            **fields,
        )
        with self._lock:
            task.steps.append(event)
            path = self.session_dir / "events.jsonl" if self.session_dir else None
        if path is not None:
            with (
                contextlib.suppress(OSError),
                path.open("a", encoding="utf-8") as handle,
            ):
                handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 - a UI callback must never kill a task
                log(f"[tasks] event listener failed: {type(exc).__name__}: {exc}")

    def _pump(self) -> None:
        """Start as many queued tasks as the concurrency cap allows."""
        while True:
            with self._lock:
                if not self._queue or len(self._running) >= self.max_parallel:
                    return
                task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if task is None or task.state == CANCELLED:
                    continue
                task.state = RUNNING
                task.started = time.time()
                # Crew/delegated tasks (steps_plan) run through the role's live
                # Pi pane via _crew_worker; plain tasks use the LLM-client worker.
                worker_fn = (
                    self._crew_worker
                    if getattr(task, "steps_plan", None)
                    else self._worker
                )
                thread = threading.Thread(target=worker_fn, args=(task,), daemon=True)
                self._running[task_id] = thread
            thread.start()

    def _client(self, model: str):
        """Cloud client first, then each fallback; injection makes tests offline."""
        if self._llm_factory is not None:
            return self._llm_factory(model)
        from llm import create_llm, find_openrouter_key

        return create_llm("openrouter", model, find_openrouter_key())

    def _spawn_subtask(self, brief: str, model: str = "") -> str:
        """Let a crew step hand further work to the task layer (CrewAI delegation)."""
        sub = self.start(brief, model=model, parent="")
        return f"{sub.id} queued: {brief[:60]}"

    def _toolbox(self):
        if self._toolbox_factory is not None:
            return self._toolbox_factory()

        # A worker's tools must behave like the front desk's: timers and further task
        # handoffs work inside a delegated job, not just in the first turn.
        from toolbox import ToolBox as _TB

        box = _TB(on_task=self._spawn_subtask)
        return box

    def _worker(self, task: Task) -> None:
        try:
            # Imported late so tests can inject a fake client and toolbox.
            from agent import Agent

            client = self._client(task.model)
            toolbox = self._toolbox()
            agent = Agent(
                client, toolbox, max_rounds=self.max_steps, prefer_tool_result=True
            )
            self._emit(task, "start", label=f"started with {task.model}")

            def on_tool(trace: Any) -> None:
                if task.cancel_requested:
                    return
                self._emit(
                    task,
                    "step",
                    label=f"{trace.name} ({'ok' if trace.ok else 'failed'})",
                    seconds=float(getattr(trace, "seconds", 0.0)),
                    state=RUNNING,
                )

            prompt = (
                "You are a background task agent. Complete this task and finish with a short "
                "plain-language report of what you found or did. Use the tools available. "
                "Task: " + task.brief
            )
            reply = agent.respond(
                prompt, on_tool=on_tool, should_stop=lambda: task.cancel_requested
            )
            if task.cancel_requested or getattr(reply, "cancelled", False):
                task.state = CANCELLED
                self._emit(
                    task, "cancelled", state=CANCELLED, label="cancelled during work"
                )
                return
            if getattr(reply, "error", False):
                raise RuntimeError(reply.text)
            text = str(reply.text or "").strip()[:4000]
            # Commit the result and the terminal state while holding the lock, and re-check
            # cancellation in the same breath. Review reproduced the race: a cancel arriving
            # between the check and the commit let a cancelled task finish as "done" with an
            # answer the user had already stopped.
            with self._lock:
                cancelled_now = bool(task.cancel_requested)
                if not cancelled_now:
                    task.result = text
                    task.tokens = int(getattr(reply, "tokens", 0) or 0)
                    task.cost_usd = float(getattr(reply, "cost_usd", 0.0) or 0.0)
                    task.state = DONE
                    task.finished = time.time()
            if cancelled_now:
                task.state = CANCELLED
                task.finished = time.time()
                self._emit(
                    task,
                    "cancelled",
                    state=CANCELLED,
                    label="cancelled while finishing",
                )
                return
            artifact = self._save_artifact(task)
            if artifact:
                task.artifacts.append(artifact)
            self._emit(
                task,
                "done",
                state=DONE,
                label="finished",
                artifact=artifact,
                tokens=task.tokens,
                cost_usd=task.cost_usd,
            )
            if self.on_report is not None:
                try:
                    self.on_report(task, task.result)
                except Exception as exc:  # noqa: BLE001 - reporting must not fail the task
                    log(f"[tasks] report callback failed: {type(exc).__name__}: {exc}")
            if self._deliver is not None:
                try:
                    self._deliver(task.result, task)
                except Exception as exc:  # noqa: BLE001
                    log(f"[tasks] deliver callback failed: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a failing task must not kill the app
            task.state = BLOCKED
            task.error = f"{type(exc).__name__}: {exc}"[:400]
            task.finished = time.time()
            self._emit(task, "error", state=BLOCKED, label=task.error)
            if self._deliver is not None:
                try:
                    self._deliver(f"task failed: {task.error}", task)
                except Exception as exc:  # noqa: BLE001
                    log(f"[tasks] deliver callback failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._running.pop(task.id, None)
            self._pump()

    def _save_artifact(self, task: Task) -> str:
        """Write the task's report next to the session so it can be reopened."""
        if self.session_dir is None or not task.result:
            return ""
        path = self.session_dir / "artifacts" / f"{task.id}.md"
        try:
            path.write_text(
                f"# Task {task.id}\n\n**Brief:** {task.brief}\n\n**Model:** {task.model}\n\n"
                f"## Report\n\n{task.result}\n",
                encoding="utf-8",
            )
        except OSError:
            return ""
        return str(path)

    def wait_all(self, timeout_s: float = 120.0) -> bool:
        """Block until nothing is running - used by tests, never by the UI."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                busy = [t for t in self._tasks.values() if t.state in {QUEUED, RUNNING}]
            if not busy:
                return True
            time.sleep(0.05)
        return False
