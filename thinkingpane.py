"""M1.2 (D-5): the Thinking pane - what each mind did, per turn.

The owner asked to be able to see how each agent thinks. Everything needed is
already recorded per turn (server.Session._record_turn): the model, every tool
with its arguments and result, the per-stage timings, and - for streamed turns -
per-sentence and approximate word timing. This pane renders that record honestly.

The A/B row sends the SAME question to two roles so two minds can be compared
side by side (server.Session.ab_compare starts them as normal background tasks;
their progress and answers appear here as they finish).
"""

from __future__ import annotations

import json
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any

FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


class ThinkingPane(ttk.Frame):
    """Read-only per-turn record + an A/B runner for two roles."""

    def __init__(
        self,
        parent,
        on_ab: Callable[[str, str, str], dict[str, Any]] | None = None,
        roles: list[str] | None = None,
        text_bg: str = "#0f172a",
        text_fg: str = "#e2e8f0",
        **kw: Any,
    ) -> None:
        super().__init__(parent, padding=(6, 4), **kw)
        self._on_ab = on_ab
        self._turns: list[dict[str, Any]] = []
        self._tasks: list[dict[str, Any]] = []

        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text="Turn", font=FONT_SMALL).pack(side="left")
        self.turn_pick = ttk.Combobox(head, values=[], state="readonly", width=20)
        self.turn_pick.pack(side="left", padx=(4, 0))
        self.turn_pick.bind("<<ComboboxSelected>>", lambda _e: self._render())

        row = ttk.Frame(self)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="A/B", font=FONT_SMALL).pack(side="left")
        self.ab_text = ttk.Entry(row)
        self.ab_text.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.role_a = ttk.Combobox(row, values=roles or [], state="readonly", width=9)
        self.role_a.pack(side="left")
        self.role_b = ttk.Combobox(row, values=roles or [], state="readonly", width=9)
        self.role_b.pack(side="left", padx=(4, 0))
        ttk.Button(row, text="Run", command=self._run_ab).pack(side="left", padx=(4, 0))
        if roles:
            self.role_a.set(roles[0])
            self.role_b.set(roles[min(1, len(roles) - 1)])

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(6, 0))
        self.text = tk.Text(
            body,
            wrap="word",
            font=FONT_MONO,
            width=44,
            height=30,
            state="disabled",
            background=text_bg,
            foreground=text_fg,
            insertbackground=text_fg,
        )
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

    # ------------------------------------------------------------------ data in
    def refresh(
        self,
        turns: list[dict[str, Any]] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        roles: list[str] | None = None,
    ) -> None:
        """Called by the app's poll with the engine's own state - no invented data."""
        self._turns = list(turns or [])
        self._tasks = list(tasks or [])
        labels = [self._label_for(t) for t in self._turns]
        current = self.turn_pick.get()
        self.turn_pick.configure(values=labels[-40:])
        if labels and current not in labels:
            self.turn_pick.set(labels[-1])
        if roles:
            self.role_a.configure(values=roles)
            self.role_b.configure(values=roles)
            if not self.role_a.get():
                self.role_a.set(roles[0])
            if not self.role_b.get():
                self.role_b.set(roles[min(1, len(roles) - 1)])
        self._render()

    @staticmethod
    def _label_for(turn: dict[str, Any]) -> str:
        return f"#{turn.get('id')} {str(turn.get('user_text', ''))[:30]}"

    def _selected_turn(self) -> dict[str, Any] | None:
        label = self.turn_pick.get()
        for turn in reversed(self._turns):
            if self._label_for(turn) == label:
                return turn
        return self._turns[-1] if self._turns else None

    # ------------------------------------------------------------------ render
    def _render(self) -> None:
        lines: list[str] = []
        turn = self._selected_turn()
        if turn is None:
            lines.append("no turns yet - ask something in the chat first")
        else:
            timings = turn.get("timings") or {}
            lines += [
                f"turn #{turn.get('id')}  {turn.get('ts', '')}  [{turn.get('source', '')}]",
                f"model: {turn.get('model') or '-'}",
                "",
                "YOU: " + str(turn.get("user_text", "")),
                "",
                "REPLY: " + str(turn.get("assistant_text", "")),
                "",
                "tools used:",
            ]
            tools = turn.get("tools") or []
            if not tools:
                lines.append("  (none)")
            for tool in tools:
                args = json.dumps(tool.get("arguments") or {}, ensure_ascii=False)
                lines.append(
                    f"  {tool.get('name')} ({'ok' if tool.get('ok') else 'FAILED'}) "
                    f"in {tool.get('seconds', 0)}s"
                )
                lines.append(f"    args: {args[:120]}")
                lines.append(f"    result: {str(tool.get('result', ''))[:200]}")
            lines.append("")
            lines.append("timings (ms):")
            for key in ("asr_ms", "llm_ms", "tts_ms", "total_ms", "ttfa_ms", "filler_ms"):
                if key in timings:
                    lines.append(f"  {key}: {timings[key]}")
            sentences = timings.get("sentences") or []
            if sentences:
                lines.append("")
                lines.append(f"spoken sentences ({len(sentences)}):")
                for sentence in sentences[:14]:
                    lines.append(
                        f"  +{sentence.get('start_ms')}ms  "
                        f"{str(sentence.get('text'))[:88]}"
                    )
            words = timings.get("words") or []
            if words:
                lines.append("")
                lines.append(
                    f"word timings: {len(words)} words (approximate; engine reports sentences)"
                )

        ab_tasks = [
            t for t in self._tasks if str(t.get("brief", "")).startswith("[A/B via ")
        ]
        if ab_tasks:
            lines += ["", "=" * 12 + " A/B runs " + "=" * 12]
            for task in ab_tasks[-6:]:
                role = str(task.get("brief", "")).split("]", 1)[0].replace("[A/B via ", "")
                lines.append(
                    f"- {task.get('id')} [{role}] {task.get('state')} "
                    f"({task.get('seconds')}s, {task.get('tokens')} tok)"
                )
                body = str(task.get("result") or task.get("error") or "").strip()
                lines.append("    " + body[:500].replace("\n", "\n    "))

        self._write("\n".join(lines))

    def _write(self, body: str) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", body)
        self.text.config(state="disabled")

    # ------------------------------------------------------------------ A/B run
    def _run_ab(self) -> None:
        if self._on_ab is None:
            return
        text = self.ab_text.get().strip()
        if not text:
            turn = self._selected_turn()
            text = str((turn or {}).get("user_text") or "").strip()
        role_a, role_b = self.role_a.get().strip(), self.role_b.get().strip()
        if not text or not role_a or not role_b:
            self._append_note("A/B needs a question and two roles")
            return
        result = self._on_ab(text, role_a, role_b)
        if result.get("ok"):
            runs = result.get("runs") or []
            self._append_note(
                "A/B started: "
                + " vs ".join(f"{r.get('role')} ({str(r.get('model'))[:28]})" for r in runs)
            )
        else:
            self._append_note("A/B failed: " + str(result.get("error", ""))[:120])

    def _append_note(self, note: str) -> None:
        self.text.config(state="normal")
        self.text.insert("end", "\n\n>> " + note + "\n")
        self.text.config(state="disabled")
        self.text.see("end")
