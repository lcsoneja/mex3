"""The live workflow canvas: tasks and their steps, drawn from the event graph.

It is a projection of the task event log, so it changes whenever a task changes and it can be
rebuilt from disk after a restart. Tasks are columns, their steps hang underneath, and colours
follow the state machine in the architecture note.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

PAD = 10
TASK_H = 26
STEP_H = 17
ROW_GAP = 6

STATE_COLOURS: dict[str, tuple[str, str]] = {
    "queued": ("#eef2f8", "#5a6b85"),
    "running": ("#e8f2ff", "#1d4ed8"),
    "waiting": ("#fdf0dd", "#a15c00"),
    "blocked": ("#ffe6e6", "#b00020"),
    "done": ("#e2f0e6", "#1b6b3a"),
    "cancelled": ("#f2f2f2", "#8b95a5"),
}
DEFAULT_COLOURS = ("#f6f9fd", "#16202e")


class WorkflowView(ttk.Frame):
    """Draws the task graph; redraws only when the graph actually changed."""

    def __init__(self, master: tk.Misc, width: int = 330, height: int = 540,
                 on_cancel=None) -> None:
        super().__init__(master, padding=(6, 4))
        self.on_cancel = on_cancel
        self._signature = ""
        self._node_ids: dict[str, tuple[float, float, float, float]] = {}
        self._selected = ""

        ttk.Label(self, text="Workflow", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="no tasks yet")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 8)).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=width, height=height, background="#ffffff",
                                highlightthickness=1, highlightbackground="#d7e0ec")
        self.canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.hint = tk.StringVar(value="click a task to select it, right-click to cancel")
        ttk.Label(self, textvariable=self.hint, font=("Segoe UI", 8),
                  wraplength=width).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------ updating
    def update_view(self, graph: dict[str, Any]) -> None:
        signature = self._signature_of(graph)
        if signature == self._signature:
            return
        self._signature = signature
        self._redraw(graph)

    def update_spend(self, tokens: int, usd: float) -> None:
        self.hint.set(f"{tokens} tokens \u00b7 ${usd:.4f} this session")

    @staticmethod
    def _signature_of(graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        return "|".join(
            [f"{n.get('id')}:{n.get('state')}:{n.get('label')}" for n in nodes]
            + [f"{e.get('from')}>{e.get('to')}" for e in edges]
        )

    # ------------------------------------------------------------------ drawing
    def _redraw(self, graph: dict[str, Any]) -> None:
        canvas = self.canvas
        canvas.delete("all")
        self._node_ids.clear()
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        tasks = [n for n in nodes if n.get("kind") == "task"]
        steps_by_task: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            if node.get("kind") != "step":
                continue
            parent = str(node.get("id", "")).split(".")[0]
            steps_by_task.setdefault(parent, []).append(node)

        if not tasks:
            canvas.create_text(
                PAD, PAD,
                text="No tasks yet.\n\nAsk for something that needs a moment - reading a\n"
                     "document, checking several things, a longer search - and it will\n"
                     "appear here while the conversation carries on.",
                anchor="nw", font=("Segoe UI", 9), fill="#5a6b85", width=300,
            )
            self.status.set("no tasks yet")
            return

        running = sum(1 for t in tasks if t.get("state") == "running")
        queued = sum(1 for t in tasks if t.get("state") == "queued")
        self.status.set(f"{len(tasks)} task(s) \u00b7 {running} running \u00b7 {queued} queued")

        x0 = 12.0
        x1 = float(canvas.winfo_reqwidth()) - 12.0
        y = 12.0
        for task in tasks:
            task_id = str(task.get("id", ""))
            fill, line = STATE_COLOURS.get(str(task.get("state")), DEFAULT_COLOURS)
            width = 2 if task_id == self._selected else 1
            canvas.create_rectangle(x0, y, x1, y + TASK_H, fill=fill, outline=line, width=width)
            canvas.create_text(x0 + 6, y + TASK_H / 2, anchor="w",
                               text=f"{task_id}  {str(task.get('label', ''))[:34]}",
                               font=("Segoe UI", 9, "bold"), fill=line)
            seconds = float(task.get("seconds") or 0.0)
            canvas.create_text(x1 - 6, y + TASK_H / 2, anchor="e",
                               text=f"{task.get('state', '?')} \u00b7 {seconds:.0f}s",
                               font=("Segoe UI", 8), fill=line)
            self._node_ids[task_id] = (x0, y, x1, y + TASK_H)
            y += TASK_H + 2

            for step in steps_by_task.get(task_id, []):
                step_fill, step_line = STATE_COLOURS.get(str(step.get("state")), DEFAULT_COLOURS)
                canvas.create_line(x0 + 12, y, x0 + 12, y + STEP_H,
                                   fill=step_line, width=1)
                canvas.create_rectangle(x0 + 20, y, x1 - 10, y + STEP_H,
                                        fill=step_fill, outline=step_line)
                canvas.create_text(x0 + 26, y + STEP_H / 2, anchor="w",
                                   text=str(step.get("label", ""))[:40],
                                   font=("Segoe UI", 8), fill=step_line)
                y += STEP_H + 1
            y += ROW_GAP
            if y > float(canvas.winfo_reqheight()) - 20:
                canvas.create_text(PAD, y, anchor="nw", text="[... more tasks] "
                                   "(the list continues in the Tasks panel)",
                                   font=("Segoe UI", 8), fill="#8b95a5")
                break

        for edge in edges:
            parent = str(edge.get("from", ""))
            child = str(edge.get("to", ""))
            if parent in self._node_ids and child not in steps_by_task:
                px0, _py0, _px1, py1 = self._node_ids[parent]
                child_box = self._node_ids.get(child)
                if child_box:
                    canvas.create_line(px0 + 12, py1, child_box[0] + 12, child_box[1],
                                       fill="#8b95a5", dash=(2, 2))

    # ------------------------------------------------------------------ events
    def _task_at(self, x: float, y: float) -> str:
        for task_id, (x0, y0, x1, y1) in self._node_ids.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return task_id
        return ""

    def _on_click(self, event: tk.Event) -> None:
        task_id = self._task_at(event.x, event.y)
        self._selected = task_id if task_id != self._selected else ""
        self._signature = ""  # force a redraw so the selection shows
        self.hint.set(f"selected {task_id}" if task_id else
                      "click a task to select it, right-click to cancel")

    def _on_right_click(self, event: tk.Event) -> None:
        task_id = self._task_at(event.x, event.y)
        if not task_id or self.on_cancel is None:
            return
        self.hint.set(f"cancelling {task_id}...")
        self.on_cancel(task_id)
        self._signature = ""
