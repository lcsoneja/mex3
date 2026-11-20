"""Subagents and their sub-workflows: the back layer, as a list.

The canvas shows the shape of the work; this pane shows the detail - which agent, which model,
how long, how many steps, and what each step did - and it is the place to cancel one.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

STATE_ORDER = {"running": 0, "queued": 1, "waiting": 2, "blocked": 3, "done": 4, "cancelled": 5}


class SubagentsPane(ttk.Frame):
    """Task rows with their steps as children."""

    def __init__(self, master: tk.Misc, on_cancel=None) -> None:
        super().__init__(master, padding=(6, 4))
        self.on_cancel = on_cancel
        self._signature = ""
        ttk.Label(self, text="Subagents and sub-workflows", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="no tasks yet")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))

        columns = ("state", "model", "steps", "secs", "cost")
        self.tree = ttk.Treeview(self, columns=columns, show="tree", height=18)
        self.tree.heading("#0", text="task / step")
        for key, title in zip(columns, ("state", "model", "steps", "s", "$"), strict=True):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=54 if key != "model" else 110, anchor="w", stretch=False)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Button-3>", self._cancel)
        self.hint = tk.StringVar(value="right-click a task to cancel it")
        ttk.Label(self, textvariable=self.hint, font=("Segoe UI", 8),
                  wraplength=330).pack(anchor="w", pady=(4, 0))
        self._tasks: dict[str, str] = {}

    def update_view(self, tasks: list[dict], graph: dict | None = None) -> None:
        signature = "|".join(f"{t.get('id')}:{t.get('state')}:{t.get('steps')}" for t in tasks)
        if signature == self._signature:
            return
        self._signature = signature
        self.tree.delete(*self.tree.get_children())
        self._tasks.clear()
        steps_by_task: dict[str, list[dict]] = {}
        for node in (graph or {}).get("nodes", []):
            if node.get("kind") == "step":
                parent = str(node.get("id", "")).split(".")[0]
                steps_by_task.setdefault(parent, []).append(node)

        ordered = sorted(tasks, key=lambda t: STATE_ORDER.get(str(t.get("state")), 9))
        running = sum(1 for t in tasks if t.get("state") == "running")
        queued = sum(1 for t in tasks if t.get("state") == "queued")
        self.status.set(f"{len(tasks)} task(s) | {running} running, {queued} queued")
        for task in ordered:
            task_id = str(task.get("id", ""))
            item = self.tree.insert(
                "", "end", text=f"{task_id}  {str(task.get('brief', ''))[:34]}",
                values=(task.get("state", "?"), task.get("model", "")[:22],
                        task.get("steps", 0), f"{float(task.get('seconds') or 0):.0f}",
                        f"{(float(task.get('cost_usd') or 0)):.4f}"),
                open=running > 0,
            )
            self._tasks[item] = task_id
            for step in steps_by_task.get(task_id, []):
                self.tree.insert(item, "end", text=str(step.get("label", ""))[:38],
                                 values=(step.get("state", ""), "", "",
                                         f"{float(step.get('seconds') or 0):.1f}", ""))

    def _cancel(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        parent = self.tree.parent(item) or item
        task_id = self._tasks.get(parent, "")
        if not task_id or self.on_cancel is None:
            return
        self.hint.set(f"cancelling {task_id} ...")
        self.on_cancel(task_id)
        self._signature = ""
