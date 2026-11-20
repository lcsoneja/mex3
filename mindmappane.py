"""The request mind map: what is being asked, and what it turned into.

Root is the latest thing the user said; branches are the router's reading of it, the tools that
ran, and the background work it started. Every node comes from recorded state, so the map is an
explanation of what happened rather than a decoration.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

ROOT_FILL = "#e8f2ff"
ROOT_LINE = "#1d4ed8"
BRANCH_FILL = "#f6f9fd"
BRANCH_LINE = "#5a6b85"
LEAF_FILL = "#ffffff"
LEAF_LINE = "#8b95a5"


class MindMapPane(ttk.Frame):
    """Draws the current request as a two-level mind map."""

    def __init__(self, master: tk.Misc, width: int = 330, height: int = 540) -> None:
        super().__init__(master, padding=(6, 4))
        self._signature = ""
        ttk.Label(self, text="What is being asked", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="nothing yet")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 8)).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=width, height=height, background="#ffffff",
                                highlightthickness=1, highlightbackground="#d7e0ec")
        self.canvas.pack(fill="both", expand=True, pady=(4, 0))

    def update_map(self, data: dict) -> None:
        signature = repr(data)
        if signature == self._signature:
            return
        self._signature = signature
        self._draw(data)

    def _draw(self, data: dict) -> None:
        canvas = self.canvas
        canvas.delete("all")
        root_text = str(data.get("root") or "(nothing asked yet)")
        branches = data.get("branches") or []
        width = float(canvas.winfo_reqwidth())
        self.status.set(f"{len(branches)} branch(es)" if branches else "nothing yet")

        # root
        canvas.create_oval(12, 10, width - 12, 54, fill=ROOT_FILL, outline=ROOT_LINE, width=2)
        canvas.create_text(width / 2, 32, text=root_text[:44], width=width - 40,
                           font=("Segoe UI", 9, "bold"), fill=ROOT_LINE)
        if data.get("reply"):
            canvas.create_text(16, 64, anchor="w", text=f"reply: {data['reply'][:38]}",
                               font=("Segoe UI", 8), fill="#1b6b3a")

        y = 92.0
        for branch in branches:
            canvas.create_line(width / 2, y - 6, width / 2, y, fill=BRANCH_LINE)
            canvas.create_rectangle(16, y, width - 16, y + 24, fill=BRANCH_FILL, outline=BRANCH_LINE)
            label = str(branch.get("label", ""))
            detail = str(branch.get("detail") or "")
            canvas.create_text(24, y + 12, anchor="w",
                               text=(label + (f"  ({detail})" if detail else ""))[:44],
                               font=("Segoe UI", 9, "bold"), fill=BRANCH_LINE)
            y += 28
            for child in branch.get("children") or []:
                canvas.create_line(40, y - 4, 52, y + 8, fill=LEAF_LINE)
                canvas.create_rectangle(52, y, width - 24, y + 20, fill=LEAF_FILL, outline=LEAF_LINE)
                canvas.create_text(60, y + 10, anchor="w",
                                   text=str(child.get("label", ""))[:40],
                                   font=("Segoe UI", 8), fill=LEAF_LINE)
                y += 23
            y += 8
            if y > float(canvas.winfo_reqheight()) - 30:
                canvas.create_text(16, y, anchor="nw", text="[... more branches]",
                                   font=("Segoe UI", 8), fill="#8b95a5")
                break
