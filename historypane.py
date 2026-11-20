"""Conversation history: past sessions, newest first, ready to reopen.

Sessions are folders under out/sessions, each with a messages.jsonl. Showing the first thing that
was asked makes the list recognisable, and opening one replays it into the chat view - so history
is browsable instead of being lost when the window closes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class HistoryPane(ttk.Frame):
    """List of past conversations; double-click asks the app to open one."""

    def __init__(
        self, master: tk.Misc, on_open=None, on_resume=None, on_fork=None, on_export=None
    ) -> None:
        super().__init__(master, padding=(6, 4))
        self.on_open = on_open
        # #26/#9/#61: the pane owns its selection, so the buttons live here and the
        # app only supplies callbacks (no cross-file attribute access).
        self.on_resume = on_resume
        self.on_fork = on_fork
        self.on_export = on_export
        ttk.Label(self, text="Conversation history", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="loading...")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))

        columns = ("turns", "when")
        self.tree = ttk.Treeview(self, columns=columns, show="tree", height=16)
        self.tree.heading("#0", text="first thing asked")
        self.tree.heading("turns", text="turns")
        self.tree.heading("when", text="session")
        self.tree.column("#0", width=190, anchor="w")
        self.tree.column("turns", width=48, anchor="e", stretch=False)
        self.tree.column("when", width=118, anchor="w", stretch=False)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._open)
        self.hint = tk.StringVar(value="double-click to load a conversation")
        ttk.Label(self, textvariable=self.hint, font=("Segoe UI", 8),
                  wraplength=330).pack(anchor="w", pady=(4, 0))
        self._rows: dict[str, str] = {}
        if on_resume or on_fork or on_export:
            buttons = ttk.Frame(self)
            buttons.pack(fill="x", pady=(4, 0))
            if on_resume:
                ttk.Button(buttons, text="Resume", command=self._resume).pack(side="left")
            if on_fork:
                ttk.Button(buttons, text="Fork", command=self._fork).pack(side="left", padx=(6, 0))
            if on_export:
                ttk.Button(buttons, text="Export Markdown",
                           command=on_export).pack(side="left", padx=(6, 0))

    def refresh(self, sessions: list[dict]) -> None:
        """Redraw from Session.list_sessions()."""
        signature = "|".join(f"{s.get('id')}:{s.get('turns')}:{s.get('title')}" for s in sessions)
        if signature == getattr(self, "_signature", ""):
            return
        self._signature = signature
        self.tree.delete(*self.tree.get_children())
        self._rows.clear()
        for session in sessions:
            marker = " (current)" if session.get("current") else ""
            label = f"{session.get('title') or '(empty)'}{marker}"
            item = self.tree.insert("", "end", text=label,
                                    values=(session.get("turns", 0), session.get("id", "")))
            self._rows[item] = str(session.get("id", ""))
        self.status.set(f"{len(sessions)} session(s)")

    def _open(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection or self.on_open is None:
            return
        session_id = self._rows.get(selection[0], "")
        if session_id:
            self.hint.set(f"loading {session_id} ...")
            self.on_open(session_id)

    def _resume(self) -> None:
        """Resume button: make the selected past session current again (#26)."""
        session_id = self.selected_session()
        if not session_id:
            self.hint.set("select a session first")
            return
        if self.on_resume is None:
            return
        self.hint.set(f"resuming {session_id} ...")
        self.on_resume(session_id)

    def _fork(self) -> None:
        """Fork button: same past, independent future (#9)."""
        session_id = self.selected_session()
        if not session_id:
            self.hint.set("select a session first")
            return
        if self.on_fork is None:
            return
        self.hint.set(f"forking {session_id} ...")
        self.on_fork(session_id)

    def selected_session(self) -> str:
        """The session id of the current tree selection ('' when nothing is selected).

        Added 2026-11-16 for the Resume/Fork buttons (#26/#9): the same selection
        logic _open uses, as a query the buttons can call without opening anything.
        """
        selection = self.tree.selection()
        if not selection:
            return ""
        return self._rows.get(selection[0], "")

    def set_hint(self, text: str) -> None:
        self.hint.set(text)
