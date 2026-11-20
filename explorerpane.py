"""Folder pane: which folders the agents may read, and what is in them.

Modelled on the file explorers people already know - a tree, a folder picker, and a preview -
but with one important difference: the tree and the readers share the same allowlist, so what you
see listed is exactly what the agents can open, and nothing else.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import filereader


class ExplorerPane(ttk.Frame):
    """Allowed folders and their readable files."""

    def __init__(self, master: tk.Misc, on_open_file=None) -> None:
        super().__init__(master, padding=(6, 4))
        self.on_open_file = on_open_file
        ttk.Label(self, text="Folders the agents can read", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 4))

        self.tree = ttk.Treeview(self, columns=("kind",), show="tree", height=18)
        self.tree.heading("#0", text="folder / file")
        self.tree.heading("kind", text="kind")
        self.tree.column("#0", width=230, anchor="w")
        self.tree.column("kind", width=70, anchor="w", stretch=False)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._open)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="Add folder", command=self._add_folder).pack(side="left")
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(side="left", padx=(6, 0))
        self.hint = tk.StringVar(value="double-click a file to send it to the chat")
        ttk.Label(self, textvariable=self.hint, font=("Segoe UI", 8),
                  wraplength=330).pack(anchor="w", pady=(4, 0))
        self._files: dict[str, str] = {}

    def refresh(self) -> None:
        """Rebuild the tree from the allowlist, so the pane cannot show what the agents cannot read."""
        self.tree.delete(*self.tree.get_children())
        self._files.clear()
        folders = filereader.allowed_folders()
        total = 0
        for folder in folders:
            node = self.tree.insert("", "end", text=str(folder), values=("folder",), open=False)
            try:
                rows = filereader.list_folder(folder, "*", limit=300)
            except filereader.FileError as exc:
                self.tree.insert(node, "end", text=f"(unreadable: {exc})", values=("error",))
                continue
            for row in rows:
                item = self.tree.insert(node, "end", text=str(row["name"]),
                                        values=(str(row["kind"]),))
                self._files[item] = str(folder / str(row["name"]))
                total += 1
        self.status.set(f"{len(folders)} folder(s), {total} readable file(s)")

    def _add_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Grant read access to a folder")
        if not chosen:
            return
        try:
            filereader.allow_folder(chosen)
            self.hint.set(f"granted read access to {chosen}")
        except filereader.FileError as exc:
            self.hint.set(str(exc))
        self.refresh()

    def _open(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection or self.on_open_file is None:
            return
        path = self._files.get(selection[0])
        if path:
            self.hint.set(f"opening {path.split(chr(92))[-1]} ...")
            self.on_open_file(path)
