"""SONJ standalone desktop app - no browser, works on a Pi with no GUI browser.

A native Tk window over the same engine API the web console uses. On a Raspberry
Pi this is the default face (Tk ships with Python; a browser costs RAM). Same
state, same session: what you type here appears in the web console's history.

Run:  python me3_laptop/sonj_tk.py      (or: sonj app)
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import tkinter as tk
import traceback
import types
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any
from urllib.parse import urlparse

import requests

from doctorpane import DoctorPane

BASE = "http://127.0.0.1:8765"


def api_json(path: str, body: dict | None = None) -> dict[str, Any]:
    """Like api(), but never bare text.

    The Doctor pane needs a dict to render; an engine that answers with HTML or
    an error page must become a readable refusal, not a type error in the UI.
    """
    raw = api(path, body)
    return raw if isinstance(raw, dict) else {"error": str(raw)[:200]}


def _local(url: str) -> str:
    # Same control-plane rule as sonj.py: local engine only.
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("SONJ talks to the local engine only")
    return url


def api(path: str, body: dict | None = None) -> dict | str:
    try:
        if body is None:
            r = requests.get(_local(BASE + path), timeout=10)
        else:
            r = requests.post(_local(BASE + path), json=body, timeout=30)
        return (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)[:80]}


class SonjTk:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SONJ - Nova & Piattos")
        # RULE-39: multiple windows are allowed - each new SONJ opens another app.
        self.root.geometry("720x520")
        self.root.configure(bg="#0d1017")

        top = tk.Frame(self.root, bg="#12161f", pady=6, padx=10)
        top.pack(fill="x")
        tk.Label(
            top, text="SONJ", bg="#12161f", fg="#5aa2ff", font=("Segoe UI", 11, "bold")
        ).pack(side="left")
        self.status = tk.Label(top, text="connecting...", bg="#12161f", fg="#8b94a7")
        self.status.pack(side="right")

        # S4: which agents are working right now - always on screen, not hidden
        # behind a tab, because "is anything actually running?" is the question
        # the owner asks during a demo.
        self.agents_strip = tk.Label(
            self.root,
            text="agents: connecting...",
            bg="#0d1017",
            fg="#8b94a7",
            anchor="w",
            font=("Segoe UI", 8),
        )
        self.agents_strip.pack(fill="x", padx=10, pady=(6, 0))

        self.log = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            state="disabled",
            bg="#0d1017",
            fg="#e7ebf3",
            insertbackground="#e7ebf3",
        )
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        for tag, color in (
            ("you", "#8b5cf6"),
            ("assistant", "#34d399"),
            ("error", "#ef4444"),
            ("dim", "#8b94a7"),
        ):
            self.log.tag_configure(tag, foreground=color)

        topb = tk.Button(
            top,
            text="Registry",
            command=self.open_registry,
            bg="#1c2231",
            fg="#e7ebf3",
            relief="flat",
        )
        topb.pack(side="right", padx=4)
        # #54 (owner request): one button expands this window into the full
        # workspace - same window, same session, same engine. Collapse returns to
        # the compact chat.
        self.expand_btn = tk.Button(
            top,
            text="Expand ▸",
            command=self.toggle_expand,
            bg="#1c2231",
            fg="#5aa2ff",
            relief="flat",
        )
        self.expand_btn.pack(side="right", padx=4)
        bar = tk.Frame(self.root, bg="#12161f", pady=6, padx=8)
        bar.pack(fill="x", side="bottom")
        self.entry = tk.Entry(
            bar, bg="#0d1017", fg="#e7ebf3", insertbackground="#e7ebf3"
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.send)
        tk.Button(
            bar, text="Send", command=self.send, bg="#5aa2ff", fg="white", relief="flat"
        ).pack(side="left", padx=4)
        self.listen_btn = tk.Button(
            bar,
            text="Listen",
            command=self.toggle_listen,
            bg="#1c2231",
            fg="#e7ebf3",
            relief="flat",
        )
        self.listen_btn.pack(side="left", padx=2)
        self._listening = False

        self.last_turns = -1
        self.expanded = False
        self._files_at = 0.0
        # S2: the repair catalogue is fetched once; the pane never invents one.
        self._doctor_loaded = False
        # Owner request: per-role model dropdowns, filled once when the rail opens.
        self._roles_loaded = False
        # S0: True only when this window started the engine, so closing a window
        # can never kill an engine it does not own.
        self._owns_engine = False
        # S1.2: a Tk callback that raises is recorded, not printed to a console
        # nobody watches (the owner's runtime error left no trace before this).
        self.root.report_callback_exception = self._report_callback_exception
        # Build the workspace rail once, hidden: the tabs live in the same window
        # and read the same engine over the same local API (no second engine).
        self._build_rail()
        # RULE-39: count open windows so the engine closes only with the last one
        cf = Path(__file__).resolve().parent / "out" / "sonj_windows.count"
        try:
            cf.write_text(
                str(int(cf.read_text(encoding="utf-8").strip() or "0") + 1),
                encoding="utf-8",
            )
        except (OSError, ValueError):
            cf.write_text("1", encoding="utf-8")
        self._pump()

    # --- actions ---------------------------------------------------------
    def send(self, event: object = None) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        api("/api/send", {"text": text})

    def toggle_listen(self) -> None:
        if self._listening:
            api("/api/stop", {})
            self._listening = False
            self.listen_btn.config(text="Listen", bg="#1c2231")
        else:
            api("/api/stop", {})
            api("/api/listen", {"mode": "continuous"})
            self._listening = True
            self.listen_btn.config(text="Stop", bg="#ef4444")

    def open_registry(self) -> None:
        """Everything the agent has available: tools, skills, prompts, MCP, resources."""
        win = tk.Toplevel(self.root)
        win.title("SONJ registry")
        win.configure(bg="#0d1017")
        win.geometry("560x520")
        raw = api("/api/registry")
        items_map: dict = raw if isinstance(raw, dict) else {}
        body = tk.Text(win, wrap="word", bg="#0d1017", fg="#e7ebf3")
        body.pack(fill="both", expand=True, padx=8, pady=8)
        if not items_map:
            body.insert("end", "engine not reachable - start it with: sonj")
            return
        for section, key in (
            ("MCP servers", "mcp"),
            ("Tools", "tools"),
            ("Skills", "skills"),
            ("Prompt templates", "prompts"),
            ("Resources (attachments)", "resources"),
        ):
            items = items_map.get(key) or []
            if key == "tools":
                items = [t.get("name") if isinstance(t, dict) else t for t in items]
            body.insert(
                "end", section.upper() + " (" + str(len(items)) + ")" + chr(10), "h"
            )
            for it in items:
                if isinstance(it, str):
                    body.insert("end", "  " + it + chr(10))
                elif isinstance(it, dict):
                    name = str(it.get("name", "?"))
                    body.insert(
                        "end",
                        "  "
                        + name
                        + " - "
                        + str(it.get("description", ""))[:90]
                        + chr(10),
                    )
            body.insert("end", chr(10))
        body.tag_configure("h", foreground="#5aa2ff", font=("Segoe UI", 11, "bold"))
        body.config(state="disabled")

    def _build_rail(self) -> None:
        """#54: the workspace rail - Sessions / Files / Agents - hidden until Expand."""
        from explorerpane import ExplorerPane
        from historypane import HistoryPane

        self.rail = ttk.Notebook(self.root, width=330)
        self.history = HistoryPane(
            self.rail,
            on_open=self._open_session_http,
            on_resume=self._resume_session_http,
            on_fork=self._fork_session_http,
        )
        self.rail.add(self.history, text="Sessions")
        self.explorer = ExplorerPane(self.rail, on_open_file=self._ask_read_file)
        self.rail.add(self.explorer, text="Files")
        agents = ttk.Frame(self.rail, padding=(6, 4))
        # Owner request: change the AI model PER ROLE from this window too. A
        # dropdown writes agents.json through /api/set_role_model, so the next turn
        # uses the new model with no restart (routing is data).
        ttk.Label(agents, text="Model per role", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self._role_rows: dict[str, ttk.Combobox] = {}
        self._role_frame = ttk.Frame(agents)
        self._role_frame.pack(fill="x", pady=(4, 6))
        self.role_hint = tk.StringVar(value="roster loads from the engine")
        ttk.Label(
            agents,
            textvariable=self.role_hint,
            font=("Segoe UI", 8),
            wraplength=300,
            justify="left",
        ).pack(anchor="w")
        self.agents_label = tk.Label(
            agents,
            text="",
            bg="#0d1017",
            fg="#8b94a7",
            justify="left",
            anchor="w",
            wraplength=300,
        )
        self.agents_label.pack(fill="both", expand=True, pady=(6, 0))
        self.rail.add(agents, text="Agents")
        # M1.3: the Thinking pane in the same rail - what each mind did, per turn.
        from thinkingpane import ThinkingPane

        self.thinking = ThinkingPane(self.rail, text_bg="#0d1017", text_fg="#e7ebf3")
        self.rail.add(self.thinking, text="Thinking")
        # S1/S2/S3: the Doctor tab - who investigates, the engine's repair
        # catalogue, the verifier, and the incident log. Same callbacks as the
        # full app, over the local API instead of in-process.
        self.doctor = DoctorPane(
            self.rail,
            on_diagnose=lambda scope, agent: api_json(
                "/api/diagnose", {"scope": scope, "agent": agent}
            ),
            on_repair=lambda repair_id, dry: api_json(
                "/api/repair", {"id": repair_id, "dry_run": dry}
            ),
            on_verify=lambda: api_json("/api/verify", {}),
            on_settings=self._push_doctor_settings,
            on_watch=lambda on: api_json("/api/settings", {"doctor_watch": bool(on)}),
            on_fix_top=lambda: api_json("/api/fix_now", {}),
            on_escalate=lambda question: api_json(
                "/api/escalate", {"question": question}
            ),
            text_bg="#0d1017",
            text_fg="#e7ebf3",
        )
        self.rail.add(self.doctor, text="Doctor")

    def toggle_expand(self) -> None:
        """Same window, same session: show or hide the workspace rail."""
        if not self.expanded:
            self.expanded = True
            self.root.geometry("1180x760")
            self.rail.pack(side="left", fill="y", padx=(8, 0), before=self.log)
            self.explorer.refresh()
            self.expand_btn.config(text="Collapse ◂")
        else:
            self.expanded = False
            self.rail.pack_forget()
            self.root.geometry("720x520")
            self.expand_btn.config(text="Expand ▸")

    def _open_session_http(self, session_id: str) -> None:
        """#60/#26: replay a past conversation from the engine over the local API."""
        raw = api("/api/session/load", {"id": session_id})
        turns = raw.get("turns") if isinstance(raw, dict) else None
        if not turns:
            self.history.set_hint(f"{session_id} has no messages")
            return
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        for t in turns:
            mine = t.get("source") != "task"
            self.log.insert(
                "end", ("you" if mine else "subagent") + "  ", "you" if mine else "dim"
            )
            self.log.insert("end", str(t.get("user_text", "")) + chr(10), "dim")
            self.log.insert("end", "Nova  ", "assistant")
            self.log.insert(
                "end", str(t.get("assistant_text", "")) + chr(10) + chr(10), ""
            )
        self.log.config(state="disabled")
        self.log.see("end")
        self.history.set_hint(f"showing {len(turns)} turns from {session_id}")

    def _resume_session_http(self, session_id: str) -> None:
        """#26: resume a past session over the local API; new turns append there."""
        raw = api("/api/session/resume", {"id": session_id})
        ok = isinstance(raw, dict) and raw.get("ok")
        self.history.set_hint(
            ("resumed " + session_id + " - new turns append there")
            if ok
            else f"could not resume {session_id}: {raw}"
        )

    def _fork_session_http(self, session_id: str) -> None:
        """#9: fork a past session over the local API."""
        raw = api("/api/session/fork", {"id": session_id})
        if isinstance(raw, dict) and raw.get("ok"):
            self.history.set_hint("forked to " + str(raw.get("id", "?")))
        else:
            self.history.set_hint(f"could not fork {session_id}: {raw}")

    def _ask_read_file(self, path: str) -> None:
        """Send the engine a read request for a file picked in the rail."""
        api("/api/send", {"text": f"Read {path} and summarise what matters in it."})

    # --- state pump (same stream the web console polls) -------------------
    def _pump(self) -> None:
        self._ensure_engine()
        s = api("/api/state")
        if isinstance(s, dict) and "status" in s:
            if self.expanded:
                # Rail data comes from the same state: session list for the pane,
                # roster for the Agents tab, session dir in the status line.
                now = time.time()
                if now - self._files_at > 15.0:
                    self._files_at = now
                    with contextlib.suppress(
                        Exception
                    ):  # a pane must never kill the pump
                        self.explorer.refresh()
                raw = api("/api/sessions")
                if isinstance(raw, dict):
                    with contextlib.suppress(Exception):
                        self.history.refresh(raw.get("sessions") or [])
                agents = s.get("agents") or {}
                if isinstance(agents, dict) and agents:
                    lines = [
                        f"{role}: {(cfg or {}).get('model') or '-'}"
                        for role, cfg in agents.items()
                    ]
                    self.agents_label.config(text=chr(10).join(lines))
                    if not self._roles_loaded:
                        self._fill_role_rows(agents)
                        self._roles_loaded = True
                session_dir = str(s.get("session_dir", ""))
                self.root.title(
                    f"SONJ - Nova & Piattos | {Path(session_dir).name}"
                    if session_dir
                    else "SONJ - Nova & Piattos"
                )
                with contextlib.suppress(Exception):  # a pane must never kill the pump
                    self.thinking.refresh(
                        list(s.get("turns") or []),
                        list(s.get("tasks") or []),
                        sorted(str(r) for r in (agents or {}))
                        if isinstance(agents, dict)
                        else [],
                    )
                if not self._doctor_loaded:
                    raw = api_json("/api/doctor")
                    if raw.get("repairs"):
                        self.doctor.set_catalogue(
                            raw.get("repairs") or [],
                            raw.get("scopes") or [],
                            raw.get("agents") or [],
                        )
                        self._doctor_loaded = True
                with contextlib.suppress(Exception):  # a pane must never kill the pump
                    self.doctor.refresh(
                        list(s.get("incidents") or []),
                        list(s.get("agents_active") or []),
                        s.get("doctor") or {},
                        int(s.get("incident_count") or 0),
                    )
            who = (
                "assistant speaking"
                if s.get("speaking")
                else "thinking"
                if s.get("status") == "thinking"
                else "listening"
                if s.get("recording") and s.get("mode") == "continuous"
                else s.get("status", "idle")
            )
            self.status.config(
                text=who + " · " + str(s.get("memory_mb", 0))[:5] + " MB"
            )
            self.agents_strip.config(
                text=self._agents_line(list(s.get("agents_active") or []))
            )
            turns = s.get("turns", [])
            if len(turns) != self.last_turns:
                self.last_turns = len(turns)
                self.log.config(state="normal")
                self.log.delete("1.0", "end")
                for t in turns:
                    mine = t.get("source") != "task"
                    self.log.insert(
                        "end",
                        ("you" if mine else "subagent") + "  ",
                        "you" if mine else "dim",
                    )
                    self.log.insert("end", str(t.get("user_text", "")) + chr(10), "dim")
                    self.log.insert("end", "Nova  ", "assistant")
                    self.log.insert(
                        "end", str(t.get("assistant_text", "")) + chr(10) + chr(10), ""
                    )
                self.log.config(state="disabled")
                self.log.see("end")
        self.root.after(900, self._pump)

    def on_close(self) -> None:
        # RULE-38: when the last window closes, no leaked processes remain - the
        # engine (a separate child process) is stopped with the app.
        import os

        count_file = Path(__file__).resolve().parent / "out" / "sonj_windows.count"
        try:
            count = int(count_file.read_text(encoding="utf-8").strip() or "1")
        except (OSError, ValueError):
            count = 1
        count -= 1
        if count <= 0:
            # S4.2: the session report is written before the engine goes away.
            with contextlib.suppress(Exception):
                api("/api/session_report", {})
            api("/api/barge", {})
            api("/api/stop", {})
            if self._owns_engine:
                api("/api/shutdown", {})
            # S0: only stop an engine this window started. Before this, closing a
            # window killed whatever pid was in the file - including an engine it
            # never owned (the full app, or a CLI-started one).
            if not self._owns_engine:
                try:
                    count_file.unlink()
                except OSError as exc:
                    print("window count file not removable:", type(exc).__name__)
                self.root.destroy()
                return
            pid_file = Path(__file__).resolve().parent / "out" / "engine.pid"
            import subprocess

            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        timeout=5,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    os.kill(pid, 15)
                pid_file.unlink()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # engine already gone or pid unreadable - the close still succeeds
                print("engine cleanup skipped:", type(exc).__name__)
            try:
                count_file.unlink()
            except OSError as exc:
                print("window count file not removable:", type(exc).__name__)
        else:
            count_file.write_text(str(count), encoding="utf-8")
        self.root.destroy()

    def _engine_healthy(self) -> bool:
        raw = api("/api/health")
        return isinstance(raw, dict) and bool(raw.get("ok"))

    def _push_doctor_settings(self, enabled: bool, kill_switch: bool) -> dict[str, Any]:
        """S2: the doctor's switches live in the engine's config, over the API."""
        return api_json(
            "/api/doctor", {"enabled": bool(enabled), "kill_switch": bool(kill_switch)}
        )

    def _report_callback_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """S1.2: keep the window alive and record why the callback died (text only)."""
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(text, file=sys.stderr)
        with contextlib.suppress(Exception):
            api(
                "/api/incidents",
                {
                    "kind": "ui",
                    "message": f"{getattr(exc_type, '__name__', exc_type)}: {exc_value}",
                    "severity": "error",
                    "where": "sonj_tk",
                    "traceback": text[-1200:],
                },
            )
        with contextlib.suppress(Exception):
            self.status.config(text="error - see the Doctor tab", fg="#ef4444")

    def _agents_line(self, rows: list) -> str:
        """S4: one line naming every agent and whether it is busy."""
        if not rows:
            return "agents: no report yet"
        quiet = {"idle", "ready", "local-only"}
        busy = [
            f"{r.get('role')}:{r.get('state')}"
            for r in rows
            if str(r.get("state")) not in quiet
        ]
        rest = [str(r.get("role")) for r in rows if str(r.get("state")) in quiet]
        head = " * ".join(busy) if busy else "nothing running"
        tail = f"   |   idle: {', '.join(rest)}" if rest else ""
        return f"agents: {head}{tail}"

    def _ensure_engine(self) -> None:
        """Start the engine only when nobody owns the voice (S0).

        The old check asked only "is something healthy?" - so two windows opened
        at the same moment both spawned an engine and the demo ended up with two
        audio stacks fighting for the speakers. Now a live lock owner means this
        window waits for it instead of starting a second voice.
        """
        if self._engine_healthy():
            return
        from doctor import engine_owner

        owner = engine_owner()
        if owner and owner != os.getpid():
            for _ in range(40):
                time.sleep(0.5)
                if self._engine_healthy():
                    return
            owner = engine_owner()
            if owner and owner != os.getpid():
                self.status.config(
                    text=f"another engine owns the voice (pid {owner})", fg="#f59e0b"
                )
                return
        import subprocess

        engine = Path(__file__).resolve().parent / "server.py"
        subprocess.Popen(
            [sys.executable, str(engine), "--port", "8765"],
            cwd=str(Path(__file__).resolve().parent),
        )
        # The engine writes its own pid file while claiming the lock; a window
        # must not write it, or a losing window would clobber the winner's pid.
        self._owns_engine = True
        for _ in range(60):
            time.sleep(0.5)
            if self._engine_healthy():
                return

    def _fill_role_rows(self, roster: dict) -> None:
        """One dropdown per roster role; a change writes agents.json via the API."""
        for child in self._role_frame.winfo_children():
            child.destroy()
        self._role_rows.clear()
        for name in sorted(roster):
            cfg = roster[name] if isinstance(roster.get(name), dict) else {}
            row = ttk.Frame(self._role_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=name, width=10).pack(side="left")
            combo = ttk.Combobox(row, values=self._role_choices(roster, cfg), width=22)
            combo.set(str(cfg.get("model") or ""))
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, n=name, c=combo: self._push_role_model(n, c.get()),
            )
            combo.pack(side="left", padx=(2, 0))
            self._role_rows[name] = combo

    @staticmethod
    def _role_choices(roster: dict, cfg: dict) -> list[str]:
        """Every model already used by a role, plus this role's fallbacks."""
        choices: list[str] = []
        for entry in roster.values():
            model = str((entry or {}).get("model") or "")
            if model and model not in choices:
                choices.append(model)
        for model in cfg.get("fallbacks") or []:
            text = str(model)
            if text and text not in choices:
                choices.append(text)
        return choices

    def _push_role_model(self, role: str, model: str) -> None:
        """Hot-swap one role's model (routing is data in agents.json)."""
        raw = api_json("/api/set_role_model", {"role": role, "model": model})
        self.role_hint.set(str(raw.get("message") or raw.get("error") or "saved"))

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()


def main() -> int:
    app = SonjTk()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
