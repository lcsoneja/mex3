"""S2/S3/S4: the Doctor pane - who investigates, what scope, and what was repaired.

The owner asked for a visible, self-healing app: when something breaks during a
demo, the window should say what happened and offer a repair. This pane is the
human face of that. It never invents a diagnosis - it renders whatever the
engine's Doctor returned (server.py /api/diagnose, /api/repair, /api/verify) and
the incident log the app already keeps (errors.py -> out/incidents.jsonl).

Two rules the pane enforces, because the owner asked for them:
  * the owner picks the investigator (doctor, verifier, front, router, ...) and
    the scope; nothing runs without a press;
  * a repair is shown as "fixed" only when the independent verifier agrees -
    the engine already rolls back when it does not.

Text only. No audio is ever read or written here.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any

FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)

SEVERITY_MARK = {
    "fatal": "!!",
    "error": "! ",
    "warning": "~ ",
    "info": "  ",
}


class DoctorPane(ttk.Frame):
    """Diagnose / repair / verify, plus the incident log and the agent strip."""

    def __init__(
        self,
        parent: Any,
        on_diagnose: Callable[[str, str], dict[str, Any]] | None = None,
        on_repair: Callable[[str, bool], dict[str, Any]] | None = None,
        on_verify: Callable[[], dict[str, Any]] | None = None,
        on_settings: Callable[[bool, bool], dict[str, Any]] | None = None,
        on_watch: Callable[[bool], dict[str, Any]] | None = None,
        on_fix_top: Callable[[], dict[str, Any]] | None = None,
        on_escalate: Callable[[str], dict[str, Any]] | None = None,
        text_bg: str = "#0f172a",
        text_fg: str = "#e2e8f0",
        **kw: Any,
    ) -> None:
        super().__init__(parent, padding=(6, 4), **kw)
        self._on_diagnose = on_diagnose
        self._on_repair = on_repair
        self._on_verify = on_verify
        self._on_settings = on_settings
        # S4.1 / S3.5: demo watch, one-click Fix now, and the escalation proposal.
        self._on_watch = on_watch
        self._on_fix_top = on_fix_top
        self._on_escalate = on_escalate
        self._repairs: list[dict[str, Any]] = []
        self._report = ""
        self._incident_count = -1

        # --- who investigates, and what --------------------------------
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text="Investigate with", font=FONT_SMALL).pack(side="left")
        self.agent_pick = ttk.Combobox(
            head, values=["doctor"], state="readonly", width=12
        )
        self.agent_pick.set("doctor")
        self.agent_pick.pack(side="left", padx=(4, 8))
        ttk.Label(head, text="Scope", font=FONT_SMALL).pack(side="left")
        self.scope_pick = ttk.Combobox(
            head, values=["all"], state="readonly", width=10
        )
        self.scope_pick.set("all")
        self.scope_pick.pack(side="left", padx=(4, 0))

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Diagnose", command=self._diagnose).pack(side="left")
        ttk.Button(actions, text="Verify", command=self._verify).pack(
            side="left", padx=(4, 0)
        )
        # S4.1: under pressure, one press repairs the worst finding with a safe fix.
        ttk.Button(actions, text="Fix now", command=self._fix_top).pack(
            side="left", padx=(4, 0)
        )
        self.dry_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            actions, text="dry run", variable=self.dry_run
        ).pack(side="left", padx=(8, 0))

        # --- the catalogue --------------------------------------------
        rep_box = ttk.LabelFrame(self, text="Repairs (pick one, then Repair)", padding=4)
        rep_box.pack(fill="x", pady=(6, 0))
        self.repair_list = tk.Listbox(
            rep_box, height=5, font=FONT_MONO, exportselection=False,
            background=text_bg, foreground=text_fg,
            selectbackground="#2b6cb0", selectforeground="#ffffff",
            borderwidth=0, highlightthickness=0,
        )
        self.repair_list.pack(fill="x", side="left", expand=True)
        rep_scroll = ttk.Scrollbar(rep_box, orient="vertical",
                                   command=self.repair_list.yview)
        rep_scroll.pack(side="right", fill="y")
        self.repair_list.configure(yscrollcommand=rep_scroll.set)
        ttk.Button(self, text="Repair selected", command=self._repair).pack(
            anchor="e", pady=(4, 0)
        )

        # --- switches -------------------------------------------------
        switches = ttk.Frame(self)
        switches.pack(fill="x", pady=(6, 0))
        self.enabled = tk.BooleanVar(value=True)
        self.kill_switch = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            switches, text="doctor on", variable=self.enabled,
            command=self._push_settings,
        ).pack(side="left")
        ttk.Checkbutton(
            switches, text="kill switch", variable=self.kill_switch,
            command=self._push_settings,
        ).pack(side="left", padx=(8, 0))
        # S4.1: demo mode - the doctor watches in the background, detect-only.
        self.watch = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            switches, text="demo watch", variable=self.watch,
            command=self._push_watch,
        ).pack(side="left", padx=(8, 0))

        # --- which agents are active right now ------------------------
        strip_box = ttk.LabelFrame(self, text="Agents active", padding=4)
        strip_box.pack(fill="x", pady=(6, 0))
        self.agents_strip = tk.Text(
            strip_box, height=4, wrap="word", font=FONT_MONO, state="disabled",
            background=text_bg, foreground=text_fg, borderwidth=0,
            highlightthickness=0,
        )
        self.agents_strip.pack(fill="x")

        # --- the report -----------------------------------------------
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(6, 0))
        self.text = tk.Text(
            body, wrap="word", font=FONT_MONO, width=44, height=14, state="disabled",
            background=text_bg, foreground=text_fg, insertbackground=text_fg,
            borderwidth=0, highlightthickness=0,
        )
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="Copy report", command=self._copy).pack(side="left")
        ttk.Button(buttons, text="Save report", command=self._save).pack(
            side="left", padx=(4, 0)
        )
        # S3.5: a code-level fix is proposed to the harness, and it stays behind
        # the approval gate; this button never applies code by itself.
        ttk.Button(buttons, text="Escalate", command=self._escalate).pack(
            side="left", padx=(4, 0)
        )
        self.hint = tk.StringVar(value="nothing diagnosed yet")
        ttk.Label(self, textvariable=self.hint, font=FONT_SMALL,
                  wraplength=320, foreground="#6b7280").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------ data in
    def set_catalogue(self, repairs: list[dict[str, Any]], scopes: list[str],
                      agents: list[str]) -> None:
        """The engine's own catalogue - the pane never hard-codes a repair."""
        if scopes:
            self.scope_pick.configure(values=list(scopes))
            if self.scope_pick.get() not in scopes:
                self.scope_pick.set(scopes[0])
        if agents:
            self.agent_pick.configure(values=list(agents))
            if self.agent_pick.get() not in agents:
                self.agent_pick.set(agents[0])
        if repairs and len(repairs) != len(self._repairs):
            self._repairs = list(repairs)
            self.repair_list.delete(0, "end")
            for entry in self._repairs:
                self.repair_list.insert(
                    "end", f"{entry.get('id', '?')}  {entry.get('title', '')}"
                )

    def refresh(
        self,
        incidents: list[dict[str, Any]] | None = None,
        agents_active: list[dict[str, Any]] | None = None,
        doctor: dict[str, Any] | None = None,
        count: int | None = None,
    ) -> None:
        """Called by the host's poll with the engine's own state."""
        if agents_active is not None:
            self._render_agents(agents_active)
        if doctor is not None:
            self.enabled.set(bool(doctor.get("enabled", True)))
            self.kill_switch.set(bool(doctor.get("kill_switch", False)))
            self.watch.set(bool(doctor.get("watch", False)))
        if incidents is not None and count != self._incident_count:
            self._incident_count = count if count is not None else len(incidents)
            self._render_incidents(incidents)

    # ------------------------------------------------------------ render
    def _render_agents(self, rows: list[dict[str, Any]]) -> None:
        lines = []
        for row in rows[:12]:
            role = str(row.get("role") or "?")
            state = str(row.get("state") or "idle")
            detail = str(row.get("detail") or "")
            mark = "*" if state not in {"idle", "ready", "local-only"} else " "
            lines.append(f"{mark} {role:10s} {state:12s} {detail}"[:70])
        self._set_text(self.agents_strip, "\n".join(lines) or "no agents reported")

    def _render_incidents(self, incidents: list[dict[str, Any]]) -> None:
        lines: list[str] = []
        if not incidents:
            lines.append("no incidents recorded - the app has not reported an error")
        for event in reversed(incidents[-40:]):
            mark = SEVERITY_MARK.get(str(event.get("severity")), "  ")
            lines.append(
                f"{mark}#{event.get('id')} {event.get('ts', '')} "
                f"[{event.get('kind', '')}] {event.get('where', '')}"
            )
            lines.append(f"    {event.get('message', '')}")
            repair = str(event.get("repair") or "")
            if repair:
                lines.append(f"    -> {repair}")
        self._set_text(self.text, "\n".join(lines))
        self._report = "\n".join(lines)
        self.hint.set(f"{len(incidents)} incident(s) in the log")

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    # ------------------------------------------------------------ actions
    def _diagnose(self) -> None:
        if self._on_diagnose is None:
            self.hint.set("no engine attached")
            return
        scope = self.scope_pick.get() or "all"
        agent = self.agent_pick.get() or "doctor"
        try:
            result = self._on_diagnose(scope, agent)
        except Exception as exc:  # noqa: BLE001 - a pane must never kill the app
            self.hint.set(f"diagnose failed: {type(exc).__name__}: {exc}")
            return
        self._render_findings(result, scope, agent)

    def _render_findings(self, result: dict[str, Any], scope: str, agent: str) -> None:
        if not isinstance(result, dict):
            self.hint.set("diagnose returned nothing")
            return
        if result.get("error"):
            self._set_text(self.text, f"diagnose failed: {result.get('error')}")
            self.hint.set("diagnose failed")
            return
        findings = result.get("findings") or []
        lines = [
            f"diagnosis by {agent} over '{scope}' in {result.get('ms', '?')} ms",
            f"{len(findings)} finding(s)",
            "",
        ]
        for item in findings:
            mark = SEVERITY_MARK.get(str(item.get("severity")), "  ")
            lines.append(f"{mark}{item.get('check', '?')}  {item.get('title', '')}")
            detail = str(item.get("detail") or "")
            if detail:
                lines.append(f"    {detail}")
            evidence = str(item.get("evidence") or "")
            if evidence:
                lines.append(f"    evidence: {evidence[:200]}")
            repair = str(item.get("repair") or "")
            lines.append(
                f"    repair: {repair}" if repair
                else "    repair: none safe - needs a person"
            )
        if not findings:
            lines.append("nothing found in this scope")
        body = "\n".join(lines)
        self._set_text(self.text, body)
        self._report = body
        self.hint.set(f"{len(findings)} finding(s) - nothing was changed")

    def _repair(self) -> None:
        if self._on_repair is None:
            self.hint.set("no engine attached")
            return
        picked = self.repair_list.curselection()
        if not picked:
            self.hint.set("pick a repair in the list first")
            return
        entry = self._repairs[int(picked[0])] if int(picked[0]) < len(self._repairs) else {}
        repair_id = str(entry.get("id") or "")
        if not repair_id:
            self.hint.set("that row has no repair id")
            return
        dry = bool(self.dry_run.get())
        try:
            result = self._on_repair(repair_id, dry)
        except Exception as exc:  # noqa: BLE001
            self.hint.set(f"repair failed: {type(exc).__name__}: {exc}")
            return
        self._render_repair(repair_id, result, dry)

    def _render_repair(self, repair_id: str, result: dict[str, Any], dry: bool) -> None:
        if not isinstance(result, dict):
            self.hint.set("repair returned nothing")
            return
        if result.get("error"):
            self._set_text(self.text, f"{repair_id}: {result.get('error')}")
            self.hint.set(f"{repair_id} refused")
            return
        detail = result.get("result") or {}
        lines = [
            f"{repair_id} {'dry run' if dry else 'applied'}",
            f"title:      {detail.get('title', '')}",
            f"detail:     {detail.get('detail', '')}",
            f"applied:    {detail.get('applied', False)}",
            f"verified:   {detail.get('verified', False)}",
            f"rolled back:{detail.get('rolled_back', False)}",
        ]
        for line in detail.get("evidence") or []:
            lines.append(f"  evidence: {line}")
        if not dry and not detail.get("verified"):
            lines.append("")
            lines.append("VERIFICATION FAILED - the change was rolled back.")
        body = "\n".join(lines)
        self._set_text(self.text, body)
        self._report = body
        self.hint.set(
            f"{repair_id} verified" if detail.get("verified")
            else f"{repair_id} not verified"
        )

    def _fix_top(self) -> None:
        if self._on_fix_top is None:
            self.hint.set("no engine attached")
            return
        try:
            result = self._on_fix_top()
        except Exception as exc:  # noqa: BLE001 - a pane must never kill the app
            self.hint.set(f"fix now failed: {type(exc).__name__}: {exc}")
            return
        entry = (result or {}).get("result") or {}
        self._render_repair(str(entry.get("id") or "fix now"), result, False)

    def _push_watch(self) -> None:
        if self._on_watch is None:
            return
        try:
            self._on_watch(bool(self.watch.get()))
        except Exception as exc:  # noqa: BLE001
            self.hint.set(f"watch setting not saved: {type(exc).__name__}")

    def _escalate(self) -> None:
        """S3.5: propose what is on screen to the harness, behind approval."""
        if self._on_escalate is None:
            self.hint.set("no engine attached")
            return
        question = " ".join((self._report or "").split())
        if not question:
            self.hint.set("run a diagnosis first - there is nothing to escalate")
            return
        try:
            result = self._on_escalate(question[:400])
        except Exception as exc:  # noqa: BLE001
            self.hint.set(f"escalation failed: {type(exc).__name__}: {exc}")
            return
        if isinstance(result, dict) and result.get("ok"):
            self.hint.set(
                f"proposed to the harness as approval #{result.get('approval')} - "
                "answer it in the approval row"
            )
        else:
            self.hint.set(f"not escalated: {(result or {}).get('error', 'unknown')}")

    def _verify(self) -> None:
        if self._on_verify is None:
            self.hint.set("no engine attached")
            return
        try:
            result = self._on_verify()
        except Exception as exc:  # noqa: BLE001
            self.hint.set(f"verify failed: {type(exc).__name__}: {exc}")
            return
        if not isinstance(result, dict):
            self.hint.set("verify returned nothing")
            return
        lines = [
            "verifier verdict: " + ("PASS" if result.get("ok") else "FAIL"),
            str(result.get("detail") or ""),
        ]
        for line in result.get("evidence") or []:
            lines.append(f"  {line}")
        body = "\n".join(lines)
        self._set_text(self.text, body)
        self._report = body
        self.hint.set("verifier " + ("passed" if result.get("ok") else "failed"))

    def _push_settings(self) -> None:
        if self._on_settings is None:
            return
        try:
            self._on_settings(bool(self.enabled.get()), bool(self.kill_switch.get()))
        except Exception as exc:  # noqa: BLE001
            self.hint.set(f"settings not saved: {type(exc).__name__}")

    def _copy(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(self._report or "nothing to copy")
            self.hint.set("report copied to the clipboard")
        except tk.TclError as exc:
            self.hint.set(f"clipboard unavailable: {exc}")

    def _save(self) -> None:
        from pathlib import Path

        out = Path(__file__).resolve().parent / "out"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "doctor_report.txt"
        try:
            path.write_text(self._report or "nothing to save", encoding="utf-8")
            self.hint.set(f"saved to {path.name}")
        except OSError as exc:
            self.hint.set(f"could not save: {exc}")
