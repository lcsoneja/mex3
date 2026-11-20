"""Live probe: S0-S4 in the real desktop app (headless build, no mainloop).

Checks the wiring the owner asked for, not the modules in isolation:
  * the Doctor pane exists in the window and shows the engine's own catalogue
  * an agents indicator is always on screen and names busy roles
  * a Tk callback crash becomes an incident, not a dead window
  * diagnose / verify answer through the pane's own callbacks
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import contextlib
import sys
import tkinter as tk

sys.path.insert(0, ".")

from app_gui import VoiceAgentApp
from server import Session, load_settings

problems: list[str] = []
notes: list[str] = []

settings = load_settings()
settings.tts_enabled = False  # a probe must not speak
session = Session(settings)
root = tk.Tk()
root.withdraw()
app = VoiceAgentApp(root, session)
root.update()

# --- 1. the pane exists, with the engine's catalogue ---------------------
if not hasattr(app, "doctor"):
    problems.append("no Doctor pane on the window")
else:
    rows = app.doctor.repair_list.size()
    notes.append(f"repair rows: {rows}")
    if rows < 8:
        problems.append(f"repair catalogue too small: {rows}")
    scopes = tuple(app.doctor.scope_pick.cget("values"))
    notes.append(f"scopes: {scopes}")
    if "engine" not in scopes or "voice" not in scopes:
        problems.append(f"scope picker missing engine/voice: {scopes}")
    agents = tuple(app.doctor.agent_pick.cget("values"))
    notes.append(f"investigators: {agents}")
    for who in ("doctor", "verifier", "harness"):
        if who not in agents:
            problems.append(f"investigator picker missing {who}: {agents}")

# --- 2. the view switch reaches the Doctor pane --------------------------
app.view_var.set("doctor")
app._show_view()
root.update()
if not app.panes_visible:
    problems.append("choosing the Doctor view did not reveal the right pane")

# --- 3. the always-on agents indicator -----------------------------------
line = app._agents_line(list(session.agents_active()))
notes.append(f"indicator: {line}")
if "front" not in line or "doctor" not in line:
    problems.append(f"agents strip does not name the roles: {line}")
app.agents_var.set(line)
if not app.agents_var.get():
    problems.append("agents indicator is empty")

# --- 4. diagnose through the pane's own callback -------------------------
result = app._doctor_diagnose("all", "doctor")
if not isinstance(result, dict) or "findings" not in result:
    problems.append(f"diagnose returned {type(result).__name__}: {str(result)[:120]}")
else:
    notes.append(f"diagnose: {len(result['findings'])} finding(s) in {result.get('ms')} ms")
    for item in result["findings"][:3]:
        notes.append(f"  {item.get('severity')}: {item.get('title')} -> {item.get('repair') or 'no repair'}")

# --- 5. the verifier on its own ------------------------------------------
verdict = app._doctor_verify()
if not isinstance(verdict, dict) or "ok" not in verdict:
    problems.append(f"verify returned {type(verdict).__name__}: {str(verdict)[:120]}")
else:
    notes.append(f"verifier: {'PASS' if verdict.get('ok') else 'FAIL'} - {str(verdict.get('detail'))[:80]}")

# --- 6. a Tk callback crash becomes an incident, window survives ---------
before = session.incidents.count()
try:
    raise RuntimeError("probe: simulated callback crash")
except RuntimeError as exc:
    app.root.report_callback_exception(type(exc), exc, exc.__traceback__)
after = session.incidents.count()
if after <= before:
    problems.append("a callback crash was not recorded as an incident")
fresh = session.incidents.snapshot(1)
if fresh:
    notes.append(f"incident: {fresh[0]['kind']}/{fresh[0]['severity']} {fresh[0]['message'][:60]}")

# --- 7. incidents reach the pane's renderer ------------------------------
app.doctor.refresh(
    list(session.incidents.snapshot(10)), list(session.agents_active()), {}, after)
app.doctor._render_incidents(list(session.incidents.snapshot(10)))
root.update()
notes.append(f"doctor hint: {app.doctor.hint.get()}")

# --- 8. S3.5: a code fix is proposed, never applied ----------------------
proposal = session.escalate("probe: the retry path swallows the provider error",
                            evidence="probe")
if not proposal.get("ok"):
    problems.append(f"escalate refused: {proposal}")
else:
    queued = [row for row in session.pending_approvals()
              if row["id"] == proposal["approval"]]
    if not queued:
        problems.append("escalation did not create a pending approval row")
    notes.append(f"escalation: approval #{proposal['approval']} pending, "
                 f"incident #{proposal['incident']}")
    session.respond_approval(str(proposal["approval"]), False)

# --- 9. S4.1: Fix now picks a repair from the last diagnosis -------------
fix = session.fix_top_finding(dry_run=True)
entry = (fix or {}).get("result") or {}
notes.append(f"fix now (dry): {entry.get('id') or fix.get('error')}")
if not fix.get("ok") and not fix.get("error"):
    problems.append(f"fix now returned neither a repair nor a refusal: {fix}")

# --- 10. S4.1: demo watch runs detect-only ------------------------------
session.update_settings({"doctor_watch": True})
watched = session.watch_tick()
notes.append(f"watch tick: {'ran' if watched else 'skipped (recent run or busy)'}")

# --- 11. S4.2: the session report is written, text only -----------------
report_path = session.write_incident_report()
if not report_path:
    problems.append("session report was not written")
else:
    body = report_path
    from pathlib import Path as _Path

    text = _Path(report_path).read_text(encoding="utf-8")
    notes.append(f"session report: {_Path(report_path).name} ({len(text)} chars)")
    for needle in ("# Session incidents", "## Repairs", "## Incident log"):
        if needle not in text:
            problems.append(f"session report missing {needle!r}")
    if any(token in text for token in (".wav", "audio bytes")):
        problems.append("session report mentions audio - the rule is text only")

session.stop_listening()
with contextlib.suppress(Exception):
    root.destroy()

print()
for note in notes:
    print("  ", note)
if problems:
    print()
    for problem in problems:
        print("PROBLEM:", problem)
    print()
    print(f"RESULT: FAIL ({len(problems)})")
    raise SystemExit(1)
print()
print("RESULT: PASS")
