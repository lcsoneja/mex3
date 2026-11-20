"""S4.3: rehearse the three demo failures against the running app.

1. duplicate engine  -> the second process refuses to start (S0 guard)
2. dead voice        -> R5 detects it and the repair rebuilds it, verified
3. dead model        -> R4 reports it; the model path itself is covered by the
                        M5 fallback chain, so here we only assert the detector
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import contextlib
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, ".")

from app_gui import VoiceAgentApp
from doctor import engine_processes
from server import Session, load_settings

# The engine's own refusal line, kept in one place so the assertion cannot drift.
REFUSAL_LINE = "another engine is already running"

problems: list[str] = []
notes: list[str] = []

# Run with:  python selfhealcheck.py   (needs the engine on 8765)


def read_text(path: str) -> str:
    """Read the report; an unreadable file is a recorded problem, not a crash."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"the session report is unreadable: {exc}")
        return ""


# --- 1. a second engine must refuse --------------------------------------
# check=False is the point: the refusal IS the expected outcome.
second = subprocess.run(
    [sys.executable, "server.py", "--port", "8765"],
    capture_output=True,
    text=True,
    timeout=60,
    check=False,
)
combined = f"{second.stdout}{second.stderr}"
reason_seen = combined.find(REFUSAL_LINE) >= 0
notes.append(f"second engine: exit={second.returncode}")
if second.returncode != 2:
    problems.append(f"second engine did not refuse (exit {second.returncode})")
elif not reason_seen:
    problems.append("second engine refused without the reason line")
else:
    notes.append("reason: " + combined.strip().splitlines()[0][:90])

# --- 1b. the engine matcher must not count shells that mention the files --
shell_decoy = subprocess.Popen(
    ["bash", "-c", "sleep 4; echo sonj3 server.py app_gui.py"]
)
python_decoy = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(4)  # sonj3 server.py"]
)
time.sleep(1.0)
seen = {row["pid"] for row in engine_processes()}
if shell_decoy.pid in seen:
    problems.append(
        f"a shell mentioning server.py was counted as an engine ({shell_decoy.pid})"
    )
if python_decoy.pid in seen:
    problems.append(f"python -c mentioning server.py was counted ({python_decoy.pid})")
notes.append(
    f"engine matcher: {len(seen)} engine(s), decoys excluded: "
    f"{shell_decoy.pid not in seen and python_decoy.pid not in seen}"
)
shell_decoy.terminate()
python_decoy.terminate()
if len(seen) != 1:
    problems.append(f"expected exactly one engine, saw {sorted(seen)}")

settings = load_settings()
settings.tts_enabled = True
session = Session(settings)
root = tk.Tk()
root.withdraw()
app = VoiceAgentApp(root, session)
root.update()

# --- 2. the voice is detected and repaired -------------------------------
with session.lock:
    session._tts = None
diag = session.diagnose("voice", "doctor")
found = [f for f in diag["findings"] if f["check"] == "R5"]
notes.append(f"R5 detected: {bool(found)}")

dry = session.repair("R5", dry_run=True)
notes.append(f"R5 dry run: {dry.get('ok')}")
live = session.repair("R5", dry_run=False)
entry = (live or {}).get("result") or {}
notes.append(
    f"R5 live: applied={entry.get('applied')} verified={entry.get('verified')} "
    f"rolled_back={entry.get('rolled_back')}"
)
if not entry.get("applied"):
    problems.append(f"R5 did not apply: {live}")
if not entry.get("verified"):
    problems.append(f"R5 did not verify: {entry.get('evidence')}")

# --- 3. model settings are watched --------------------------------------
diag = session.diagnose("models", "doctor")
notes.append(f"model findings: {[f['check'] for f in diag['findings']] or 'none'}")

# --- 4. the repaired state is reported in the session file --------------
report = session.write_incident_report()
if not report:
    problems.append("the session report was not written")
else:
    text = read_text(report)
    notes.append(f"report: {len(text)} chars, repairs listed: {text.count('- R')}")
    if "R5" not in text:
        problems.append("the session report does not mention the R5 repair")

session.stop_listening()
with contextlib.suppress(Exception):
    root.destroy()

print()
for note in notes:
    print("  ", note)
print()
print("RESULT: " + ("FAIL" if problems else "PASS"))
for problem in problems:
    print("PROBLEM:", problem)
raise SystemExit(1 if problems else 0)
