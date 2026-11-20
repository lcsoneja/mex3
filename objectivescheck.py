"""Fails when OBJECTIVES.md is out of step with REQUESTS.md.

The objectives document is generated, never hand-written, so it cannot drift from the
ledger - and the ledger cannot mark anything verified without evidence.

Run:  python me3_laptop/objectivescheck.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROW = re.compile(r"^\|\s*(R\d+)\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|")


def main() -> int:
    ledger = (ROOT / "REQUESTS.md").read_text(encoding="utf-8")
    objectives = (ROOT / "OBJECTIVES.md").read_text(encoding="utf-8")
    ids = [m.group(1) for line in ledger.splitlines() if (m := ROW.match(line))]
    missing = [rid for rid in ids if rid not in objectives]
    print(f"ledger rows: {len(ids)}   objectives doc present: {bool(objectives.strip())}")
    if missing:
        print("  MISSING from OBJECTIVES.md:", ", ".join(missing))
    ledger_ok = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "requestcheck.py")],
        capture_output=True, text=True, check=False,
    ).returncode == 0
    print("  ledger consistent:", ledger_ok)
    ok = not missing and ledger_ok
    print("objectives document is in step" if ok else "objectives document is OUT OF STEP")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
