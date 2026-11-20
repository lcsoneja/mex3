"""Consistency check: every ask in REQUESTS.md is tracked and evidenced.

Fails when a row marked "verified" has no evidence, or when a status is not one of the
four allowed values. Open "asked" rows are reported as promises, not failures - the point
is that nothing can be called done without a check behind it.

Run:  python me3_laptop/requestcheck.py
"""

from __future__ import annotations

import re
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "REQUESTS.md"
ALLOWED = ("asked", "planned", "implemented", "verified")
ROW = re.compile(r"^\|\s*(R\d+)\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*$")


def main() -> int:
    if not LEDGER.exists():
        print(f"FAIL ledger missing: {LEDGER}")
        return 1
    rows, bad, counts = [], [], dict.fromkeys(ALLOWED, 0)
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        rid, ask, status, evidence = (m.group(i) for i in (1, 2, 3, 4))
        status = status.lower()
        rows.append((rid, ask, status, evidence))
        if status not in counts:
            bad.append(f"{rid}: unknown status {status!r}")
            continue
        counts[status] += 1
        if status == "verified" and not evidence:
            bad.append(f"{rid}: marked verified with no evidence")
    print("request ledger consistency")
    for rid, ask, status, evidence in rows:
        print(f"  [{status:12s}] {rid:4s} {ask[:58]:58s} {evidence[:52]}")
    print("  counts:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    if bad:
        print("  INCONSISTENCIES:")
        for item in bad:
            print("   -", item)
    print("all rows consistent" if not bad else f"{len(bad)} inconsistency(ies)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
