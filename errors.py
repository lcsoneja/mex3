"""S1.1: the app's error memory - structured incidents, not a single string.

Errors used to live in one `Session.error` string that the next event overwrote,
plus console lines nobody sees when the window is launched. This module keeps a
bounded in-memory ring buffer AND appends every incident to out/incidents.jsonl,
so a question like "what happened at 23:34?" has an answer after the fact.

The rule the owner asked for: incidents carry TEXT only. No audio is ever stored,
and nothing here writes a voice recording - the jsonl holds diagnostics, the turn
journal holds transcripts.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

SEVERITIES = ("info", "warning", "error", "fatal")
DEFAULT_PATH = Path(__file__).resolve().parent / "out" / "incidents.jsonl"


class ErrorBus:
    """Thread-safe incident store: ring buffer + durable jsonl append."""

    def __init__(self, path: Path | None = None, keep: int = 500) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self._events: deque[dict[str, Any]] = deque(maxlen=max(50, int(keep)))
        self._lock = threading.Lock()
        self._next_id = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def record(
        self,
        kind: str,
        message: str,
        severity: str = "error",
        where: str = "",
        context: dict[str, Any] | None = None,
        traceback_id: str = "",
    ) -> dict[str, Any]:
        """Record one incident and return it (the caller can hold the id)."""
        clean = " ".join(str(message).split())[:600]
        if severity not in SEVERITIES:
            severity = "error"
        with self._lock:
            event = {
                "id": self._next_id,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": round(time.time(), 3),
                "kind": str(kind or "app")[:24],
                "severity": severity,
                "where": str(where)[:80],
                "message": clean,
                "traceback_id": str(traceback_id)[:40],
                "context": _plain(context),
                "repair": "",
            }
            self._next_id += 1
            self._events.append(event)
            self._append(event)
        return dict(event)

    def annotate(self, incident_id: int, repair: str) -> None:
        """Attach the repair that followed this incident."""
        with self._lock:
            for event in self._events:
                if event["id"] == int(incident_id):
                    event["repair"] = str(repair)[:160]
                    self._append(event)
                    return

    def _append(self, event: dict[str, Any]) -> None:
        """Best effort: a full disk must never break a turn."""
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------- read
    def snapshot(self, limit: int = 60) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        return [dict(event) for event in events[-max(1, int(limit)):]][::-1]

    def count(self, kind: str | None = None, severity: str | None = None) -> int:
        with self._lock:
            events = list(self._events)
        if kind is not None:
            events = [event for event in events if event["kind"] == kind]
        if severity is not None:
            events = [event for event in events if event["severity"] == severity]
        return len(events)

    def report(self, limit: int = 40) -> str:
        """Copy/paste-able incident report (what the Doctor pane saves)."""
        rows = self.snapshot(limit)
        lines = [
            "SONJ incident report",
            f"generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{len(rows)} incident(s) shown, {self.count()} in memory",
            "",
        ]
        for event in rows:
            lines.append(
                f"[{event['ts']}] {event['severity'].upper():7s} {event['kind']:9s} "
                f"#{event['id']} {event['where'] or '-'}"
            )
            lines.append("    " + event["message"])
            if event.get("traceback_id"):
                lines.append("    traceback: " + str(event["traceback_id"]))
            if event.get("context"):
                lines.append("    context: " + json.dumps(event["context"], ensure_ascii=False)[:240])
            if event.get("repair"):
                lines.append("    repair: " + str(event["repair"]))
        return "\n".join(lines)


def _plain(value: Any) -> dict[str, Any]:
    """Only JSON-safe, bounded context reaches the incident store."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in list(value.items())[:12]:
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)[:40]] = (str(item)[:200] if isinstance(item, str) else item)
        else:
            out[str(key)[:40]] = str(item)[:160]
    return out


BUS = ErrorBus()
