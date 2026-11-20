"""Regression check: delegated-task results must reach Front, persist, and speak.

Reproduces the live defect where a finished crew task appeared in the chat but
was never persisted and never spoken (Task has .agent, not .role, so the report
callback raised AttributeError after appending the in-memory turn).

Run:  python announcecheck.py   (no microphone/speaker needed; TTS is faked)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["ME3_NO_PERSIST"] = "1"

import server as engine
from server import Session, load_settings
from tasks import Task
from tts import TtsError

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


class FakeTts:
    backend = "fake"

    def __init__(self) -> None:
        self.said: list[str] = []
        self.voice_name = "fake-voice"

    def say(self, text: str) -> None:
        self.said.append(text)

    def wait(self, timeout_s: float = 30.0) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_rate(self, rate: int) -> None:
        return None


class BrokenTts(FakeTts):
    def say(self, text: str) -> None:
        raise TtsError("primary voice unavailable")


class FakeSapi(FakeTts):
    pass


def new_session(tmp: Path) -> Session:
    st = load_settings()
    st.front_uses_pi = False
    st.tts_enabled = True
    st.tts_simulate = False
    st.announce_task_completion = True
    s = Session(st)
    s._session_dir = tmp
    s.mode = "off"
    s.status = "idle"
    return s


def read_turns(tmp: Path) -> list[dict]:
    p = tmp / "messages.jsonl"
    if not p.is_file():
        return []
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sonj-announce-"))

    # 1. Completed crew task: report must not raise, must queue speech + persist.
    s = new_session(tmp)
    fake = FakeTts()
    s.tts = lambda: fake  # type: ignore[method-assign]
    crew = Task(
        id="crew-t1", brief="history of the Philippines", state="done", agent="back"
    )
    crew.steps_plan = [{"role": "researcher", "model": "m"}]
    crew.result = "Here is the history."
    crew.finished = crew.created + 5
    try:
        s._deliver_task_report(crew.result, crew)
        raised = ""
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    ann_role = s._pending_announcements[0]["role"] if s._pending_announcements else ""
    persisted = read_turns(tmp)
    report(
        "crew-report-no-raise",
        not raised,
        raised or "ok",
    )
    report(
        "crew-report-queues-speech",
        len(s._pending_announcements) == 1 and ann_role == "researcher",
        f"role={ann_role!r}",
    )
    report(
        "crew-report-persisted",
        any(t.get("source") == "task" for t in persisted),
        f"turns={len(persisted)}",
    )

    # 2. Deliver the queued announcement while the room is noisy (recording=True).
    s.mode = "continuous"
    s.status = "listening"
    s.recording = True
    s._last_user_speech_at = time.perf_counter()  # no pause yet
    s._pending_announcements[0]["queued_at"] = time.perf_counter() - 20.0
    delivered = s._announce_pending_once()
    announce_turns = [t for t in read_turns(tmp) if t.get("source") == "announce"]
    report(
        "noisy-room-still-speaks",
        bool(delivered) and len(fake.said) >= 1,
        f"said={len(fake.said)}",
    )
    report(
        "announce-turn-persisted", len(announce_turns) == 1, f"n={len(announce_turns)}"
    )
    report(
        "announce-turn-has-tts-ms",
        bool(announce_turns) and announce_turns[0]["timings"]["tts_ms"] > 0,
        "",
    )

    # 3. Failed plain task uses the task's agent as the role.
    s2 = new_session(tmp)
    s2.tts = lambda: FakeTts()  # type: ignore[method-assign]
    plain = Task(id="task-t2", brief="build a game", state="blocked", agent="back")
    plain.error = "model timeout"
    plain.finished = plain.created + 2
    try:
        s2._deliver_task_report(f"task failed: {plain.error}", plain)
        ok2 = (
            s2._pending_announcements
            and s2._pending_announcements[0]["role"] == "back"
            and s2._pending_announcements[0]["state"] == "failed"
        )
    except Exception as exc:  # noqa: BLE001
        ok2 = False
        print("  plain deliver raised:", exc)
    report("failed-plain-task-role", bool(ok2), "")

    # 4. Primary TTS failure falls back to SAPI and still posts the announce turn.
    s3 = new_session(tmp)
    s3.tts = lambda: BrokenTts()  # type: ignore[method-assign]
    saved_sapi = engine.SapiTts
    engine.SapiTts = FakeSapi  # type: ignore[misc]
    try:
        t3 = Task(id="crew-t3", brief="download gazette", state="done", agent="back")
        t3.steps_plan = [{"role": "researcher", "model": "m"}]
        t3.result = "done"
        t3.finished = t3.created + 1
        s3._deliver_task_report(t3.result, t3)
        s3._pending_announcements[0]["queued_at"] = time.perf_counter() - 20.0
        s3._announce_pending_once()
    finally:
        engine.SapiTts = saved_sapi  # type: ignore[misc]
    report(
        "tts-failure-still-posts-turn",
        any(t.get("source") == "announce" for t in read_turns(tmp)),
        "",
    )

    # 5. TTS disabled: the announce turn is still written to the conversation.
    s4 = new_session(tmp)
    s4.settings.tts_enabled = False
    t4 = Task(id="crew-t4", brief="summarize", state="done", agent="back")
    t4.steps_plan = [{"role": "summarizer", "model": "m"}]
    t4.result = "summary"
    t4.finished = t4.created + 1
    s4._deliver_task_report(t4.result, t4)
    s4._pending_announcements[0]["queued_at"] = time.perf_counter() - 20.0
    s4._announce_pending_once()
    report(
        "tts-off-still-posts-turn",
        any(t.get("source") == "announce" for t in read_turns(tmp)),
        "",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - a check reports failure, not a traceback
        print(f"[FAIL] announcecheck raised: {type(exc).__name__}: {exc}")
        sys.exit(1)
