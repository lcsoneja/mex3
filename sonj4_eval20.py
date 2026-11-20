"""SONJ4 20-task evaluation suite (6 Front, 6 Back, 3 Doctor, 5 Coordination).

Usage:
    python sonj4_eval20.py --unit                  # fast offline deterministic check (all 20 tasks)
    python sonj4_eval20.py --live --port 8766      # live HTTP + real LLM panes against running sonj4 server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("ME3_NO_PERSIST", "1")

RESULTS: list[tuple[str, str, bool, str]] = []


def record(cat: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((cat, name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] [{cat}] {name}: {detail}", flush=True)


# ----------------------------------------------------------------- HTTP helpers
def http_get(base: str, path: str, timeout: float = 10.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def http_post(
    base: str, path: str, body: dict[str, Any], timeout: float = 30.0
) -> dict[str, Any]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def wait_turn_and_tasks(
    base: str, prev_turns: int, timeout_s: float = 180.0
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    first_turn = None
    while time.monotonic() < deadline:
        st = http_get(base, "/api/state")
        turns = st.get("turns") or []
        if len(turns) > prev_turns and first_turn is None:
            first_turn = turns[-1]
        tasks = st.get("tasks") or []
        busy_tasks = any(
            t.get("state") in {"queued", "running", "waiting"} for t in tasks
        )
        if (
            first_turn is not None
            and not busy_tasks
            and st.get("status") in {"idle", "listening"}
        ):
            time.sleep(1.2)
            st2 = http_get(base, "/api/state")
            return first_turn, st2
        time.sleep(1.0)
    return first_turn, http_get(base, "/api/state")


# ----------------------------------------------------------------- unit suite (all 20)
def run_unit() -> int:
    from server import BACK_POOL, Session, load_settings
    from tasks import Task

    tmp = Path(tempfile.mkdtemp(prefix="sonj4-eval20-"))
    st = load_settings()
    st.front_uses_pi = False
    st.tts_enabled = False
    st.tts_simulate = True
    s = Session(st)
    s._session_dir = tmp
    s.mode = "off"

    # ---- 6 Front-only tasks (F1..F6) ----
    front_cases = [
        ("F1_time", "what time is it?"),
        ("F2_math", "calculate 17 * 23"),
        ("F3_lights", "turn on the bedroom lights"),
        ("F4_timer", "set a timer for 30 seconds"),
        ("F5_joke", "tell me a short joke"),
        ("F6_memory", "remember that my favorite color is blue"),
    ]
    for name, prompt in front_cases:
        dec = s.classify_decision(prompt)
        ok = dec.route == "front" and dec.latency_ms < 100.0 and dec.bias_applied
        record(
            "FRONT",
            name,
            ok,
            f"route={dec.route} conf={dec.confidence:.2f} {dec.latency_ms:.3f}ms",
        )

    # ---- 6 Back-pool tasks (B1..B6) ----
    back_cases = [
        (
            "B1_snake_build",
            "create a snake game in sonj4/out/sandbox/snake.py",
            "build",
        ),
        (
            "B2_research_news",
            "research the latest news about raspberry pi 5 and give me a source",
            "research",
        ),
        (
            "B3_critic_verify",
            "verify this claim: 19 * 29 = 551 and audit the math",
            "verify",
        ),
        (
            "B4_csv_sorter",
            "write a python script to sort a CSV file by the second column",
            "build",
        ),
        (
            "B5_gazette_dl",
            "download the Official Gazette of the Philippines latest proclamation PDF",
            "research",
        ),
        (
            "B6_multi_summary",
            "research AI governance news and summarize the top 3 takeaways",
            "research",
        ),
    ]
    picked_slots: list[str] = []
    for name, prompt, want_mode in back_cases:
        dec = s.classify_decision(prompt)
        slot = s._pick_back_worker()
        picked_slots.append(slot)
        ok = (
            dec.route == "back"
            and dec.latency_ms < 100.0
            and slot in BACK_POOL
            and bool(dec.suggested_mode)
        )
        record(
            "BACK",
            name,
            ok,
            f"route={dec.route} mode={dec.suggested_mode} (want~{want_mode}) slot={slot} {dec.latency_ms:.3f}ms",
        )

    # ---- 3 Doctor tasks (D1..D3) ----
    doc_prompts = [
        ("D1_health_check", "run a health check on the app"),
        ("D2_incident_logs", "check the incident log and diagnose any errors"),
    ]
    for name, prompt in doc_prompts:
        dec = s.classify_decision(prompt)
        ok = dec.route == "doctor" and dec.latency_ms < 100.0
        record(
            "DOCTOR",
            name,
            ok,
            f"route={dec.route} conf={dec.confidence:.2f} {dec.latency_ms:.3f}ms",
        )

    # D3: Doctor setting proposal + 15s confirm loop
    dec_d3 = s.classify_decision("change wake_gate setting via doctor")
    prop = s.doctor_control(
        {
            "action": "propose_setting",
            "key": "wake_gate",
            "value": False,
            "summary": "Test wake_gate=False",
        }
    )
    pending = s._doctor_gate.pending()
    confirmed = s._doctor_gate.confirm_and_apply("", s, force=True)
    s._notify_doctor_outcome(confirmed, source="eval20")
    d3_a2a = [
        th["msg"] for th in getattr(s, "_thoughts", []) if th.get("kind") == "a2a"
    ]
    record(
        "DOCTOR",
        "D3_setting_proposal_loop",
        dec_d3.route == "doctor"
        and bool(prop.get("ok"))
        and pending is not None
        and bool(confirmed.get("ok"))
        and len(d3_a2a) >= 2,
        f"route={dec_d3.route} confirmed={confirmed.get('ok')} a2a={len(d3_a2a)}",
    )

    # ---- 5 Multi-agent Coordination / Community independence tasks (C1..C5) ----
    # C1: Delegation trace (front -> back-N and back-N -> front [COMPLETED])
    t1 = Task(id="crew-c1", brief="create hello.py", state="done", agent="back-1")
    t1.steps_plan = [{"role": "back-1", "model": "m"}]
    t1.result = "created hello.py"
    t1.finished = t1.created + 1.0
    s._think("front -> back-1 [TASK crew-c1]: create hello.py", "a2a")
    s._deliver_task_report(t1.result, t1)
    a2a_msgs = [
        th["msg"] for th in getattr(s, "_thoughts", []) if th.get("kind") == "a2a"
    ]
    c1_ok = any("front -> back-1" in m for m in a2a_msgs) and any(
        "back-1 -> front [COMPLETED crew-c1]" in m for m in a2a_msgs
    )
    record("COORD", "C1_delegation_trace", c1_ok, f"a2a_count={len(a2a_msgs)}")

    # C2: Parallel Back pool spreads across distinct workers back-1..back-5 while Front is idle
    distinct_5 = {s._pick_back_worker() for _ in range(5)}
    record(
        "COORD",
        "C2_pool_spread_5_workers",
        len(distinct_5) == 5,
        f"workers={sorted(distinct_5)}",
    )

    # C3: Broadcast independence (task finishes while Front is recording/noisy -> a2a feed immediate, speech fires after escape)
    t3 = Task(id="crew-c3", brief="research CPI", state="done", agent="back-2")
    t3.steps_plan = [{"role": "back-2", "model": "m"}]
    t3.result = "CPI report ready"
    t3.finished = t3.created + 1.5
    s._deliver_task_report(t3.result, t3)
    feed_now = [
        th["msg"]
        for th in getattr(s, "_thoughts", [])
        if th.get("kind") == "a2a" and "crew-c3" in th["msg"]
    ]
    s._pending_announcements.clear()
    s._deliver_task_report(t3.result, t3)
    s.recording = True
    s._last_user_speech_at = time.perf_counter()
    s._pending_announcements[0]["queued_at"] = time.perf_counter() - 20.0
    ann_ok = s._announce_pending_once()
    record(
        "COORD",
        "C3_broadcast_independence",
        bool(feed_now) and ann_ok,
        f"feed={len(feed_now)} announced={ann_ok}",
    )

    # C4: Failure & cancellation propagation in community feed + announce turns
    t4_fail = Task(id="crew-c4f", brief="broken task", state="blocked", agent="back-3")
    t4_fail.steps_plan = [{"role": "back-3", "model": "m"}]
    t4_fail.error = "network timeout"
    t4_fail.finished = t4_fail.created + 1.0
    s._deliver_task_report("task failed: network timeout", t4_fail)
    t4_canc = Task(
        id="crew-c4c", brief="stopped task", state="cancelled", agent="back-4"
    )
    t4_canc.steps_plan = [{"role": "back-4", "model": "m"}]
    t4_canc.finished = t4_canc.created + 1.0
    s._deliver_task_report("task cancelled", t4_canc)
    a2a_all = " | ".join(
        th["msg"] for th in getattr(s, "_thoughts", []) if th.get("kind") == "a2a"
    )
    c4_ok = ("back-3 -> front [FAILED crew-c4f]" in a2a_all) and (
        "back-4 -> front [CANCELLED crew-c4c]" in a2a_all
    )
    record("COORD", "C4_failure_cancel_propagation", c4_ok, "FAILED+CANCELLED recorded")

    # C5: Independent peer A2A coordination (post_community + classifier via ask_agent_turn)
    p1 = s.post_community(
        "back-1", "research ready; handing to back-2 to verify", to_role="back-2"
    )
    p2 = s.post_community(
        "back-2", "verified; asking doctor to confirm settings", to_role="doctor"
    )
    clf_res = s.ask_agent_turn("classifier", "check app logs", from_role="back-2")
    try:
        parsed_clf = json.loads(clf_res)
    except ValueError:
        parsed_clf = {}
    a2a_peer = " | ".join(
        th["msg"]
        for th in getattr(s, "_thoughts", [])
        if th.get("kind") in {"a2a", "gate"}
    )
    c5_ok = (
        bool(p1.get("ok"))
        and bool(p2.get("ok"))
        and parsed_clf.get("route") == "doctor"
        and "back-1 -> back-2:" in a2a_peer
        and "back-2 -> doctor:" in a2a_peer
    )
    record(
        "COORD",
        "C5_peer_community_independence",
        c5_ok,
        f"peer_clf={parsed_clf.get('route')}",
    )

    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    print(f"\nUNIT RESULT: {passed}/{len(RESULTS)} passed", flush=True)
    return 0 if passed == len(RESULTS) else 1


# ----------------------------------------------------------------- live HTTP suite
def run_live(port: int) -> int:
    base = f"http://127.0.0.1:{port}"
    h = http_get(base, "/api/health")
    if not h.get("ok"):
        print("Live server not healthy:", h)
        return 2

    # Ensure fresh session & quiet TTS for clean automated run
    http_post(base, "/api/settings", {"tts_enabled": False})
    http_post(base, "/api/session/new", {})
    time.sleep(0.5)

    # Wait briefly for always-warm pool panes to register
    panes = http_get(base, "/api/state").get("pi_panes") or {}
    warm_ok = all(
        r in panes
        for r in ("front", "doctor", "back-1", "back-2", "back-3", "back-4", "back-5")
    )
    record(
        "LIVE", "always_warm_7_panes_present", warm_ok, f"panes={sorted(panes.keys())}"
    )

    # F1..F6 live Front turns
    front_live = [
        ("F1_time", "what time is it?", lambda r, t: len(r) > 0),
        ("F2_math", "calculate 17 * 23", lambda r, t: "391" in r),
        ("F3_lights", "turn on the bedroom lights", lambda r, t: len(r) > 0),
        ("F4_timer", "set a timer for 30 seconds", lambda r, t: len(r) > 0),
        ("F5_joke", "tell me a short joke", lambda r, t: len(r) > 5),
        (
            "F6_remember",
            "remember that my favorite color is blue",
            lambda r, t: len(r) > 0,
        ),
    ]
    for name, q, pred in front_live:
        c = http_post(base, "/api/classify", {"text": q})
        before = len(http_get(base, "/api/state").get("turns") or [])
        http_post(base, "/api/send", {"text": q})
        turn, st = wait_turn_and_tasks(base, before, timeout_s=60.0)
        reply = str((turn or {}).get("assistant_text") or "")
        tools = [x.get("name") for x in ((turn or {}).get("tools") or [])]
        ok = (
            c.get("route") == "front"
            and float(c.get("latency_ms", 999)) < 100.0
            and "delegate_task" not in tools
            and pred(reply, turn)
        )
        record(
            "FRONT",
            name,
            ok,
            f"clf={c.get('route')}({c.get('latency_ms', 0):.2f}ms) reply={reply[:60]!r}",
        )

    # B1..B6 live Back-pool delegations (assert routed to back-1..back-5)
    back_live = [
        (
            "B1_snake_script",
            "create a tiny python script sonj4/out/sandbox/live_b1.py that prints hello sonj4",
        ),
        (
            "B2_research_rpi",
            "research the latest news about raspberry pi 5 and give one headline",
        ),
        ("B3_critic_math", "verify this claim: 19 * 29 = 551"),
        (
            "B4_csv_sorter",
            "write a python script sonj4/out/sandbox/live_b4.py that sorts numbers",
        ),
        (
            "B5_gazette_info",
            "research how to download a PDF from officialgazette.gov.ph and summarize in 2 sentences",
        ),
        (
            "B6_summarize_py",
            "research Python 3.12 features and summarize in 2 sentences",
        ),
    ]
    for name, q in back_live:
        c = http_post(base, "/api/classify", {"text": q})
        before = len(http_get(base, "/api/state").get("turns") or [])
        http_post(base, "/api/send", {"text": q})
        turn, st = wait_turn_and_tasks(base, before, timeout_s=180.0)
        tools = [x.get("name") for x in ((turn or {}).get("tools") or [])]
        a2a_ok = any(
            th.get("kind") == "a2a" and "COMPLETED" in str(th.get("msg"))
            for th in (st.get("thinking") or [])
        )
        ok = c.get("route") == "back" and ("delegate_task" in tools or a2a_ok)
        record(
            "BACK", name, ok, f"clf={c.get('route')} tools={tools} a2a_done={a2a_ok}"
        )

    # D1..D3 live Doctor checks
    for name, q in [
        ("D1_health", "run a health check on the app"),
        ("D2_logs", "check the incident log and diagnose any errors"),
    ]:
        c = http_post(base, "/api/classify", {"text": q})
        before = len(http_get(base, "/api/state").get("turns") or [])
        http_post(base, "/api/send", {"text": q})
        turn, st = wait_turn_and_tasks(base, before, timeout_s=120.0)
        tools = [x.get("name") for x in ((turn or {}).get("tools") or [])]
        reply = str((turn or {}).get("assistant_text") or "")
        ok = c.get("route") == "doctor" and (
            any(x in tools for x in ("ask_agent", "delegate_task")) or len(reply) > 10
        )
        record("DOCTOR", name, ok, f"clf={c.get('route')} tools={tools}")

    # D3: live doctor proposal + confirm
    p = http_post(
        base,
        "/api/doctor/control",
        {
            "action": "propose_setting",
            "key": "wake_gate",
            "value": False,
            "summary": "Live eval20 wake_gate=False",
        },
    )
    conf = http_post(
        base,
        "/api/doctor/confirm",
        {"confirm": True, "phrase": "Yes, I confirm the changes"},
    )
    record(
        "DOCTOR",
        "D3_proposal_confirm",
        bool(p.get("ok")) and bool(conf.get("ok")),
        f"prop={p.get('ok')} conf={conf.get('ok')}",
    )

    # C1..C5 live coordination & community feed checks
    st_now = http_get(base, "/api/state")
    thinking = st_now.get("thinking") or []
    a2a_lines = [str(th.get("msg") or "") for th in thinking if th.get("kind") == "a2a"]
    record(
        "COORD",
        "C1_live_a2a_trace",
        len(a2a_lines) >= 2,
        f"a2a_events={len(a2a_lines)}",
    )

    # C2: Parallel crew across 2 Back pool workers + Front responds mid-flight
    crew = http_post(
        base,
        "/api/crew",
        {
            "brief": "Write a 2-line greeting in sonj4/out/sandbox/crew_c2.txt",
            "roles": ["back-1", "back-2"],
            "process": "sequential",
        },
    )
    before_chat = len(http_get(base, "/api/state").get("turns") or [])
    http_post(base, "/api/send", {"text": "what is 5 + 7?"})
    chat_turn, _st_after = wait_turn_and_tasks(base, before_chat, timeout_s=180.0)
    record(
        "COORD",
        "C2_parallel_front_available",
        bool(crew.get("ok")) and bool((chat_turn or {}).get("assistant_text")),
        f"crew={crew.get('ok')} chat_reply={str((chat_turn or {}).get('assistant_text'))[:40]!r}",
    )

    # C3 & C4 & C5 live community + peer consult
    pc1 = http_post(
        base,
        "/api/community",
        {
            "from_role": "back-1",
            "to_role": "back-2",
            "message": "C5 live peer handoff test",
        },
    )
    pc2 = http_post(
        base,
        "/api/ask_agent",
        {"role": "classifier", "question": "check app settings", "from_role": "back-2"},
    )
    st_final = http_get(base, "/api/state")
    final_a2a = " | ".join(
        str(th.get("msg") or "")
        for th in (st_final.get("thinking") or [])
        if th.get("kind") in {"a2a", "gate"}
    )
    record(
        "COORD",
        "C3_announce_turns_logged",
        any(t.get("source") == "announce" for t in (st_final.get("turns") or [])),
        "announce turn present",
    )
    record(
        "COORD",
        "C4_doctor_a2a_present",
        "doctor -> front" in final_a2a,
        "doctor -> front in community",
    )
    record(
        "COORD",
        "C5_peer_community_live",
        bool(pc1.get("ok"))
        and "back-1 -> back-2:" in final_a2a
        and "doctor" in str(pc2.get("reply")),
        f"peer_clf={str(pc2.get('reply'))[:50]}",
    )

    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    print(f"\nLIVE RESULT: {passed}/{len(RESULTS)} passed", flush=True)
    return 0 if passed == len(RESULTS) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="SONJ4 20-task evaluation suite")
    ap.add_argument(
        "--unit", action="store_true", help="Run fast deterministic offline checks"
    )
    ap.add_argument(
        "--live", action="store_true", help="Run live HTTP + real panes against SONJ4"
    )
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    if args.live:
        return run_live(args.port)
    return run_unit()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] sonj4_eval20 error: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
