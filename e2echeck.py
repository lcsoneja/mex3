"""End-to-end checks for the three requested scenarios (request 6).
All network I/O is wrapped: _urlopen/_decode turn every failure into EngineError.

Drives the real engine over HTTP exactly as the UI does: POST /api/send, then
wait for the turn to land and inspect the record (reply, tools, timings).
Reports pass/fail per scenario with the evidence, and writes out/e2e_report.md.
Failure modes: HTTP errors, dead engine, bad JSON, read failures -> EngineError.

Run with the engine up:  python e2echeck.py
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_name("out")
CASES = [
    {
        "name": "a-mongolia",
        "ask": "What is the capital of Mongolia",
        "want": ["ulaanbaatar"],
        "expect_tools": False,
    },
    {
        "name": "b-snake",
        "ask": "create a snake game",
        "want": [],
        "expect_tools": True,
    },
    {
        "name": "c-gazette",
        "ask": "fetch me the philippine gazette and open it",
        "want": [],
        "expect_tools": True,
    },
]


class EngineError(RuntimeError):
    """The engine could not be reached, or answered with something unusable."""


def _decode(method: str, path: str, r) -> dict:
    """Read + parse a response; every failure mode becomes an EngineError."""
    try:
        raw = r.read().decode("utf-8", "replace")
    except (UnicodeDecodeError, OSError) as exc:
        raise EngineError(f"{method} {path} -> read failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EngineError(
            f"{method} {path} -> invalid JSON ({exc}): {raw[:200]!r}"
        ) from exc


def _safe_body(exc: urllib.error.HTTPError) -> str:
    """Best-effort error body text; returns a marker instead of raising."""
    try:
        return exc.read().decode("utf-8", "replace")[:200]
    except (OSError, ValueError) as read_exc:
        return f"<body unreadable: {read_exc}>"


def _urlopen(method: str, path: str, req, timeout: float):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = _safe_body(exc)
        raise EngineError(
            f"{method} {path} -> HTTP {exc.code} {exc.reason}"
            + (f": {detail}" if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        raise EngineError(f"{method} {path} -> engine unreachable: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineError(f"{method} {path} -> {type(exc).__name__}: {exc}") from exc


def get(base: str, path: str, timeout: float = 10.0) -> dict:
    with _urlopen("GET", path, base + path, timeout) as r:
        return _decode("GET", path, r)


def post(base: str, path: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with _urlopen("POST", path, req, timeout) as r:
        return _decode("POST", path, r)


def wait_turn(base: str, before: int, timeout_s: float = 240.0) -> dict:
    """Wait until a new turn lands and the engine is idle again."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2)
        st = get(base, "/api/state")
        turns = st.get("turns") or []
        if len(turns) > before and st.get("status") == "idle":
            return turns[-1]
    return {}


def run_case(base: str, case: dict, timeout_s: float) -> dict:
    before = len(get(base, "/api/state").get("turns") or [])
    t0 = time.time()
    post(base, "/api/send", {"text": case["ask"]})
    turn = wait_turn(base, before, timeout_s)
    wall = time.time() - t0
    reply = str(turn.get("assistant_text") or "")
    tools = [str(t.get("name")) for t in (turn.get("tools") or [])]
    timings = turn.get("timings") or {}
    low = reply.lower()
    notes = []
    for word in case["want"]:
        notes.append((f"reply contains {word!r}", word in low))
    if case["expect_tools"]:
        notes.append(("a tool ran", bool(tools)))
    else:
        notes.append(("no tool needed", not tools))
    notes.append(("a reply came back", bool(reply.strip())))
    ok = all(passed for _, passed in notes)
    return {
        "name": case["name"],
        "ask": case["ask"],
        "ok": ok,
        "wall_s": round(wall, 1),
        "reply": reply[:400],
        "tools": tools,
        "timings": {k: timings.get(k) for k in
                    ("asr_ms", "llm_ms", "tts_ms", "first_token_ms", "total_ms")},
        "checks": notes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E2E scenario checks")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--only", default="", help="substring of one case name")
    args = parser.parse_args(argv)
    try:
        get(args.url, "/api/health", timeout=5)
    except EngineError as exc:
        print("engine not reachable:", exc)
        return 2
    cases = [c for c in CASES if not args.only or args.only in c["name"]]
    results = []
    for case in cases:
        print("=== " + case["name"] + ": " + case["ask"])
        try:
            res = run_case(args.url, case, args.timeout)
        except EngineError as exc:
            print("  engine error:", exc)
            results.append({
                "name": case["name"], "ask": case["ask"], "wall_s": 0.0,
                "reply": "", "tools": [], "timings": {},
                "checks": [("engine error: " + str(exc), False)], "ok": False,
            })
            continue
        results.append(res)
        print("  wall:", res["wall_s"], "s  tools:", res["tools"] or "none")
        for label, passed in res["checks"]:
            print(("  ok   " if passed else "  FAIL ") + label)
        print("  reply:", res["reply"][:220].replace(chr(10), " "))
    OUT.mkdir(exist_ok=True)
    lines = ["# E2E scenarios - " + time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    for res in results:
        lines.append("## " + res["name"] + (" PASS" if res["ok"] else " FAIL"))
        lines.append("- ask: " + res["ask"])
        lines.append("- wall: " + str(res["wall_s"]) + " s")
        lines.append("- tools: " + (", ".join(res["tools"]) or "none"))
        lines.append("- timings: " + json.dumps(res["timings"]))
        for label, passed in res["checks"]:
            lines.append("- " + ("PASS " if passed else "FAIL ") + label)
        lines.append("- reply: " + res["reply"].replace(chr(10), " ")[:300])
        lines.append("")
    (OUT / "e2e_report.md").write_text(chr(10).join(lines), encoding="utf-8")
    passed = sum(1 for r in results if r["ok"])
    print("report: out/e2e_report.md")
    print(f"RESULT: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())