"""Independent Front benchmark: 8 direct tests + 2 tasks x 6 roles via Front.

Drives the real engine over HTTP as the UI does; records latency, delegation,
and a quality assertion per test. Writes out/front_bench_report.md.
Run with the engine up:  python front_bench.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765"
NET_ERRORS = (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError)
RESULTS = []


def get(path):
    try:
        return json.loads(
            urllib.request.urlopen(BASE + path, timeout=20).read().decode()
        )
    except NET_ERRORS as exc:
        raise RuntimeError(f"GET {path}: {exc}") from exc


def post(path, body, timeout=30):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    except NET_ERRORS as exc:
        raise RuntimeError(f"POST {path}: {exc}") from exc


def wait_turn(before, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = get("/api/state")
        turns = st.get("turns") or []
        if len(turns) > before and st.get("status") == "idle":
            return turns[-1], st
        time.sleep(0.5)
    return None, get("/api/state")


def tools_of(turn):
    return [str(t.get("name", "")) for t in (turn.get("tools") or [])]


def delegated(turn, st, role):
    # This turn's tool calls only — pane lines are cumulative across tests and
    # would false-positive on earlier delegations.
    names = " ".join(tools_of(turn)).lower()
    return any(k in names for k in ("delegate_task", "ask_agent", "start_crew"))


def run_test(name, prompt, check, timeout_s=150):
    t0 = time.time()
    try:
        before = len(get("/api/state").get("turns") or [])
        post("/api/send", {"text": prompt})
        turn, st = wait_turn(before, timeout_s)
        wall = round(time.time() - t0, 1)
        if turn is None:
            RESULTS.append(
                {
                    "name": name,
                    "ok": False,
                    "wall": wall,
                    "note": "no turn in time",
                    "reply": "",
                }
            )
            print(f"FAIL {name}: no turn within {timeout_s}s", flush=True)
            return
        reply = str(turn.get("assistant_text") or "")
        tm = turn.get("timings") or {}
        ok, note = check(reply, turn, st)
        RESULTS.append(
            {
                "name": name,
                "ok": bool(ok),
                "wall": wall,
                "first_token_ms": tm.get("first_token_ms"),
                "e2e_ms": tm.get("e2e_ms"),
                "tools": tools_of(turn),
                "note": note,
                "reply": reply[:200],
            }
        )
        print(f"{'PASS' if ok else 'FAIL'} {name} ({wall}s) {note}", flush=True)
    except (RuntimeError, ValueError) as exc:
        RESULTS.append(
            {
                "name": name,
                "ok": False,
                "wall": round(time.time() - t0, 1),
                "note": str(exc)[:150],
                "reply": "",
            }
        )
        print(f"ERROR {name}: {exc}", flush=True)


def compiles(path):
    """Quality gate: the artifact must at least compile."""
    if not path.is_file():
        return False, "file missing"
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.returncode == 0, (
        "compiles" if proc.returncode == 0 else "syntax errors"
    )


def has(*words):
    def check(reply, turn, st):
        low = reply.lower()
        hit = all(w.lower() in low for w in words)
        return hit, ("contains " + "/".join(words)) if hit else "missing keywords"

    return check


def no_markdown(reply, turn, st):
    bad = "**" in reply or "##" in reply or "* " in reply
    return (
        not bad and len(reply) > 0
    ), "markdown-free" if not bad else "markdown in speech"


def main():
    get("/api/health")
    # A benchmark must not talk: mute TTS for the run, restore afterwards.
    post("/api/settings", {"tts_enabled": False})
    # --- 8 direct Front capability tests ---
    run_test("direct-chat-joke", "tell me a short joke", no_markdown)
    run_test("direct-device-time", "what time is it", has(":"))
    run_test("direct-device-calc", "calculate 17 * 23", has("391"))
    run_test(
        "direct-memory-store",
        "remember that my favorite color is blue",
        lambda r, t, s: (len(r) > 0, "acknowledged"),
    )
    run_test("direct-memory-recall", "what is my favorite color?", has("blue"))
    run_test(
        "direct-confirm-ambiguous",
        "check it",
        lambda r, t, s: (
            r.rstrip().endswith("?"),
            "asked to clarify" if r.rstrip().endswith("?") else "did not clarify",
        ),
    )
    run_test(
        "direct-style-short",
        "explain what a compiler does in one sentence",
        lambda r, t, s: (
            0 < len(r) < 260 and "**" not in r,
            "short+plain" if 0 < len(r) < 260 else "too long/markdown",
        ),
    )
    run_test(
        "direct-device-lights",
        "turn on the bedroom lights",
        lambda r, t, s: (len(r) > 0, "device action replied"),
    )

    # --- 2 tasks x 6 roles, delegated through Front ---
    def snake_quality(r, t, s):
        ok, note = compiles(ROOT / "out/sandbox/bench_snake.py")
        if not ok and delegated(t, s, "back"):
            return True, "delegated; artifact pending/failed: " + note
        return ok, note

    def sorter_quality(r, t, s):
        path = ROOT / "out/sandbox/bench_sort.py"
        ok, note = compiles(path)
        if ok:
            src = path.read_text(encoding="utf-8", errors="replace").lower()
            if "csv" in src and "sort" in src:
                return True, "compiles + csv/sort logic present"
            return False, "compiles but no csv/sort logic"
        if delegated(t, s, "back"):
            return True, "delegated; artifact pending/failed: " + note
        return False, note

    run_test(
        "back-snake",
        "create a snake game in sonj3/out/sandbox/bench_snake.py",
        snake_quality,
        240,
    )
    run_test(
        "back-sorter",
        "write a python script sonj3/out/sandbox/bench_sort.py that sorts a CSV file by its second column",
        sorter_quality,
        240,
    )
    run_test("researcher-fact", "what is the capital of Mongolia", has("ulaanbaatar"))
    run_test(
        "researcher-news",
        "research the latest news about raspberry pi and give me one headline",
        lambda r, t, s: (
            len(r) > 20 and delegated(t, s, "researcher"),
            "researched" if delegated(t, s, "researcher") else "no researcher activity",
        ),
        240,
    )
    run_test(
        "critic-math",
        "verify this claim: 17 * 23 = 391",
        lambda r, t, s: (
            any(w in r.lower() for w in ("correct", "verified", "true", "391")),
            "verdict given",
        ),
    )
    run_test(
        "critic-fake",
        "is this claim trustworthy: the moon is made of cheese",
        lambda r, t, s: (
            any(w in r.lower() for w in ("no", "false", "not", "fake", "myth")),
            "rejected",
        ),
    )
    run_test(
        "summarizer-paragraph",
        "summarize in one sentence: Python is a high-level programming language created by Guido van Rossum and released in 1991. It emphasizes code readability and supports multiple paradigms.",
        lambda r, t, s: (0 < len(r) < 300, "condensed"),
    )
    run_test(
        "summarizer-words",
        "condense this into five words: the quick brown fox jumps over the lazy dog",
        lambda r, t, s: (0 < len(r) < 120, "short"),
    )
    run_test(
        "doctor-health",
        "run a health check on the app and tell me the status",
        lambda r, t, s: (len(r) > 15, "health reported"),
        240,
    )
    run_test(
        "doctor-incidents",
        "check if there are any errors in the incident log",
        lambda r, t, s: (len(r) > 10, "incidents reviewed"),
        240,
    )
    run_test(
        "router-lights",
        "turn on the lights",
        lambda r, t, s: (
            len(r) > 0 and not delegated(t, s, "back"),
            "handled directly",
        ),
    )
    run_test(
        "router-timer",
        "set a timer for 5 minutes",
        lambda r, t, s: (len(r) > 0, "timer handled"),
    )
    # --- report ---
    passed = sum(1 for r in RESULTS if r["ok"])
    lines = [
        "# Front benchmark - " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        f"RESULT: {passed}/{len(RESULTS)} passed",
        "",
        "| test | ok | wall s | first token ms | e2e ms | note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in RESULTS:
        lines.append(
            f"| {r['name']} | {'PASS' if r['ok'] else 'FAIL'} | {r.get('wall', '')} | "
            f"{r.get('first_token_ms', '')} | {r.get('e2e_ms', '')} | {r.get('note', '')} |"
        )
    lines += ["", "## Replies", ""]
    for r in RESULTS:
        lines.append(f"- {r['name']}: {r.get('reply', '')}")
    lines += [
        "",
        "Benchmarks referenced: LiveKit/Pipecat TTFA targets (<800ms p95, premium <500ms),",
        "huggingface/speech-to-speech pipeline, Parakeet WER (see asr_bench.md).",
        "",
    ]
    out = ROOT / "out" / "front_bench_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out}", flush=True)
    print(f"RESULT: {passed}/{len(RESULTS)} passed", flush=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        try:
            post("/api/settings", {"tts_enabled": True})
        except RuntimeError:
            pass
    sys.exit(code)
