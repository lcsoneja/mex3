"""Measure the real turn path, with the latency fast paths on and off.

This drives the same Session the GUI uses, so it measures the router, the pre-rendered
acknowledgements and the tool-result fast path exactly as a spoken turn hits them.

Two first-audio numbers are reported because they answer different questions:

  first  - when the acknowledgement the user hears actually sounded
  answer - when the real reply's first audio sounded (the baseline's only first audio,
           and therefore the column that makes the A/B comparable)

Both are stamped at the audio device, not when a function returned: an earlier version
measured a thread start and over-reported the win.

    python me3_laptop/turn_bench.py --label before --no-fast
    python me3_laptop/turn_bench.py --label after

Real audio is the default. --simulate is for machines with no audio device; it renders
replies and clips as silence, which is why the mode is recorded in the report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from console import use_utf8_stdout
from server import Session, load_settings

UTTERANCES = [
    "what time is it",
    "turn on the bedroom light at 40 percent",
    "set a timer for 30 seconds",
    "turn the kitchen light off",
    "what is the weather in Manila",
    "dim the bedroom light by 20 percent",
]


def stats(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return f"{p50:7.0f} / {p95:7.0f}"


def wait_ready(session: Session, settings, timeout_s: float) -> None:
    """Wait for the warm-up to finish instead of sleeping a fixed amount.

    Polling real readiness keeps the benchmark honest on a slow machine (the router
    model can take ~11 s to page in) and stops it from measuring a cold start.
    """
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        fillers_ready = True
        if settings.filler_enabled:
            try:
                bank = session.fillers()
                fillers_ready = bank.ready("okay") and bank.ready("hmm")
            except Exception:  # noqa: BLE001 - readiness is best effort
                fillers_ready = False
        router_ready = True
        if settings.router_enabled:
            try:
                router_ready = session.router().available()
            except Exception:  # noqa: BLE001
                router_ready = False
        if fillers_ready and router_ready:
            print("warm-up complete")
            return
        time.sleep(0.5)
    print("warm-up timed out - measuring anyway (numbers include a cold start)")


def main() -> int:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--turns", type=int, default=len(UTTERANCES))
    parser.add_argument("--no-fast", action="store_true",
                        help="disable router, fillers and tool-result fast path")
    parser.add_argument("--simulate", action="store_true",
                        help="no audio device: replies and clips are silence")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    settings = load_settings()
    settings.tts_simulate = bool(args.simulate)
    if args.no_fast:
        settings.router_enabled = False
        settings.filler_enabled = False
        settings.fast_tool_replies = False
    else:
        settings.router_enabled = True
        settings.filler_enabled = True
        settings.fast_tool_replies = True

    mode = "simulated" if args.simulate else "real"
    session = Session(settings)
    session.warm_extras()
    wait_ready(session, settings, timeout_s=45.0)

    print(f"== turn bench [{args.label}]  fast paths: {'off' if args.no_fast else 'on'}  audio: {mode}")
    started = time.perf_counter()
    for text in UTTERANCES[: args.turns]:
        session.submit_text(text)
        deadline = time.perf_counter() + 75.0
        while time.perf_counter() < deadline:
            time.sleep(0.2)
            turns = session.state()["turns"]
            if turns and turns[-1]["assistant_text"] and turns[-1]["user_text"] == text.strip():
                break

    turns = session.state()["turns"]
    wall = time.perf_counter() - started
    rows = [t for t in turns if t["user_text"]]
    print(f"turns recorded: {len(rows)}   wall: {wall:.1f} s")
    print(f"{'utterance':42s} {'asr':>6s} {'llm':>6s} {'tts':>6s} {'total':>6s} {'first':>6s} {'answer':>6s}")
    for turn in rows:
        timings = turn["timings"]
        print(f"{turn['user_text'][:40]:42s} {timings['asr_ms']:6.0f} {timings['llm_ms']:6.0f} "
              f"{timings['tts_ms']:6.0f} {timings['total_ms']:6.0f} "
              f"{timings.get('filler_ms', 0.0):6.0f} {timings.get('reply_audio_ms', 0.0):6.0f}")

    total = [t["timings"]["total_ms"] for t in rows]
    llm = [t["timings"]["llm_ms"] for t in rows]
    asr = [t["timings"]["asr_ms"] for t in rows]
    heard = [v for v in (t["timings"].get("filler_ms", 0.0) for t in rows) if v > 0]
    answered = [v for v in (t["timings"].get("reply_audio_ms", 0.0) for t in rows) if v > 0]
    print(f"\n  acknowledgement first audio p50/p95: {stats(heard)} ms   (n={len(heard)}/{len(rows)})")
    print(f"  real answer first audio  p50/p95: {stats(answered)} ms   (n={len(answered)}/{len(rows)})")
    print(f"  asr      p50/p95: {stats(asr)} ms")
    print(f"  llm      p50/p95: {stats(llm)} ms")
    print(f"  total    p50/p95: {stats(total)} ms")
    bank = session.fillers() if settings.filler_enabled else None
    if bank is not None:
        print(f"  fillers  clips: {sum(1 for k in bank.phrases if bank.ready(k))}"
              f"   device start lag: {bank.started_ms:.1f} ms")

    report = Path(__file__).resolve().parent / (args.report or f"out/turn_bench_{args.label}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "{\n"
        f'  "label": "{args.label}",\n'
        f'  "audio": "{mode}",\n'
        f'  "fast_paths": {str(not args.no_fast).lower()},\n'
        f'  "turns": {len(rows)},\n'
        f'  "wall_s": {wall:.1f},\n'
        f'  "ack_first_audio_p50_p95": "{stats(heard)}",\n'
        f'  "answer_first_audio_p50_p95": "{stats(answered)}",\n'
        f'  "asr_p50_p95": "{stats(asr)}",\n'
        f'  "llm_p50_p95": "{stats(llm)}",\n'
        f'  "total_p50_p95": "{stats(total)}"\n'
        "}\n",
        encoding="utf-8",
    )
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
