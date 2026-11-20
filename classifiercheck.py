"""TDD suite for SONJ4's non-LLM DecisionClassifier (<100ms, +10% Front bias).

Run: python classifiercheck.py
"""

from __future__ import annotations

import sys
import time

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> int:
    from classifier import (
        FRONT_BIAS,
        Decision,
        DecisionClassifier,
        DecisionRequest,
        apply_front_bias,
    )

    check("bias_constant_is_10pct", abs(FRONT_BIAS - 0.10) < 1e-9, f"bias={FRONT_BIAS}")

    # 1. Exact +10% Front bias math
    d_border = apply_front_bias(score_front=0.46, score_back=0.52, score_doctor=0.05)
    check(
        "front_bias_wins_borderline",
        d_border.route == "front" and d_border.bias_applied,
        f"route={d_border.route} sf={d_border.score_front} sb={d_border.score_back}",
    )

    d_clear_back = apply_front_bias(
        score_front=0.35, score_back=0.70, score_doctor=0.05
    )
    check(
        "clear_back_beats_10pct_bias",
        d_clear_back.route == "back",
        f"route={d_clear_back.route}",
    )

    d_abstain = apply_front_bias(score_front=0.20, score_back=0.30, score_doctor=0.10)
    check(
        "low_conf_abstains_to_front",
        d_abstain.route == "front",
        f"route={d_abstain.route} reason={d_abstain.reason_code}",
    )

    d_doc = apply_front_bias(score_front=0.20, score_back=0.15, score_doctor=0.80)
    check(
        "doctor_signal_routes_doctor", d_doc.route == "doctor", f"route={d_doc.route}"
    )

    # 2. Strict typed contract + 20-task routing accuracy
    clf = DecisionClassifier()
    front_prompts = [
        "what time is it?",
        "calculate 17 * 23",
        "turn on the bedroom lights",
        "set a timer for 30 seconds",
        "tell me a short joke",
        "remember that my favorite color is blue",
    ]
    back_prompts = [
        "create a snake game in sonj4/out/sandbox/snake.py",
        "research the latest news about raspberry pi 5 and give me a source",
        "verify this claim: the earth is flat and audit the evidence",
        "write a python script to sort a CSV file by the second column",
        "download the Official Gazette of the Philippines latest proclamation PDF",
        "research AI governance news and summarize the top 3 takeaways",
    ]
    doctor_prompts = [
        "run a health check on the app",
        "check the incident log and diagnose any errors",
        "change wake_gate setting or inspect app settings via doctor",
    ]
    coord_prompts = [
        "create a small calculator webpage and save it in sandbox",
        "build two small scripts in parallel: hello.py and bye.py",
        "research Manila weather trends and verify the source link",
        "check if any tasks failed and inspect system health logs",
        "review app architecture and verify config.json settings",
    ]

    for i, q in enumerate(front_prompts, 1):
        dec = clf.classify(DecisionRequest(text=q))
        check(
            f"route_F{i}",
            isinstance(dec, Decision)
            and dec.route == "front"
            and dec.latency_ms < 100.0,
            f"-> {dec.route} ({dec.reason_code}, {dec.latency_ms:.3f}ms)",
        )
    for i, q in enumerate(back_prompts, 1):
        dec = clf.classify(DecisionRequest(text=q))
        check(
            f"route_B{i}",
            dec.route == "back" and dec.latency_ms < 100.0,
            f"-> {dec.route} ({dec.reason_code}, mode={dec.suggested_mode}, {dec.latency_ms:.3f}ms)",
        )
    for i, q in enumerate(doctor_prompts, 1):
        dec = clf.classify(DecisionRequest(text=q))
        check(
            f"route_D{i}",
            dec.route == "doctor" and dec.latency_ms < 100.0,
            f"-> {dec.route} ({dec.reason_code}, {dec.latency_ms:.3f}ms)",
        )
    expected_coord = ["back", "back", "back", "doctor", "doctor"]
    for i, (q, want) in enumerate(zip(coord_prompts, expected_coord, strict=True), 1):
        dec = clf.classify(DecisionRequest(text=q))
        check(
            f"route_C{i}",
            dec.route == want and dec.latency_ms < 100.0,
            f"-> {dec.route} (want={want}, {dec.latency_ms:.3f}ms)",
        )

    # 3. Sub-100ms latency guarantee over 1,000 calls
    samples = (front_prompts + back_prompts + doctor_prompts) * 67
    latencies: list[float] = []
    for q in samples[:1000]:
        d = clf.classify(DecisionRequest(text=q))
        latencies.append(d.latency_ms)
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies))]
    worst = latencies[-1]
    check(
        "latency_1000_p95_under_100ms",
        p95 < 100.0 and worst < 100.0,
        f"p95={p95:.3f}ms max={worst:.3f}ms",
    )

    # 4. Hard deadline fallback test (simulated slow Stage 2 exceeding 90ms)
    def slow_stage2(_req: DecisionRequest):
        time.sleep(0.11)
        return (0.1, 0.9, 0.0, "slow_onnx")

    slow_clf = DecisionClassifier(stage2_fn=slow_stage2, deadline_ms=50.0)
    d_slow = slow_clf.classify(DecisionRequest(text="ambiguous borderline thing maybe"))
    check(
        "deadline_fallback_routes_front",
        d_slow.route == "front" and d_slow.fallback,
        f"route={d_slow.route} fallback={d_slow.fallback} ms={d_slow.latency_ms:.1f}",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] classifiercheck error: {type(exc).__name__}: {exc}")
        sys.exit(1)
