"""Run the standardized SONJ latency benchmark against a live engine.

The harness sends fixed text prompts to /api/send and reads the engine's own
per-turn timing stamps from /api/state. It does not emulate microphone audio.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out"
DEFAULT_URL = "http://127.0.0.1:8765"
WARMUP_PROMPT = "What time is it?"
MIN_STANDARD_TURNS = 20
FIRST_SOUND_TARGET_MS = 300.0
BARGE_IN_TARGET_MS = 75.0

# Fixed, voice-style commands spanning the deterministic device/tool families.
PROMPTS = (
    "What time is it in Manila?",
    "Calculate 17 times 23.",
    "Turn on the bedroom lights at 40 percent.",
    "Set the kitchen lights to off.",
    "Dim the living room lights by 20 percent.",
    "Set the bedroom temperature one degree cooler.",
    "Pause the music.",
    "Turn the volume down by 10 percent.",
    "Brighten the living room lights by 10 percent.",
    "What time is it?",
)

METRICS = {
    "acknowledgement_latency_ms": ("filler_ms", "First audible pre-rendered acknowledgement"),
    "first_response_ms": ("reply_audio_ms", "First audible audio of the actual answer"),
    "asr_ms": ("asr_ms", "Speech recognition stage (zero for typed API input)"),
    "llm_ms": ("llm_ms", "Language-model stage"),
    "tts_ms": ("tts_ms", "Text-to-speech rendering stage"),
    "turn_total_ms": ("total_ms", "Engine-recorded turn processing total"),
}


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile; p95 is the maximum when n is small."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    p50 = statistics.median(values) if values else None
    p95 = percentile(values, 0.95)
    return {
        "sample_size": len(values),
        "p50_ms": round(p50, 1) if p50 is not None else None,
        "p95_ms": round(p95, 1) if p95 is not None else None,
    }


def request_json(
    base_url: str,
    route: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("engine URL must use http:// or https:// and include a host")
    if parsed_url.query or parsed_url.fragment:
        raise ValueError("engine base URL must not include a query or fragment")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    connection_type = HTTPSConnection if parsed_url.scheme == "https" else HTTPConnection
    connection = connection_type(parsed_url.netloc, timeout=timeout_s)
    request_path = f"{parsed_url.path.rstrip('/')}{route}"
    headers = {"Content-Type": "application/json"} if body is not None else {}
    method = "POST" if body is not None else "GET"
    try:
        connection.request(method, request_path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        if response.status >= 400:
            detail = response_body.decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} returned HTTP {response.status}: {detail}")
        try:
            result = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{route} returned invalid JSON") from exc
    finally:
        connection.close()
    if not isinstance(result, dict):
        raise TypeError(f"{route} returned a non-object JSON response")
    return result


def turn_id(turn: dict[str, Any]) -> int:
    try:
        return int(turn.get("id", 0))
    except (TypeError, ValueError):
        return 0


def latest_turn_id(state: dict[str, Any]) -> int:
    turns = state.get("turns") or []
    return max((turn_id(turn) for turn in turns if isinstance(turn, dict)), default=0)


def wait_for_idle(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        health = request_json(base_url, "/api/health", timeout_s=min(timeout_s, 10.0))
        if health.get("ok") is True and health.get("status") == "idle":
            return health
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"engine did not become idle before timeout (last health={health!r})"
            )
        time.sleep(0.25)


def submit_and_wait(
    base_url: str,
    prompt: str,
    previous_id: int,
    timeout_s: float,
) -> dict[str, Any]:
    response = request_json(
        base_url, "/api/send", payload={"text": prompt}, timeout_s=timeout_s
    )
    if response.get("ok") is not True:
        raise RuntimeError(f"/api/send rejected prompt {prompt!r}: {response!r}")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = request_json(
            base_url, "/api/state", timeout_s=min(timeout_s, 10.0)
        )
        turns = state.get("turns") or []
        for turn in turns:
            if not isinstance(turn, dict) or turn_id(turn) <= previous_id:
                continue
            if str(turn.get("user_text") or "").strip() != prompt.strip():
                continue
            if str(turn.get("assistant_text") or "").strip():
                return turn
        time.sleep(0.25)
    raise TimeoutError(f"no completed turn for {prompt!r} within {timeout_s:.1f}s")


def successful_turn(turn: dict[str, Any]) -> bool:
    answer = str(turn.get("assistant_text") or "").strip()
    return bool(answer) and not answer.lower().startswith("(model error)")


def numeric_timing(turn: dict[str, Any], field: str) -> float | None:
    timings = turn.get("timings") or {}
    raw_value = timings.get(field)
    if not isinstance(raw_value, (int, float)):
        return None
    value = float(raw_value)
    return value if math.isfinite(value) and value > 0 else None


def stage_summaries(turns: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    answered = [turn for turn in turns if successful_turn(turn)]
    summaries: dict[str, dict[str, float | int | None]] = {}
    for name, (field, _description) in METRICS.items():
        values = [value for turn in answered if (value := numeric_timing(turn, field)) is not None]
        summaries[name] = summarize(values)

    first_sound: list[float] = []
    for turn in answered:
        sounds = [
            value
            for field in ("filler_ms", "reply_audio_ms")
            if (value := numeric_timing(turn, field)) is not None
        ]
        if sounds:
            first_sound.append(min(sounds))
    summaries["first_sound_ms"] = summarize(first_sound)
    return summaries


def classify_route(turn: dict[str, Any], configured_provider: str) -> tuple[str, str]:
    model = str(turn.get("model") or "")
    model_provider = model.split("/", 1)[0].lower() if "/" in model else ""
    provider = (
        model_provider
        if model_provider in {"ollama", "openrouter"}
        else configured_provider.lower()
    )
    if provider == "ollama":
        return "local", model or f"ollama/{turn.get('model', '')}"
    if provider == "openrouter":
        return "cloud", model or "openrouter/unknown"
    return "unknown", model


def route_summary(turns: list[dict[str, Any]], configured_provider: str) -> dict[str, Any]:
    answered = [turn for turn in turns if successful_turn(turn)]
    counts = {"local": 0, "cloud": 0, "unknown": 0}
    turn_routes: list[str] = []
    for turn in answered:
        category, model = classify_route(turn, configured_provider)
        counts[category] += 1
        turn_routes.append(model)
    total = len(answered)
    return {
        "answered_turns": total,
        "local_turns": counts["local"],
        "cloud_turns": counts["cloud"],
        "unknown_turns": counts["unknown"],
        "local_percent": round(100 * counts["local"] / total, 1) if total else None,
        "cloud_percent": round(100 * counts["cloud"] / total, 1) if total else None,
        "models_by_turn": turn_routes,
    }


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def target_status(stats: dict[str, Any], limit_ms: float) -> str:
    p95 = stats.get("p95_ms")
    if p95 is None:
        return "not measured"
    return "met" if p95 < limit_ms else "not met"


def build_markdown(report: dict[str, Any]) -> str:
    benchmark = report["benchmark"]
    stages = report["stage_stats"]
    rows = [
        "# SONJ standardized latency benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Engine: `{report['base_url']}`  ",
        f"Model route: `{report['configured_model_route']}`  ",
        f"Measured turns: {benchmark['measured_turns']} (standard minimum: {MIN_STANDARD_TURNS})  ",
        f"Run classification: **{benchmark['sample_status']}**  ",
        f"Wall time: {benchmark['wall_time_s']:.1f} s",
        "",
        "## Stage results",
        "",
        "All latency values are milliseconds. p50 is the median and p95 uses nearest-rank; each row reports its own valid sample count.",
        "",
        "| Metric | n | p50 | p95 | Target | p95 result |",
        "|---|---:|---:|---:|---|---|",
    ]
    rows.extend(
        [
            (
                f"| Acknowledgement latency | {stages['acknowledgement_latency_ms']['sample_size']} | "
                f"{format_ms(stages['acknowledgement_latency_ms']['p50_ms'])} | "
                f"{format_ms(stages['acknowledgement_latency_ms']['p95_ms'])} | — | — |"
            ),
            (
                f"| First response audio | {stages['first_response_ms']['sample_size']} | "
                f"{format_ms(stages['first_response_ms']['p50_ms'])} | "
                f"{format_ms(stages['first_response_ms']['p95_ms'])} | — | — |"
            ),
            (
                f"| First audible sound | {stages['first_sound_ms']['sample_size']} | "
                f"{format_ms(stages['first_sound_ms']['p50_ms'])} | "
                f"{format_ms(stages['first_sound_ms']['p95_ms'])} | <{FIRST_SOUND_TARGET_MS:.0f} ms | "
                f"{target_status(stages['first_sound_ms'], FIRST_SOUND_TARGET_MS)} |"
            ),
        ]
    )
    for metric in ("asr_ms", "llm_ms", "tts_ms", "turn_total_ms"):
        stats = stages[metric]
        description = METRICS[metric][1]
        rows.append(
            f"| {description} | {stats['sample_size']} | {format_ms(stats['p50_ms'])} | "
            f"{format_ms(stats['p95_ms'])} | — | — |"
        )

    barge = report["barge_in"]
    route = report["route_share"]
    rows.extend(
        [
            "",
            "## Barge-in and route share",
            "",
            f"- Barge-in detections observed during this run: {barge['observed_count']}",
            f"- Barge-in detection latency: **not measured** (n=0); the text-only `/api/send` path does not inject microphone audio while the agent is speaking. Target: <{BARGE_IN_TARGET_MS:.0f} ms.",
            f"- Answered locally: {route['local_turns']}/{route['answered_turns']} ({format_ms(route['local_percent'])}%).",
            f"- Answered through cloud model route: {route['cloud_turns']}/{route['answered_turns']} ({format_ms(route['cloud_percent'])}%).",
            f"- Unknown route: {route['unknown_turns']}/{route['answered_turns']}.",
            "",
            "## Conditions and limitations",
            "",
            f"- `/api/health` was checked before the run; one unscored warm-up turn is recorded separately: `{report['warmup']['prompt']}`.",
            "- The engine's health endpoint reports readiness only; the separate warm-up turn primes request/model paths when they were cold. It is excluded from measured-turn statistics.",
            "- Prompts are sent sequentially. They are typed API text, so VAD and live ASR are bypassed; `asr_ms` may have n=0. This measures the engine response path, not end-of-human-utterance-to-response latency.",
            "- Simulated device/tool state persists and is not reset by this runner. Start comparisons from the same default state (preferably a fresh engine) or restore and record state; the current state API does not expose a complete device-state snapshot.",
            "- Barge-in latency needs real or replayed microphone audio during playback and must be reported separately; the runner only records the engine's observed barge-in count delta.",
            "- Only successful answer turns are included in latency percentiles and local/cloud shares. Missing/zero stage stamps are not treated as zero-latency samples.",
            "",
            "## Fixed prompt set",
            "",
        ]
    )
    rows.extend(f"{index}. {prompt}" for index, prompt in enumerate(report["prompt_set"], start=1))
    rows.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON: `{report['output_json']}`",
            f"- Markdown: `{report['output_markdown']}`",
            "",
        ]
    )
    return "\n".join(rows)


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    base_url = args.url.rstrip("/")
    health = wait_for_idle(base_url, args.timeout)
    initial_state = request_json(base_url, "/api/state", timeout_s=min(args.timeout, 10.0))
    settings = initial_state.get("settings") or {}
    configured_provider = str(settings.get("provider") or "unknown")
    model_name = str(settings.get("model") or "unknown")
    configured_model_route = f"{configured_provider}/{model_name}"
    before_barge_ins = int(initial_state.get("barge_ins") or 0)

    warmup_prompt = WARMUP_PROMPT
    warmup_turn = submit_and_wait(
        base_url,
        warmup_prompt,
        latest_turn_id(initial_state),
        args.timeout,
    )
    warm_state = request_json(base_url, "/api/state", timeout_s=min(args.timeout, 10.0))
    baseline_barge_ins = int(warm_state.get("barge_ins") or before_barge_ins)
    previous_id = latest_turn_id(warm_state)

    measured: list[dict[str, Any]] = []
    start = time.monotonic()
    for index in range(args.turns):
        prompt = PROMPTS[index % len(PROMPTS)]
        turn = submit_and_wait(base_url, prompt, previous_id, args.timeout)
        measured.append({"sequence": index + 1, "prompt": prompt, "turn": turn})
        previous_id = turn_id(turn)
    wall_time_s = time.monotonic() - start

    final_state = request_json(base_url, "/api/state", timeout_s=min(args.timeout, 10.0))
    final_barge_ins = int(final_state.get("barge_ins") or baseline_barge_ins)
    turns = [item["turn"] for item in measured]
    successful_count = sum(successful_turn(turn) for turn in turns)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = OUT_DIR / f"bench_run_{stamp}.json"
    output_markdown = OUT_DIR / f"bench_run_{stamp}.md"
    suffix = 1
    while output_json.exists() or output_markdown.exists():
        suffix += 1
        output_json = OUT_DIR / f"bench_run_{stamp}_{suffix}.json"
        output_markdown = OUT_DIR / f"bench_run_{stamp}_{suffix}.md"

    stage_stats = stage_summaries(turns)
    report: dict[str, Any] = {
        "benchmark": {
            "name": "SONJ standardized latency benchmark",
            "measured_turns": len(turns),
            "successful_turns": successful_count,
            "failed_turns": len(turns) - successful_count,
            "standard_minimum_turns": MIN_STANDARD_TURNS,
            "sample_status": "standard" if len(turns) >= MIN_STANDARD_TURNS else "smoke test / below n>=20",
            "wall_time_s": round(wall_time_s, 2),
            "percentile_method": "p50 median; p95 nearest-rank",
            "timing_source": "/api/state turn records",
            "tool_state_reset_by_runner": False,
            "tool_state_note": "Reset/restore the same simulated device state before comparing runs; the runner does not reset it.",
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": base_url,
        "health_before_run": health,
        "configured_model_route": configured_model_route,
        "warmup": {
            "prompt": warmup_prompt,
            "excluded_from_statistics": True,
            "turn": warmup_turn,
            "note": "This unscored first request captures cold-start cost if the engine was not already warm.",
        },
        "prompt_set": list(PROMPTS),
        "stage_stats": stage_stats,
        "targets": {
            "first_sound_ms": {
                "target": f"<{FIRST_SOUND_TARGET_MS:.0f} ms",
                "comparison": "p95",
                "p95_status": target_status(stage_stats["first_sound_ms"], FIRST_SOUND_TARGET_MS),
            },
            "barge_in_detection_ms": {
                "target": f"<{BARGE_IN_TARGET_MS:.0f} ms",
                "measured_p50_ms": None,
                "measured_p95_ms": None,
                "sample_size": 0,
                "status": "not measured by text-only API runner",
            },
        },
        "barge_in": {
            "observed_count": max(0, final_barge_ins - baseline_barge_ins),
            "latency_sample_size": 0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        },
        "route_share": route_summary(turns, configured_provider),
        "turns": measured,
        "output_json": str(output_json),
        "output_markdown": str(output_markdown),
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_markdown.write_text(build_markdown(report), encoding="utf-8")
    return report, str(output_json), str(output_markdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standardized SONJ latency benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"engine base URL (default: {DEFAULT_URL})")
    parser.add_argument("--turns", type=int, default=MIN_STANDARD_TURNS, help="measured turns; standard runs require at least 20")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds allowed per turn and for engine readiness")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report to stdout")
    args = parser.parse_args()
    if args.turns <= 0:
        parser.error("--turns must be a positive integer")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        report, output_json, output_markdown = run_benchmark(args)
    except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as exc:
        print(f"benchrun: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON file: {output_json}", file=sys.stderr)
        print(f"Markdown summary: {output_markdown}", file=sys.stderr)
    else:
        print(f"model route: {report['configured_model_route']}")
        print(
            f"turns: {report['benchmark']['successful_turns']}/"
            f"{report['benchmark']['measured_turns']} successful; "
            f"{report['benchmark']['sample_status']}"
        )
        for name, stats in report["stage_stats"].items():
            print(
                f"{name}: p50={format_ms(stats['p50_ms'])} ms, "
                f"p95={format_ms(stats['p95_ms'])} ms, n={stats['sample_size']}"
            )
        print(f"JSON: {output_json}")
        print(f"Markdown: {output_markdown}")
    return 0 if report["benchmark"]["successful_turns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
