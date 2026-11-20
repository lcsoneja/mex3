"""Sample OpenRouter latency for the app's exact request shape.

Run: python me3_laptop/openrouter_latency.py --model z-ai/glm-5.3-flash --calls 5

Prints, per call, the wall-clock latency and which upstream provider answered -
OpenRouter routes the same model to different providers with very different
speeds, which is the difference between a 2 s reply and a 25 s reply.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, cast

import requests

from llm import OPENROUTER_URL, find_openrouter_key
from toolbox import ToolBox

SYSTEM = (
    "You are a voice assistant running on a small device. Always call a tool when one "
    "can answer the question. Your replies are spoken aloud, so keep them short."
)
QUESTION = "what time is it right now?"


def one_call(
    model: str, api_key: str, provider_order: list[str] | None = None
) -> tuple[float, str, str]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION},
        ],
        "tools": ToolBox().schemas(),
        "temperature": 0.4,
    }
    if provider_order:
        body["provider"] = {"order": provider_order, "allow_fallbacks": True}

    started = time.perf_counter()
    response = requests.post(
        OPENROUTER_URL,
        json=cast("dict[str, Any]", body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    provider = str(payload.get("provider") or "?")
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    calls = [c.get("function", {}).get("name") for c in message.get("tool_calls") or []]
    return elapsed, provider, ", ".join(str(c) for c in calls) or "no tool"


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenRouter latency sampler")
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument(
        "--order", default="", help="comma-separated provider preference"
    )
    args = parser.parse_args()

    api_key = find_openrouter_key()
    if not api_key:
        print("no OpenRouter key found")
        return 1
    order = [p.strip() for p in args.order.split(",") if p.strip()] or None

    print(f"model: {args.model}   calls: {args.calls}   order: {order or 'automatic'}")
    timings: list[float] = []
    for index in range(1, args.calls + 1):
        try:
            elapsed, provider, tools = one_call(args.model, api_key, order)
        except requests.RequestException as exc:
            print(f"  call {index}: FAILED {str(exc)[:80]}")
            continue
        timings.append(elapsed)
        print(f"  call {index}: {elapsed:6.2f}s  provider={provider:14s} tools={tools}")

    if timings:
        print()
        print(
            f"min {min(timings):.2f}s   median {statistics.median(timings):.2f}s   "
            f"max {max(timings):.2f}s   (n={len(timings)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
