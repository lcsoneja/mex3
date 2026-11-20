"""Inspect the raw tool-calling exchange with the local Ollama model.

Prints every request payload and reply so the message shape can be checked
against what Ollama actually expects.

Usage:
    python me3_laptop/llmcheck.py --model llama3.2:3b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import json
import sys

from agent import DEFAULT_SYSTEM_PROMPT
from llm import LlmError, OllamaClient
from toolbox import ToolBox

PROMPT = "Turn on the bedroom light and set it to 40 percent."


def dump(label: str, payload: object) -> None:
    print(f"--- {label} ---")
    print(json.dumps(payload, indent=2, default=str)[:2000])


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama tool-call inspector")
    parser.add_argument("--model", default="llama3.2:3b-instruct-q4_K_M")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    llm = OllamaClient(args.model)
    toolbox = ToolBox()
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]

    try:
        first = llm.chat(messages, toolbox.schemas())
    except LlmError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"model: {llm.label}")
    print(f"round 1 content: {first.content!r}")
    print(f"round 1 tool_calls: {[(c.name, c.arguments) for c in first.tool_calls]}")
    if not first.tool_calls:
        print("Model did NOT call a tool - tool support or prompt is the problem.")
        return 1

    messages.append(llm.assistant_tool_message(first))
    for call in first.tool_calls:
        result = toolbox.call(call.name, call.arguments)
        messages.append(llm.tool_result_message(call, result.content))
        print(f"tool {call.name} -> ok={result.ok} result={result.content!r}")

    dump("messages sent for round 2", messages)

    try:
        second = llm.chat(messages, toolbox.schemas())
    except LlmError as exc:
        print(f"FAILED on round 2: {exc}")
        return 1

    print(f"round 2 content: {second.content!r}")
    print(f"round 2 tool_calls: {[(c.name, c.arguments) for c in second.tool_calls]}")
    print()
    if second.tool_calls:
        print("VERDICT: model wanted another tool round")
    elif second.content:
        print(
            "VERDICT: model produced a final answer (check whether it matches the tool result)"
        )
    else:
        print("VERDICT: empty final answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
