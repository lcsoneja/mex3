"""Validate tool-call adherence for any provider/model.

Run: python me3_laptop/agentcheck.py --provider openrouter --model nvidia/nemotron-3.5-lightning:free

The failure mode this guards against: a model calls a tool, then ignores the
result and tells the user it could not do the thing.
"""

from __future__ import annotations

import argparse
import sys

from agent import Agent
from console import use_utf8_stdout
from llm import create_llm, find_openrouter_key
from toolbox import ToolBox

CASES = [
    ("set a timer for 20 seconds", "set_timer", ("timer",)),
    ("turn on the bedroom light at 40 percent", "set_lights", ("light", "bedroom")),
    ("what time is it right now?", "get_time", ("time", ":", "am", "pm")),
]
DENIALS = ("i don't have the capability", "i cannot control", "i was unable", "i'm unable")


def main() -> int:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="tool-adherence check")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning:free")
    args = parser.parse_args()

    llm = create_llm(
        args.provider,
        args.model,
        find_openrouter_key(),
        openrouter_order="",
    )
    print(f"provider: {llm.label}")

    failures = 0
    for prompt, expected_tool, expected_words in CASES:
        toolbox = ToolBox()
        agent = Agent(llm, toolbox)
        reply = agent.respond(prompt)
        called = [trace.name for trace in reply.tools]
        text = reply.text.lower()
        problems: list[str] = []

        if expected_tool not in called:
            problems.append(f"did not call {expected_tool} (called {called or 'nothing'})")
        if any(denial in text for denial in DENIALS):
            problems.append("denied capability after a successful tool call")
        if not any(word in text for word in expected_words):
            problems.append("final answer does not reflect the tool result")
        if reply.error:
            problems.append("reply flagged as an error")

        status = "PASS" if not problems else "FAIL"
        failures += 0 if not problems else 1
        print(f"  [{status}] {prompt!r}")
        print(f"          tools={called or 'none'}  llm={reply.llm_seconds:.1f}s")
        print(f"          reply={reply.text[:110]!r}")
        for problem in problems:
            print(f"          -> {problem}")

    print()
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} case(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
