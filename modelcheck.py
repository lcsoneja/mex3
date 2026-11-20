"""Check that every model the app offers actually answers.

Run: python me3_laptop/modelcheck.py
"""

from __future__ import annotations

import sys

import requests

from llm import (
    LlmError,
    OllamaClient,
    list_ollama_models,
    loaded_ollama_models,
    unload_ollama_model,
)
from toolbox import ToolBox

PROMPT = "Reply with the single word: ok"
OLLAMA_DIRECT = "http://127.0.0.1:11434/api/chat"


def check_ollama(model: str) -> tuple[bool, str]:
    """Ask one model for a one-word reply through the app's own client."""
    try:
        # keep_alive 0s: this is a probe, so it must not pin the model in RAM.
        reply = OllamaClient(model, keep_alive="0s").chat(
            [{"role": "user", "content": PROMPT}]
        )
    except LlmError as exc:
        return False, str(exc)[:90]
    text = (reply.content or "").strip().replace(chr(10), " ")[:40]
    return True, f"{reply.latency_s:5.1f}s -> {text!r}"


def check_raw_status(model: str) -> str:
    """Show the raw HTTP status so a 4xx/5xx is unambiguous."""
    try:
        response = requests.post(
            OLLAMA_DIRECT,
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": False,
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        return f"transport error: {exc}"
    return f"HTTP {response.status_code}: {response.text[:120]}"


def check_openrouter(api_key: str = "") -> str:
    """Confirm the configured OpenRouter ids exist on the public model list."""
    try:
        payload = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()
    except (requests.RequestException, ValueError) as exc:
        return f"could not list OpenRouter models: {exc}"
    available = {str(m.get("id")) for m in payload.get("data", [])}
    wanted = [
        "z-ai/glm-5.3-flash",
        "openai/gpt-6-luna",
        "google/gemini-3.8-flash",
        "qwen/qwen3.8-27b",
    ]
    lines = [
        f"  {name:28s} {'FOUND' if name in available else 'NOT ON OPENROUTER'}"
        for name in wanted
    ]
    lines.append(
        f"  ({len(available)} models listed; key {'set' if api_key else 'not set'})"
    )
    return chr(10).join(lines)


def check_openrouter_live(model: str = "z-ai/glm-5.3-flash") -> str:
    """One real chat plus one tool request, using the key already on this machine."""
    from llm import OpenRouterClient, find_openrouter_key

    api_key = find_openrouter_key()
    if not api_key:
        return "  no OpenRouter key found - skipped"
    client = OpenRouterClient(model, api_key)
    lines: list[str] = []
    try:
        plain = client.chat(
            [{"role": "user", "content": "Reply with the single word: ok"}]
        )
        lines.append(
            f"  plain reply : {plain.latency_s:5.1f}s -> {plain.content[:40]!r}"
        )
    except LlmError as exc:
        return f"  FAILED: {exc}"

    toolbox = ToolBox()
    try:
        tooled = client.chat(
            [{"role": "user", "content": "Set a timer for 30 seconds."}],
            toolbox.schemas(),
        )
        names = [call.name for call in tooled.tool_calls]
        lines.append(
            f"  tool request: {tooled.latency_s:5.1f}s -> tool_calls={names or 'none'}"
        )
    except LlmError as exc:
        lines.append(f"  tool request: FAILED {exc}")
    return chr(10).join(lines)


def main() -> int:
    failures = 0

    print("Ollama daemon:")
    try:
        models = list_ollama_models()
    except LlmError as exc:
        print(f"  UNREACHABLE: {exc}")
        print("RESULT: FAIL (start the Ollama app)")
        return 1
    print(f"  reachable, {len(models)} model(s) installed")

    print()
    print("each model the app offers:")
    for name in models:
        ok, detail = check_ollama(name)
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")

    print()
    print("raw status for the model the app is configured with:")
    print(f"  {check_raw_status('rafw007/qwen35-claude-coder:9b')}")

    print()
    print("OpenRouter catalogue:")
    print(check_openrouter())

    print()
    print("OpenRouter live (key from this machine, memory only):")
    print(check_openrouter_live())

    print()
    print("memory left behind by this check:")
    try:
        for entry in loaded_ollama_models():
            name = str(entry.get("name", ""))
            if name:
                unload_ollama_model(name)
                print(f"  unloaded {name}")
        remaining = [str(m.get("name", "")) for m in loaded_ollama_models()]
        print(f"  still in RAM: {', '.join(remaining) if remaining else 'none'}")
    except LlmError as exc:
        print(f"  could not verify: {exc}")

    print()
    print(
        "RESULT:",
        "PASS" if failures == 0 else f"FAIL ({failures} model(s) did not answer)",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
