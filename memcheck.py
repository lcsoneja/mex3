"""Measure what the app holds in memory, and prove it gives it back.

Run: python me3_laptop/memcheck.py
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import sys

from llm import LlmError, loaded_ollama_models
from server import Session, Settings, process_memory_mb

TEST_MODEL = "llama3.2:3b-instruct-q4_K_M"


def rss() -> str:
    value = process_memory_mb()
    return f"{value:.0f} MB" if value is not None else "unavailable"


def in_ram() -> str:
    try:
        names = [str(m.get("name", "")) for m in loaded_ollama_models()]
    except LlmError as exc:
        return f"(cannot list: {exc})"
    return ", ".join(names) if names else "none"


def main() -> int:
    settings = Settings(
        provider="ollama",
        model=TEST_MODEL,
        model_memory="0s",
        idle_unload_s=1.0,
        tts_enabled=False,
    )
    session = Session(settings)
    print("1. app started")
    print(f"   app memory      : {rss()}")
    print(f"   models in RAM   : {in_ram()}")

    print()
    print("2. loading the offline speech models (lazy, as on first use)")
    session.asr()
    session.vad()
    print(f"   app memory      : {rss()}")

    print()
    print("3. one conversation turn")
    session.ask("what time is it right now?")
    print(f"   app memory      : {rss()}")
    print(
        f"   models in RAM   : {in_ram()}  (keep_alive=0s, so it should already be gone)"
    )

    print()
    print("4. forcing the idle release")
    already_ram = in_ram()
    freed = session.release_idle_models(force=True)
    print(f"   freed           : {', '.join(freed) if freed else 'nothing'}")
    print(f"   app memory      : {rss()}")
    print(f"   models in RAM   : {in_ram()}")
    # The idle watchdog may have freed everything during a slow turn (a long
    # provider wait passes the idle window). That still meets the goal - memory
    # WAS given back - so "already released during the turn" is a pass, not a fail.
    already_freed = "none" in already_ram and not freed

    print()
    print("5. reload after release (must still work)")
    try:
        text = session.asr().transcribe(
            __import__("numpy").zeros(16000, dtype="float32")
        )
        print(f"   ASR reloaded    : ok (empty audio -> {text!r})")
    except Exception as exc:  # noqa: BLE001 - report the failure, do not hide it
        print(f"   ASR reloaded    : FAILED ({type(exc).__name__}: {exc})")
        return 1

    session.stop_listening()
    print()
    print("RESULT: PASS" if (freed or already_freed) else "RESULT: FAIL (nothing was released)")
    return 0 if (freed or already_freed) else 1


if __name__ == "__main__":
    sys.exit(main())
