"""Regression checks for MEX3 Deck alignment (Front prompt, speech sanitize, instant_barge, front_only, context logging).

Run: python deckcheck.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ME3_NO_PERSIST", "1")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> int:
    from agent import DEFAULT_SYSTEM_PROMPT
    from server import Session, load_settings, sanitize_for_speech

    st = load_settings()
    st.front_uses_pi = False
    st.tts_enabled = False
    st.tts_simulate = True
    s = Session(st)

    # 1. Task-agnostic Front prompt + versioned prompt file
    pfile = Path(__file__).resolve().parent / "prompts" / "front_system_v4.txt"
    check("prompt_file_versioned_exists", pfile.is_file(), str(pfile.name))

    front_prompt = s._role_system_prompt("front")
    forbidden_tasks = (
        "weather", "get_time", "set_timer", "set_alarm", "set_lights",
        "thermostat", "calculate", "snake", "gazette", "music", "volume",
    )
    for txt_name, txt in [("server_front", front_prompt), ("agent_default", DEFAULT_SYSTEM_PROMPT)]:
        hits = [w for w in forbidden_tasks if w in txt.lower()]
        check(f"{txt_name}_no_specific_tasks", not hits, f"forbidden_hits={hits}")
        low = txt.lower()
        check(
            f"{txt_name}_has_deck_rules",
            "nova" in low and "markdown" in low and "url" in low and "mcp" in low and "stop and listen" in low,
            "missing required deck directive",
        )

    # 2. Never read markdown or URLs aloud (_sanitize_for_speech)
    dirty = "**Hello** `world`! Check [the Python docs](https://docs.python.org/3/library/pathlib.html) and visit https://example.com/a/b?x=1 #note"
    clean = sanitize_for_speech(dirty)
    check(
        "sanitize_strips_markdown_and_urls",
        ("https://" not in clean) and ("http://" not in clean) and ("**" not in clean) and ("`" not in clean) and ("the Python docs" in clean),
        f"clean={clean!r}",
    )

    # 3. instant_barge defaults to True; interrupt() aborts front pane & drops stale audio
    check("instant_barge_default_true", bool(getattr(st, "instant_barge", None) is True), f"val={getattr(st, 'instant_barge', None)}")
    aborted: list[str] = []
    s.pi_abort = lambda role="": aborted.append(role) or "aborted"  # type: ignore[method-assign]
    s._speaking = True
    s._speech_run = [object()]  # type: ignore[list-item]
    s._held_text = "stale partial"
    s.interrupt()
    check(
        "interrupt_stops_llm_and_drops_stale_audio",
        ("front" in aborted) and (not s._speaking) and (len(s._speech_run) == 0) and (s._held_text == ""),
        f"aborted={aborted} speaking={s._speaking}",
    )

    # 4. front_only defaults to False; when True forces Front & blocks delegation
    check("front_only_default_false", getattr(st, "front_only", None) is False, f"val={getattr(st, 'front_only', None)}")
    s.update_settings({"front_only": True})
    dec = s.classify_decision("build a web app and research news")
    del_msg = s.delegate_task("build something", role="back")
    ask_msg = s.ask_agent_turn("doctor", "check logs")
    check(
        "front_only_forces_front_and_blocks_delegation",
        dec.route == "front" and ("front-only" in del_msg.lower() or "only" in del_msg.lower()) and ("front-only" in ask_msg.lower() or "only" in ask_msg.lower()),
        f"route={dec.route} del={del_msg[:50]!r}",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] deckcheck error: {type(exc).__name__}: {exc}")
        sys.exit(1)
