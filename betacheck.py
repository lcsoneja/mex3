"""Beta test: walk the app like a user and report what actually happens.

Journeys: first contact, the grouped model picker + search, the directory card,
voice (push-to-talk submit and continuous listen), and one real text turn.
Writes screenshots to out/ so findings can be looked at.
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

FINDINGS: list[str] = []


def note(ok: bool, text: str) -> None:
    print(("  ok   " if ok else "  BUG  ") + text)
    if not ok:
        FINDINGS.append(text)


def main() -> int:
    with sync_playwright() as play:
        browser = play.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://127.0.0.1:8765/", wait_until="load")
        page.wait_for_timeout(3500)

        # --- journey 1: the grouped, searchable model picker -----------------
        page.click("#btn-model")
        page.wait_for_timeout(1500)
        groups = page.locator("#model-pop optgroup").count()
        note(groups >= 3, f"model picker groups by provider/family ({groups} groups)")
        page.fill("#model-search", "deepseek")
        page.wait_for_timeout(500)
        opts = page.locator("#model-select option").all_inner_texts()
        note(any("deepseek" in o.lower() for o in opts), f"search 'deepseek' filters ({len(opts)} hits)")
        page.screenshot(path="out/beta_model_picker.png")
        page.fill("#model-search", "")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

        # --- journey 2: the directory card -----------------------------------
        page.wait_for_timeout(600)
        path = page.inner_text("#db-dir-path").strip()
        note(path.endswith("Exercise 3"), f"directory card shows the main path ({path[-24:]})")
        note("me3_laptop" in page.inner_text("#db-dir-list"), "directory listing shows entries")

        # --- journey 3: voice (the mic lives in the Chat view) ---------------
        page.click('.rail button[data-view="chat"]')
        page.wait_for_timeout(600)

        def state() -> dict:
            return page.evaluate("() => fetch('/api/state').then(r => r.json())")

        page.click("#btn-mic")
        page.wait_for_timeout(1200)
        note(str(state().get("mode")) == "ptt", "mic button starts push-to-talk")
        label = page.inner_text("#btn-mic")
        note("Send" in label, "mic button becomes a submit action once recording")
        page.click("#btn-mic")
        page.wait_for_timeout(1200)
        s = state()
        note(str(s.get("mode")) == "off" and not s.get("recording"),
             f"second click submits and stops ({s.get('mode')}, rec={s.get('recording')})")

        page.click("#btn-listen")
        page.wait_for_timeout(1200)
        note(str(state().get("mode")) == "continuous", "Listen starts continuous listening")
        page.click("#btn-listen")
        page.wait_for_timeout(1200)
        note(str(state().get("mode")) == "off", "Listen again stops continuous listening")
        page.screenshot(path="out/beta_voice_idle.png")

        # --- journey 4: one real text turn -----------------------------------
        before = len(state().get("turns") or [])
        page.fill("#input", "Reply with exactly: beta test ok")
        page.keyboard.press("Enter")
        deadline = time.time() + 90
        grew = False
        while time.time() < deadline:
            time.sleep(3)
            turns = state().get("turns") or []
            if len(turns) > before and any((t.get("source") != "task") for t in turns[before:]):
                grew = True
                break
        note(grew, "a text turn produced a reply")
        page.wait_for_timeout(800)
        page.screenshot(path="out/beta_after_turn.png")

        note(not errors, "no uncaught page errors" + (f": {errors[0][:100]}" if errors else ""))
        browser.close()

    print()
    print("RESULT: " + ("ISSUES" if FINDINGS else "PASS"))
    for finding in FINDINGS:
        print("BUG:", finding)
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    sys.exit(main())
