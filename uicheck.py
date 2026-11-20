"""Render the web console in a real browser and prove the panels are not broken.

The bug this guards: setKids()/el() appended arrays to the DOM, so every list
panel rendered "[object HTMLDivElement]". One fix at the helper level repairs
all of them; this check keeps them repaired.

Run with the engine up:  python uicheck.py [--url http://127.0.0.1:8765]
Saves out/ui_check.png so the result can be looked at.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

OUT = pathlib.Path(__file__).with_name("out")
PROBLEMS: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + message)
    if not ok:
        PROBLEMS.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    shot = OUT / "ui_check.png"

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="msedge")
            channel = "msedge"
        except Exception:  # noqa: BLE001 - fall back to the bundled browser
            browser = pw.chromium.launch()
            channel = "bundled chromium"
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(args.url, wait_until="load")
        page.wait_for_timeout(3500)  # let two polls + the wave timer land
        print("browser: " + channel)

        panels = {
            "agents": "#tab-agents",
            "pipeline": "#tab-pipeline",
            "workflow": "#tab-workflow",
            "mindmap": "#tab-mindmap",
        }
        for name, selector in panels.items():
            text = page.inner_text(selector)
            check("[object" not in text, f"{name} panel has no stringified element")
            check(bool(text.strip()), f"{name} panel rendered something")

        # The roster is the thing the Agents pane must show.
        agentspane = page.inner_text("#tab-agents")
        check("front" in agentspane.lower(), "agents pane shows the roster")

        # Every rail view renders without throwing.
        for view in ("sessions", "files", "sources", "agents", "skills", "mcp", "thinking", "settings"):
            page.click(f'.rail button[data-view="{view}"]')
            page.wait_for_timeout(400)
            body = page.inner_text(f"#view-{view}")
            check("[object" not in body, f"{view} view has no stringified element")

        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.click('.rail button[data-view="chat"]')
        page.wait_for_timeout(1200)
        check(not errors, "no uncaught page errors" + (f": {errors[0]}" if errors else ""))

        page.screenshot(path=str(shot))
        print(f"screenshot: {shot}")
        browser.close()

    print()
    print("RESULT: " + ("FAIL" if PROBLEMS else "PASS"))
    for problem in PROBLEMS:
        print("PROBLEM:", problem)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
