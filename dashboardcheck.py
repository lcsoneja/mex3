"""Browser check: the dashboard aggregates status, the reasoning trace, the inline
roster (model + thinking) and the crew studio, and the chat shows the main directory.

Run with the engine up:  python dashboardcheck.py
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

PROBLEMS: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + message)
    if not ok:
        PROBLEMS.append(message)


def main() -> int:
    with sync_playwright() as play:
        browser = play.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://127.0.0.1:8765/", wait_until="load")
        page.wait_for_timeout(3500)

        check(page.is_visible("#view-dashboard"), "dashboard is the default view")
        check(not page.is_visible("#view-chat"), "chat is hidden by default")

        body = page.inner_text("#dashboard-body")
        check("[object" not in body, "dashboard has no stringified element")
        low = body.lower()  # the card headings are CSS-uppercased, so compare folded
        for marker in ("reasoning trace", "change model", "crew studio", "execution"):
            check(marker in low, "dashboard shows: " + marker)

        roles = page.locator("#db-roster .rrow").count()
        selects = page.locator("#db-roster select").count()
        check(roles >= 6, f"roster rows rendered ({roles})")
        check(selects >= roles * 2, f"model+thinking selects ({selects} for {roles} roles)")
        check(page.locator("#db-canvas .col").count() == 3, "canvas has 3 columns (agent/task/tool)")
        check("agents" in page.inner_text("#db-resources").lower(), "resources panel lists agents")

        check(page.inner_text("#db-cwd").strip() != "", "dashboard shows the main directory")
        page.click('.rail button[data-view="chat"]')
        page.wait_for_timeout(600)
        check(page.inner_text("#chat-cwd").strip().endswith("Exercise 3"), "chat shows the project root")

        # the Agents pane is interactive too (request 3)
        page.click('.rail button[data-view="chat"]')
        page.wait_for_timeout(600)
        n = page.locator("#tab-agents select").count()
        check(n >= 12, f"agents pane has model+thinking selects ({n})")

        # changing thinking through the pane persists via /api/set_role_thinking
        row = page.locator("#tab-agents .rrow", has_text="researcher")
        row.locator("select").nth(1).select_option("high")
        page.wait_for_timeout(1500)
        roles = page.evaluate("() => fetch('/api/roles').then(r => r.json())")
        check(
            (roles.get("roles", {}).get("researcher", {}) or {}).get("thinking") == "high",
            "thinking change persisted through the API",
        )
        row.locator("select").nth(1).select_option("medium")
        page.wait_for_timeout(1000)

        # the roster select must survive the poll loop (no rebuild under an open dropdown)
        page.click('.rail button[data-view="dashboard"]')
        page.wait_for_timeout(600)
        page.evaluate("() => { window.__sel = document.querySelector('#db-roster select'); }")
        page.wait_for_timeout(1900)  # two poll cycles
        check(
            page.evaluate("() => window.__sel === document.querySelector('#db-roster select')"),
            "the roster select survives polls (built once)",
        )

        page.screenshot(path="out/ui_dashboard.png")
        check(not errors, "no uncaught page errors: " + (errors[0][:120] if errors else ""))
        browser.close()

    print()
    print("RESULT: " + ("FAIL" if PROBLEMS else "PASS"))
    for problem in PROBLEMS:
        print("PROBLEM:", problem)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
