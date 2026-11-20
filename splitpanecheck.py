"""Focused probe: the split view, the agent terminals, and the countdown chip."""
import sys

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"
problems = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        problems.append(msg)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge")
    page = b.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL, wait_until="load")
    page.wait_for_timeout(3500)
    check(page.locator("#chat-top").count() == 1, "chat-top wrapper exists")
    check(page.locator("#lower").count() == 1, "lower pane grid exists")
    check(page.locator("#btn-count").count() == 1, "countdown chip exists")
    check(page.locator("#btn-lower").count() == 1, "lower toggle exists")
    roles = page.evaluate("Object.keys((state && state.agents) || {}).length")
    panes = page.locator(".lpane[data-role]").count()
    check(panes >= roles, f"one pane per roster role ({panes} panes, {roles} roles)")
    check(page.locator("#lbody-community").count() == 1, "community pane body exists")
    fs = page.evaluate("getComputedStyle(document.querySelector('.lpane .lbody')).fontSize")
    check(fs.startswith("6"), f"pane font size is 6px (got {fs})")
    rows = page.evaluate("document.querySelectorAll('#lower .lpane').length")
    check(rows > 1, "bottom half has multiple panes")
    check(not errors, f"no uncaught page errors ({errors[:2] or 'none'})")
    page.screenshot(path="out/split_panes.png")
    b.close()

print("screenshot: out/split_panes.png")
print("RESULT:", "PASS" if not problems else "FAIL")
sys.exit(0 if not problems else 1)
