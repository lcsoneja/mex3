"""ME3 browser MCP server: a web operative with human-like page control.

A third sibling of tts_mcp.py / asr_mcp.py using the same mcp_core stdio protocol.
Tools give an agent what a human has: navigate, click, type, press keys, read the
page, and screenshot - fully automated, no human at the mouse.

Safety rules baked in:
- navigation is http/https only (file:// and custom schemes are refused)
- the browser is headless unless SONJ_BROWSER_HEADLESS=0
- every action is logged to stderr, never to stdout (stdout is the protocol)

Run standalone (stdio):
    python me3_laptop/browser_mcp.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeout,
)

from mcp_core import Server, ToolError, log, make_tool

OUT_DIR = Path(__file__).resolve().parent / "out"
HEADLESS = os.environ.get("SONJ_BROWSER_HEADLESS", "1") != "0"
# Playwright's bundled Chromium may not be downloaded; use the machine's browsers.
# Order: Chrome, then Edge - whatever this machine actually has.
_BROWSER_CANDIDATES = (
    ("chrome", Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")),
    ("chrome", Path(r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe")),
    ("msedge", Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")),
)


class _BrowserState:
    """The one browser instance this server keeps alive across tool calls."""

    def __init__(self) -> None:
        self.pw: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None


_state = _BrowserState()


def _page() -> Page:
    if _state.page is None:
        if _state.pw is None:
            _state.pw = sync_playwright().start()
        if _state.browser is None:
            channel, executable = next(
                ((c, p) for c, p in _BROWSER_CANDIDATES if p.is_file()), ("", Path())
            )
            _state.browser = _state.pw.chromium.launch(
                headless=HEADLESS,
                channel=channel or None,
                executable_path=str(executable) if executable else None,
            )
            log("browser launched: " + (channel or "bundled") + " headless=" + str(HEADLESS))
        page = _state.browser.new_page()
        page.set_default_timeout(15000)
        _state.page = page
    return _state.page


def _safe_url(url: str) -> str:
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("only http(s) URLs with a host can be visited, got: " + url)
    return url


def navigate(url: str) -> str:
    page = _page()
    try:
        page.goto(_safe_url(url), wait_until="domcontentloaded")
    except PlaywrightTimeout as exc:
        raise ToolError("navigation timed out: " + str(exc)[:120]) from exc
    return "navigated to " + url + " | title: " + page.title()


def click(target: str) -> str:
    """Click by CSS selector, or by visible text when target starts with 'text='."""
    page = _page()
    if target.startswith("text="):
        page.get_by_text(target[5:], exact=False).first.click()
    else:
        page.click(target)
    return "clicked " + target


def type_text(target: str, text: str) -> str:
    page = _page()
    page.fill(target, text)
    return "typed into " + target


def press_key(key: str) -> str:
    page = _page()
    page.keyboard.press(key)
    return "pressed " + key


def page_text() -> str:
    page = _page()
    body = page.inner_text("body")
    return body[:4000]


def screenshot(name: str = "") -> str:
    page = _page()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUT_DIR / ("browser_" + (name or stamp) + ".png")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path))
    return str(path)


def page_title() -> str:
    return _page().title()


def build_server() -> Server:
    server = Server("me3-browser")
    server.add(
        make_tool(
            "navigate",
            "Open a URL in the automated browser (http/https only).",
            {"url": {"type": "string", "description": "the page to open"}},
            lambda url: navigate(url),
            required=["url"],
        ),
        make_tool(
            "click",
            "Click an element: CSS selector, or 'text=some visible words'.",
            {"target": {"type": "string", "description": "CSS selector or text=..."}},
            lambda target: click(target),
            required=["target"],
        ),
        make_tool(
            "type",
            "Type text into an input (CSS selector).",
            {"target": {"type": "string"}, "text": {"type": "string"}},
            lambda target, text: type_text(target, text),
            required=["target", "text"],
        ),
        make_tool(
            "press",
            "Press a keyboard key in the page (e.g. Enter).",
            {"key": {"type": "string"}},
            lambda key: press_key(key),
            required=["key"],
        ),
        make_tool(
            "read_page",
            "Read the visible page text (up to 4000 chars).",
            {},
            lambda: page_text(),
        ),
        make_tool(
            "page_title",
            "The current page title.",
            {},
            lambda: page_title(),
        ),
        make_tool(
            "screenshot",
            "Save a PNG screenshot; returns the file path.",
            {"name": {"type": "string", "description": "optional file name"}},
            lambda name="": screenshot(str(name)),
        ),
    )
    return server


def main() -> int:
    log("browser-mcp: playwright ready, headless=" + str(HEADLESS))
    return build_server().serve()


if __name__ == "__main__":
    raise SystemExit(main())
