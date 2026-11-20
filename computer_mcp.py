"""ME3 computer-use MCP server: full desktop control for the agent.

Gives an agent eyes and hands on the desktop: screenshot, click, type, hotkeys,
scroll, launch applications, open URLs, and run shell commands. Sibling of
browser_mcp.py / tts_mcp.py using the same mcp_core stdio protocol.

Safety:
- pyautogui FAILSAFE stays on (slam the mouse into a screen corner to abort).
- run_command is powerful; the calling agent is expected to gate risky commands
  through the owner's approval flow. Output is truncated and time-bounded.
- Every action is logged to stderr (stdout is the MCP protocol).

Run standalone (stdio):  python sonj3/computer_mcp.py
"""

from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from pathlib import Path

import pyautogui

from mcp_core import Server, ToolError, log, make_tool

OUT_DIR = Path(__file__).resolve().parent / "out" / "computer_shots"
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05
MAX_OUTPUT = 6000


def _shot() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = pyautogui.screenshot()
    path = OUT_DIR / f"shot_{int(time.time() * 1000)}.png"
    img.save(str(path))
    log(f"screenshot {img.width}x{img.height} -> {path}")
    return f"Saved screenshot {img.width}x{img.height} to {path}"


def _click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    pyautogui.click(x=int(x), y=int(y), button=button, clicks=int(clicks))
    return f"Clicked {button} x{clicks} at ({x}, {y})."


def _type(text: str, interval: float = 0.02) -> str:
    pyautogui.write(str(text), interval=float(interval))
    return f"Typed {len(str(text))} characters."


def _hotkey(keys: str) -> str:
    parts = [k.strip() for k in str(keys).split("+") if k.strip()]
    if not parts:
        raise ToolError("hotkey needs keys like 'ctrl+c'")
    pyautogui.hotkey(*parts)
    return f"Pressed {'+'.join(parts)}."


def _scroll(clicks: int) -> str:
    pyautogui.scroll(int(clicks))
    return f"Scrolled {clicks}."


def _move(x: int, y: int) -> str:
    pyautogui.moveTo(int(x), int(y), duration=0.15)
    return f"Moved mouse to ({x}, {y})."


def _screen_size() -> str:
    w, h = pyautogui.size()
    return f"Screen is {w}x{h}. Mouse at {pyautogui.position()}."


def _launch_app(app: str) -> str:
    target = str(app or "").strip()
    if not target:
        raise ToolError("launch_app needs an app name or path")
    if os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        subprocess.Popen([target])
    log(f"launched {target}")
    return f"Launched {target}."


def _open_url(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        raise ToolError("open_url needs a url")
    if not target.startswith(("http://", "https://")):
        target = "https://" + target
    webbrowser.open(target)
    log(f"opened {target}")
    return f"Opened {target} in the default browser."


def _run_command(command: str, timeout: int = 60) -> str:
    cmd = str(command or "").strip()
    if not cmd:
        raise ToolError("run_command needs a command")
    log(f"run_command: {cmd[:120]}")
    try:
        # Intentional computer-use shell tool (owner requested full command access);
        # risky commands are expected to be gated by the agent's owner-approval flow.
        # pi-lens-ignore: python-subprocess-shell
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout), 300)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out: {cmd[:120]}"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    out = out.strip()[:MAX_OUTPUT]
    return (
        f"exit={proc.returncode}\n{out}"
        if out
        else f"exit={proc.returncode} (no output)"
    )


def build_server() -> Server:
    server = Server("me3-computer")
    server.add(
        make_tool(
            "screenshot",
            "Capture the whole screen and save a PNG; returns the path and size.",
            {},
            lambda: _shot(),
        ),
        make_tool(
            "click",
            "Click at screen coordinates (x, y).",
            {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {
                    "type": "string",
                    "description": "left|right|middle, default left",
                },
                "clicks": {"type": "integer", "description": "1=single, 2=double"},
            },
            lambda x, y, button="left", clicks=1: _click(x, y, button, clicks),
            required=["x", "y"],
        ),
        make_tool(
            "type_text",
            "Type text on the keyboard (into whatever is focused).",
            {"text": {"type": "string"}, "interval": {"type": "number"}},
            lambda text, interval=0.02: _type(text, interval),
            required=["text"],
        ),
        make_tool(
            "hotkey",
            "Press a key combination, e.g. 'ctrl+c', 'alt+tab', 'win+d'.",
            {"keys": {"type": "string"}},
            lambda keys: _hotkey(keys),
            required=["keys"],
        ),
        make_tool(
            "scroll",
            "Scroll vertically; positive=up, negative=down.",
            {"clicks": {"type": "integer"}},
            lambda clicks: _scroll(clicks),
            required=["clicks"],
        ),
        make_tool(
            "move",
            "Move the mouse to (x, y) without clicking.",
            {"x": {"type": "integer"}, "y": {"type": "integer"}},
            lambda x, y: _move(x, y),
            required=["x", "y"],
        ),
        make_tool(
            "screen_size",
            "Get screen dimensions and current mouse position.",
            {},
            lambda: _screen_size(),
        ),
        make_tool(
            "launch_app",
            "Launch a desktop application by name or path (notepad, calculator, ...).",
            {"app": {"type": "string"}},
            lambda app: _launch_app(app),
            required=["app"],
        ),
        make_tool(
            "open_url",
            "Open a URL in the owner's default browser, visible on screen.",
            {"url": {"type": "string"}},
            lambda url: _open_url(url),
            required=["url"],
        ),
        make_tool(
            "run_command",
            "Run a shell command and return exit code + output. Powerful: gate risky commands via owner approval.",
            {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            lambda command, timeout=60: _run_command(command, timeout),
            required=["command"],
        ),
    )
    return server


if __name__ == "__main__":
    raise SystemExit(build_server().serve())
