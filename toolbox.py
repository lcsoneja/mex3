"""Tools the voice agent can call.

Most tools are local and cheap: the point of the laptop build is to prove the
conversation loop and the tool plumbing. The lights tool is a stub that the
Raspberry Pi version replaces with real GPIO output. Web search and page
reading go through websearch.py, which needs no API key.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

import filereader
from websearch import SearchError as WebSearchError
from websearch import chain_is_public, format_results, web_search
from websearch import read_page as fetch_page

WEATHER_TIMEOUT_S = 8
MAX_TIMER_SECONDS = 24 * 3600

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
_MATH_FUNCTIONS: dict[str, Callable[..., float]] = {
    # abs/round carry generic overloads, so wrap them to a plain float result.
    "abs": lambda x: float(abs(x)),
    "round": lambda x, digits=0: float(round(x, int(digits))),
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


HARNESS_TIMEOUT_S = float(os.environ.get("ME3_HARNESS_TIMEOUT_S", "300"))
MAX_HARNESS_CHARS = 1500


def harness_command(task: str, tools: str = "") -> list[str] | None:
    """Build the command that runs the agent harness headlessly.

    `pi --print` writes the final answer to stdout and exits, which is what makes
    the harness usable as a tool. ME3_HARNESS_SSH sends the run to the Raspberry
    Pi, where the harness is part of the deliverable rather than a laptop extra.
    """
    plain = os.environ.get("ME3_HARNESS_CMD", "").strip()
    if plain:
        base = shlex.split(plain)
    else:
        found = shutil.which("pi") or shutil.which("pi.cmd") or shutil.which("pi.exe")
        if not found:
            return None
        base = [found]
    if tools.strip():
        base += ["--tools", tools.strip()]
    base += ["--print", task]
    if os.name == "nt" and base[0].lower().endswith((".cmd", ".bat")):
        # CreateProcess cannot start a batch file directly.
        base = ["cmd.exe", "/c", *base]
    host = os.environ.get("ME3_HARNESS_SSH", "").strip()
    if host:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            " ".join(shlex.quote(p) for p in base),
        ]
    return base


class ToolError(RuntimeError):
    """Raised when a tool cannot complete its work."""


@dataclass
class ToolResult:
    """Tool output as the model will see it."""

    content: str
    ok: bool = True


class ToolBox:
    """Owns the tool schemas, the implementations, and any timers they start."""

    def __init__(
        self,
        on_timer: Callable[[str], None] | None = None,
        on_task: Callable[[str, str], str] | None = None,
        on_approval: Callable[[str, str], int] | None = None,
        on_speak_script: Callable[[str], int] | None = None,
        on_crew: Callable[[str, list[str], str], str] | None = None,
        on_ask_agent: Callable[[str, str, str], str] | None = None,
    ) -> None:
        self._on_timer = on_timer
        # A2A: consult ONE roster role (researcher, critic, ...) and get its answer
        # back inside this turn - the front agent stays the supervisor.
        self._on_ask_agent = on_ask_agent
        # #24: hands a NOVA:/PIATTOS: script to the voice layer; returns lines spoken.
        self._on_speak_script = on_speak_script
        # W2.1: starts a crew (roles from agents.json) on background work.
        self._on_crew = on_crew
        # #30: the engine queues a risky action and returns its approval id; the
        # owner answers through the UI or POST /api/approval.
        self._on_approval = on_approval
        # Injected by the app: hands a brief to the background task layer and returns a
        # sentence the agent can say. None means the build has no task layer.
        self._on_task = on_task
        # M4.1: the session that learned a fact, written into MEMORY.md lines as
        # provenance ("- [stamp | session] fact"). The engine sets it.
        self.session_tag: str = ""
        # Filesystem navigation state: the agent browses the machine like pi.dev does -
        # cd/pwd/tree keep a cwd so every later read_file/list_files is relative to it.
        self._cwd = Path.cwd()
        self._timers: list[threading.Timer] = []
        self._lights: dict[str, dict[str, Any]] = {}
        self._reminders: list[dict[str, str]] = []
        # Simulated device state for the ME2 command vocabulary: on the Pi these
        # become real GPIO / service calls.
        self._device: dict[str, Any] = {
            "media": "stopped",
            "volume": 40,
            "temperature": 22.0,
            "alarm": None,
        }

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI/Ollama-compatible function schemas."""
        entries = [
            _function(
                "list_apis",
                "List the registered external APIs the agent can call (#23).",
                {},
            ),
            _function(
                "register_api",
                "Register an external HTTP(S) API endpoint in apis.json so call_api can use it.",
                {
                    "name": {
                        "type": "string",
                        "description": "short API name, e.g. 'linkedin'",
                    },
                    "url": {
                        "type": "string",
                        "description": "public https:// endpoint URL",
                    },
                    "description": {
                        "type": "string",
                        "description": "what this API does",
                    },
                    "token": {
                        "type": "string",
                        "description": "optional Bearer token for Authorization header",
                    },
                },
                required=["name", "url"],
            ),
            _function(
                "call_api",
                "Call one registered API by name with optional query parameters or JSON body (#23).",
                {
                    "name": {"type": "string", "description": "registered API name"},
                    "params": {
                        "type": "object",
                        "description": 'query parameters, e.g. {"q": "weather"}',
                        "additionalProperties": {"type": "string"},
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method, default GET",
                    },
                    "headers": {
                        "type": "object",
                        "description": "optional extra HTTP headers",
                        "additionalProperties": {"type": "string"},
                    },
                },
                required=["name"],
            ),
            _function(
                "open_url",
                "Open a website or URL in the owner's default browser, visible on screen (e.g. youtube.com). Use this to open YouTube or any site.",
                {
                    "url": {
                        "type": "string",
                        "description": "full URL or domain, e.g. https://youtube.com",
                    },
                },
                required=["url"],
            ),
            _function(
                "open_app",
                "Launch a desktop application by name or path (e.g. notepad, calculator, spotify).",
                {
                    "app": {
                        "type": "string",
                        "description": "app name or executable path",
                    },
                },
                required=["app"],
            ),
            _function(
                "browse_page",
                "Open a public web page in a real headless browser (handles JS-rendered pages) and return its visible text.",
                {
                    "url": {"type": "string", "description": "full http(s) URL"},
                    "max_chars": {
                        "type": "integer",
                        "description": "max text chars to return, default 4000",
                    },
                },
                required=["url"],
            ),
            _function(
                "send_email",
                "Send an email. Queued for owner approval before anything is sent (#21/#30).",
                {
                    "to": {"type": "string", "description": "recipient address"},
                    "subject": {"type": "string", "description": "subject line"},
                    "body": {"type": "string", "description": "the message"},
                },
                required=["to", "subject", "body"],
            ),
            _function(
                "run_code",
                "Run a short Python snippet locally in a subprocess and return its "
                "stdout/stderr (feature b: able to code). No shell is involved.",
                {
                    "code": {"type": "string", "description": "Python source to run"},
                    "timeout_s": {
                        "type": "number",
                        "description": "seconds before it is killed (max 20)",
                    },
                },
                required=["code"],
            ),
            _function(
                "audio_overview",
                "Two-voice podcast overview of a file: Nova and Piattos discuss it (#24).",
                {"path": {"type": "string", "description": "file to overview"}},
                required=["path"],
            ),
            _function(
                "delegate_work",
                "Hand a task to a coworker agent by name (#49, CrewAI pattern).",
                {
                    "task": {"type": "string", "description": "what to do"},
                    "context": {"type": "string", "description": "why / background"},
                    "coworker": {"type": "string", "description": "agent name"},
                },
                required=["task", "coworker"],
            ),
            _function(
                "ask_question",
                "Ask a coworker agent a question instead of guessing (#49).",
                {
                    "question": {"type": "string", "description": "the question"},
                    "context": {"type": "string", "description": "background"},
                    "coworker": {"type": "string", "description": "agent name"},
                },
                required=["question", "coworker"],
            ),
            _function(
                "forget",
                "Remove matching facts from MEMORY.md (#47/#62).",
                {
                    "fact": {
                        "type": "string",
                        "description": "prefix of the fact to forget",
                    }
                },
                required=["fact"],
            ),
            _function(
                "get_time",
                "Current date and time, optionally in a named timezone.",
                {
                    "timezone": {
                        "type": "string",
                        "description": "e.g. 'Asia/Manila', or 'local'",
                    }
                },
            ),
            _function(
                "set_timer",
                "Start a timer. The assistant will announce when it finishes.",
                {
                    "seconds": {
                        "type": "integer",
                        "description": "duration in seconds",
                    },
                    "label": {"type": "string", "description": "what the timer is for"},
                },
                required=["seconds"],
            ),
            _function(
                "set_lights",
                "Turn a room light on or off, or set its brightness.",
                {
                    "room": {
                        "type": "string",
                        "description": "room name, e.g. 'bedroom'",
                    },
                    "state": {"type": "string", "enum": ["on", "off"]},
                    "brightness": {
                        "type": "integer",
                        "description": "0-100 percent, only when turning on",
                    },
                },
                required=["room", "state"],
            ),
            _function(
                "get_weather",
                "Current weather for a city.",
                {"city": {"type": "string", "description": "city name, e.g. 'Manila'"}},
                required=["city"],
            ),
            _function(
                "calculate",
                "Evaluate an arithmetic expression.",
                {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '(12 + 8) * 3'",
                    }
                },
                required=["expression"],
            ),
            _function(
                "get_stock_price",
                "Look up the live current stock price for a ticker (e.g. AAPL, NVDA, TSLA) or company name (e.g. Apple, Nvidia).",
                {
                    "symbol": {
                        "type": "string",
                        "description": "ticker symbol or company name, e.g. 'AAPL' or 'Apple'",
                    }
                },
                required=["symbol"],
            ),
            _function(
                "web_search",
                "Search the web for current information: news, facts, prices, people, "
                "events, or anything you do not already know.",
                {
                    "query": {"type": "string", "description": "what to search for"},
                    "recency": {
                        "type": "string",
                        "enum": ["any", "day", "week", "month", "year"],
                        "description": "use day or week when the user asks for the latest news",
                    },
                },
                required=["query"],
            ),
            _function(
                "read_page",
                "Fetch one web page and return its readable text. Use it after web_search "
                "when a snippet is not enough to answer. Many sites block automated "
                "readers (HTTP 403); when that happens, answer from the search snippets "
                "instead of retrying the same page.",
                {
                    "url": {"type": "string", "description": "full http(s) URL"},
                    "max_chars": {
                        "type": "integer",
                        "description": "how much text to return, default 4000",
                    },
                },
                required=["url"],
            ),
            _function(
                "set_alarm",
                "Set an alarm for a clock time. The assistant announces it when it rings.",
                {
                    "time": {
                        "type": "string",
                        "description": "e.g. '07:00' or '6:30 pm'",
                    },
                    "label": {"type": "string", "description": "what the alarm is for"},
                },
                required=["time"],
            ),
            _function(
                "adjust_lights",
                "Dim or brighten a room light without turning it off.",
                {
                    "room": {
                        "type": "string",
                        "description": "room name, e.g. 'bedroom'",
                    },
                    "direction": {"type": "string", "enum": ["dim", "brighten"]},
                    "percent": {
                        "type": "integer",
                        "description": "how much to change it, default 20",
                    },
                },
                required=["room", "direction"],
            ),
            _function(
                "adjust_temperature",
                "Make the room warmer or cooler.",
                {
                    "direction": {"type": "string", "enum": ["warmer", "cooler"]},
                    "degrees": {
                        "type": "number",
                        "description": "how many degrees, default 1",
                    },
                },
                required=["direction"],
            ),
            _function(
                "media_control",
                "Control playback: play, pause, next track or previous track.",
                {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "next", "previous"],
                    }
                },
                required=["action"],
            ),
            _function(
                "adjust_volume",
                "Turn the volume up or down.",
                {"direction": {"type": "string", "enum": ["up", "down"]}},
                required=["direction"],
            ),
            _function(
                "set_reminder",
                "Remember something for later.",
                {
                    "text": {"type": "string", "description": "what to remember"},
                    "when": {
                        "type": "string",
                        "description": "optional time, e.g. '7 pm'",
                    },
                },
                required=["text"],
            ),
            _function(
                "list_reminders", "Read back the reminders that are stored.", {}, None
            ),
            _function(
                "make_call",
                "Place a call to a contact.",
                {"contact": {"type": "string", "description": "who to call"}},
                required=["contact"],
            ),
            _function(
                "silence", "Stop talking and go quiet until spoken to again.", {}, None
            ),
        ]

        entries.append(
            _function(
                "list_agents",
                (
                    "List the agent roster: every role (front, back, router, "
                    "researcher, critic, summarizer, doctor, verifier) and the model "
                    "each one uses. Call this before ask_agent if the role name is "
                    "not obvious."
                ),
                {},
                None,
            )
        )
        entries.append(
            _function(
                "ask_agent",
                (
                    "Ask ONE agent on the roster a question or give it an instruction "
                    "and get its answer inside this turn. Use it to split work: a "
                    "researcher to gather, a critic to check, a summarizer to condense. "
                    "The specialist answers on its own model and does not see this "
                    "conversation, so put the facts it needs in question/context and "
                    "keep them short - that is what keeps the consult cheap."
                ),
                {
                    "role": {
                        "type": "string",
                        "description": "roster role, e.g. researcher, critic, summarizer",
                    },
                    "question": {
                        "type": "string",
                        "description": "the message, question or instruction to send",
                    },
                    "context": {
                        "type": "string",
                        "description": "short facts it needs (optional, <=600 chars)",
                    },
                },
                required=["role", "question"],
            )
        )
        entries.append(
            _function(
                "ask_jev",
                (
                    "Ask the Jev routing/decision delegate which specialist is best "
                    "for a task when you are unsure and would rather not ask the owner. "
                    "Returns the recommended agent plus a one-line reason. Use it for "
                    "long tasks before delegating."
                ),
                {
                    "task": {
                        "type": "string",
                        "description": "the task or decision to route",
                    },
                },
                required=["task"],
            )
        )
        entries.append(
            _function(
                "delegate_task",
                (
                    "Hand an open-ended task to the agent harness and return what it "
                    "produced. Use it for work the fixed device commands do not cover: "
                    "research, writing a file, building something, multi-step jobs."
                ),
                {
                    "task": {
                        "type": "string",
                        "description": "the job, in plain language",
                    },
                    "tools": {
                        "type": "string",
                        "description": "harness tools to allow, e.g. 'read,grep,find,ls'",
                    },
                },
                required=["task"],
            )
        )
        entries.append(
            _function(
                "read_file",
                (
                    "Read a document and quote it back with citations (file#page). Handles "
                    "PDF, DOCX, XLSX/CSV, PPTX, images and any text or markdown file."
                ),
                {
                    "path": {"type": "string", "description": "file to read"},
                    "query": {
                        "type": "string",
                        "description": "what to look for inside it (optional)",
                    },
                },
                required=["path"],
            )
        )
        entries.append(
            _function(
                "write_file",
                (
                    "Write or create a file inside the workspace (e.g. a Python script, HTML game, "
                    "markdown note, or config) so the user can open or run it."
                ),
                {
                    "path": {
                        "type": "string",
                        "description": "file path relative to current cwd or project",
                    },
                    "content": {
                        "type": "string",
                        "description": "full text content to write",
                    },
                },
                required=["path", "content"],
            )
        )
        entries.append(
            _function(
                "describe_image",
                "Inspect an attached image and report its real properties (format, size, dominant colours, path).",
                {"path": {"type": "string", "description": "path to the image file"}},
            )
        )
        entries.append(
            _function(
                "cd",
                "Change the working directory for file tools, like a terminal cd. Use this to browse the machine.",
                {
                    "folder": {
                        "type": "string",
                        "description": "the directory to move into (relative to the current one, or absolute)",
                    }
                },
            )
        )
        entries.append(
            _function(
                "pwd",
                "Print the current working directory for file tools.",
                {},
            )
        )
        entries.append(
            _function(
                "tree",
                "Show a compact directory tree of the current folder, like a terminal agent's folder view.",
                {
                    "depth": {
                        "type": "string",
                        "description": "how deep to draw (1-4, default 2)",
                    }
                },
            )
        )
        entries.append(
            _function(
                "remember",
                "Save a durable fact to MEMORY.md so you remember it after a restart.",
                {
                    "fact": {
                        "type": "string",
                        "description": "the fact worth remembering",
                    },
                },
            )
        )
        entries.append(
            _function(
                "recall",
                "Read remembered facts; optionally filter by a keyword.",
                {
                    "query": {
                        "type": "string",
                        "description": "optional keyword filter",
                    },
                },
            )
        )
        entries.append(
            _function(
                "template",
                "Run a stored prompt template from prompts/<name>.md.",
                {
                    "name": {"type": "string", "description": "template name"},
                    "args": {"type": "string", "description": "optional arguments"},
                },
            )
        )
        entries.append(
            _function(
                "list_files",
                "List readable files in a folder, so you know what can be read.",
                {
                    "folder": {"type": "string", "description": "folder to list"},
                    "pattern": {"type": "string", "description": "glob, e.g. *.pdf"},
                },
            )
        )
        entries.append(
            _function(
                "start_task",
                (
                    "Hand longer work to the background task layer, which keeps working while "
                    "the conversation continues. Use it for research, reading several "
                    "documents, or anything that would take more than a moment."
                ),
                {
                    "brief": {
                        "type": "string",
                        "description": "the job, in plain language",
                    },
                    "model": {
                        "type": "string",
                        "description": "optional model override",
                    },
                },
                required=["brief"],
            )
        )
        entries.append(
            _function(
                "start_crew",
                "Run a crew of agents on a task: sequential (each role builds on the last) "
                "or hierarchical (a manager plans, workers execute). Roles come from "
                "agents.json. Runs in the background; the result is reported to the chat.",
                {
                    "task": {"type": "string", "description": "what the crew must do"},
                    "roles": {
                        "type": "array",
                        "description": "role names from agents.json, e.g. ['researcher', 'critic', 'summarizer']",
                        "items": {"type": "string"},
                    },
                    "process": {
                        "type": "string",
                        "enum": ["sequential", "hierarchical"],
                        "description": "how the crew works, default sequential",
                    },
                },
                required=["task", "roles"],
            )
        )
        return entries

    def _open_url(self, url: str = "") -> ToolResult:
        import webbrowser

        target = str(url or "").strip()
        if not target:
            raise ToolError("open_url needs a url")
        if not target.startswith(("http://", "https://")):
            target = "https://" + target
        try:
            opened = webbrowser.open(target)
        except Exception as exc:
            raise ToolError(f"could not open {target}: {exc}") from exc
        if not opened:
            raise ToolError(f"no browser available to open {target}")
        return ToolResult(content=f"Opened {target} in the default browser.")

    def _open_app(self, app: str = "") -> ToolResult:
        import os
        import subprocess

        target = str(app or "").strip()
        if not target:
            raise ToolError("open_app needs an app name or path")
        try:
            if os.name == "nt":
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([target])
        except Exception as exc:
            raise ToolError(f"could not launch {target}: {exc}") from exc
        return ToolResult(content=f"Launched {target}.")

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool call, converting any failure into a tool-level error."""
        handlers = {
            "get_time": self._get_time,
            "set_timer": self._set_timer,
            "set_lights": self._set_lights,
            "get_weather": self._get_weather,
            "calculate": self._calculate,
            "get_stock_price": self._get_stock_price,
            "web_search": self._web_search,
            "open_url": self._open_url,
            "open_app": self._open_app,
            "read_page": self._read_page,
            "set_alarm": self._set_alarm,
            "adjust_lights": self._adjust_lights,
            "adjust_temperature": self._adjust_temperature,
            "media_control": self._media_control,
            "adjust_volume": self._adjust_volume,
            "set_reminder": self._set_reminder,
            "list_reminders": self._list_reminders,
            "describe_image": self._describe_image,
            "remember": self._remember,
            "recall": self._recall,
            "template": self._template,
            "cd": self._cd,
            "pwd": self._pwd,
            "tree": self._tree,
            "make_call": self._make_call,
            "silence": self._silence,
            "delegate_task": self._delegate_task,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_files": self._list_files,
            "start_task": self._start_task,
            "start_crew": self._start_crew,
            "list_apis": self._list_apis,
            "register_api": self._register_api,
            "call_api": self._call_api,
            "browse_page": self._browse_page,
            "send_email": self._send_email,
            "run_code": self._run_code,
            "audio_overview": self._audio_overview,
            "delegate_work": self._delegate_work,
            "ask_question": self._ask_question,
            "forget": self._forget,
            "list_agents": self._list_agents,
            "ask_agent": self._ask_agent,
            "ask_jev": self._ask_jev,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(content=f"Unknown tool '{name}'.", ok=False)
        try:
            return handler(**arguments)
        except ToolError as exc:
            return ToolResult(content=f"{name} failed: {exc}", ok=False)
        except ValueError as exc:
            # A model can pass "lots" where a number belongs; that is a tool
            # error to report back, not a crash.
            return ToolResult(content=f"{name} got an unusable value: {exc}", ok=False)
        except TypeError as exc:
            return ToolResult(
                content=f"{name} called with bad arguments: {exc}", ok=False
            )

    def _describe_image(self, path: str) -> ToolResult:
        """Real facts about an attached image, via PIL: size, format, palette.

        The chat model is text-only, so this returns what is verifiably true about the
        file plus its storage path; the harness (Builder) can be delegated for deeper
        visual analysis when a vision model is configured.
        """
        from PIL import Image

        source = Path(path).expanduser()
        if not source.is_absolute():
            source = self._cwd / source
        resolved = source.resolve()
        if not resolved.is_file():
            return ToolResult(content=f"image not found: {resolved}")
        try:
            with Image.open(resolved) as im:
                width, height = im.size
                fmt = im.format or "?"
                mode = im.mode
                small = im.convert("RGB").resize((4, 4))
                counts = small.getcolors(maxcolors=16) or []
                counts = sorted(counts, key=lambda item: -int(item[0]))[:3]
                palette_parts = []
                for item in counts:
                    from typing import Any, cast

                    rgb = cast(
                        Any, item[1]
                    )  # PIL getcolors stubs are imprecise; runtime value is (r,g,b)
                    red = int(rgb[0])
                    green = int(rgb[1])
                    blue = int(rgb[2])
                    palette_parts.append(
                        "#" + f"{red:02x}" + f"{green:02x}" + f"{blue:02x}"
                    )
                palette = ", ".join(palette_parts)
            return ToolResult(
                content=(
                    f"image: {resolved.name} | {fmt} {mode} {width}x{height}px | "
                    f"dominant colours: {palette} | path: {resolved}"
                )
            )
        except OSError as exc:
            return ToolResult(content=f"could not open image: {exc}")

    def _read_file(self, path: str, query: str = "") -> ToolResult:
        """Read a document and return text with citations that can be checked."""
        try:
            document = filereader.read_document(path, base_dir=self._cwd)
        except filereader.FileError as exc:
            raise ToolError(str(exc)) from exc
        header = f"{document.path.name}: {document.kind}, {document.chars} chars"
        if query.strip():
            hits = filereader.search(document, query, limit=4)
            if hits:
                body = "\n\n".join(
                    f"[{hit.label}]\n{hit.text.strip()[:900]}" for hit in hits
                )
                return ToolResult(
                    content=f"{header}. Best matches for {query!r}:\n\n{body}"
                )
        labels = ", ".join(citation.label for citation in document.citations[:4])
        return ToolResult(
            content=f"{header}; sections: {labels}\n\n{document.excerpt(4000)}"
        )

    def _remember(self, fact: str) -> ToolResult:
        """Save a durable fact to MEMORY.md - survives restarts.

        M4.1: the line carries provenance - when the fact was learned and by
        which session ("- [stamp | session] fact") - so the inspector can show
        where a memory came from.
        """
        mem = Path(__file__).resolve().parent / "MEMORY.md"
        stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.session_tag:
            stamp += " | " + self.session_tag
        try:
            with open(mem, "a", encoding="utf-8") as fh:
                fh.write("- [" + stamp + "] " + str(fact)[:300] + "\n")
        except OSError as exc:
            raise ToolError(f"could not write memory: {exc}") from exc
        return ToolResult(content="remembered: " + str(fact)[:80])

    def _recall(self, query: str = "") -> ToolResult:
        """Read MEMORY.md back, ranked when a query is given.

        W2.4 (D-7): recall is scored, not flat - semantic match (the query words
        in the fact), recency (newer facts score higher on a 30-day half-life)
        and importance (wording signals: dates, names, numbers, "important",
        "always", "never", "prefer"). The provenance stamp travels with the fact
        so the inspector can show which session learned it.
        """
        mem = Path(__file__).resolve().parent / "MEMORY.md"
        if not mem.is_file():
            return ToolResult(content="memory is empty")
        try:
            lines = mem.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ToolError(f"could not read memory: {exc}") from exc
        rows = [(line, self._fact_score(line, query)) for line in lines if line.strip()]
        if query:
            hits = [(line, score) for line, score in rows if score["match"] > 0]
            hits.sort(key=lambda pair: pair[1]["total"], reverse=True)
            if not hits:
                return ToolResult(content="nothing matches " + query)
            body = "\n".join(
                f"[{score['total']:.2f}] {line}" for line, score in hits[:10]
            )
            return ToolResult(content=body)
        rows.sort(key=lambda pair: pair[1]["recency"], reverse=True)
        return ToolResult(
            content=chr(10).join(line for line, _ in rows[:12]) or "memory is empty"
        )

    def relevant_facts(self, text: str, limit: int = 3) -> str:
        """M4.3: the top scored memories for this utterance, ready to inject.

        Recall normally costs a tool round-trip; when one of the stored facts
        actually matches what the user just said, it rides along in the system
        prompt instead. Returns "" when nothing matches, so the prompt stays
        clean on ordinary turns.
        """
        mem = Path(__file__).resolve().parent / "MEMORY.md"
        if not mem.is_file() or not str(text).strip():
            return ""
        try:
            lines = mem.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        scored = [
            (line.strip(), self._fact_score(line, text))
            for line in lines
            if line.strip().startswith("- [")
        ]
        hits = [pair for pair in scored if pair[1]["match"] > 0]
        hits.sort(key=lambda pair: pair[1]["total"], reverse=True)
        if not hits:
            return ""
        body = "\n".join("- " + fact for fact, _ in hits[:limit])
        return (
            "Memories from earlier sessions that may help (use them naturally "
            "if they fit; do not mention this note):\n" + body
        )

    @staticmethod
    def _fact_score(line: str, query: str) -> dict[str, float]:
        """Semantic match + recency + importance for one MEMORY.md row."""
        import re as _re

        text = line.lower()
        now = datetime.now().timestamp()
        stamp_match = _re.search(r"\[(\d{4}-\d{2}-\d{2})", line)
        age_days = 30.0
        if stamp_match:
            try:
                stamp = datetime.strptime(stamp_match.group(1), "%Y-%m-%d")
                age_days = max(0.0, (now - stamp.timestamp()) / 86400.0)
            except ValueError:
                pass
        recency = pow(0.5, age_days / 30.0)  # 30-day half-life
        words = set(_re.findall(r"[a-z0-9']+", text))
        wanted = set(_re.findall(r"[a-z0-9']+", str(query).lower()))
        overlap = len(words & wanted - {"the", "a", "is", "my", "i", "of", "and"})
        denom = max(1, len(wanted))
        match = min(1.0, overlap / denom)
        important_markers = ("important", "always", "never", "prefer", "remember")
        has_numbers = bool(_re.search(r"\d", text))
        importance = 0.7
        if any(m in text for m in important_markers):
            importance += 0.2
        if has_numbers:
            importance += 0.1
        return {
            "match": match,
            "recency": recency,
            "importance": min(1.0, importance),
            "total": match * 0.6 + recency * 0.2 + importance * 0.2,
        }

    def _template(self, name: str, args: str = "") -> ToolResult:
        """Run a prompt template from prompts/<name>.md."""
        path = Path(__file__).resolve().parent / "prompts" / (name + ".md")
        if not path.is_file():
            avail = ", ".join(x.stem for x in path.parent.glob("*.md")) or "none"
            return ToolResult(content="template not found; available: " + avail)
        return ToolResult(content=path.read_text(encoding="utf-8"))

    # --------------------------------------------- external APIs (#23)
    def _registered_apis(self) -> dict[str, dict[str, Any]]:
        """Owner-editable registry (me3_laptop/apis.json), like personas.json:
        name -> {"url": "https://...", "description": "..."}."""
        path = Path(__file__).resolve().parent / "apis.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _list_apis(self) -> ToolResult:
        """List the registered APIs the agent can call by name (#23)."""
        registry = self._registered_apis()
        if not registry:
            return ToolResult(
                content="no APIs registered; add one to me3_laptop/apis.json, e.g. "
                '{ "wiki_search": { "url": "https://en.wikipedia.org/w/api.php", '
                '"description": "Wikipedia search" } }'
            )
        rows = [
            f"{name}: {str(entry.get('description', ''))[:70]} | {entry.get('url', '')}"
            for name, entry in sorted(registry.items())
        ]
        return ToolResult(content=f"{len(rows)} registered API(s):\n" + "\n".join(rows))

    def _register_api(
        self, name: str, url: str, description: str = "", token: str = ""
    ) -> ToolResult:
        """Register an external public API in sonj3/apis.json."""
        key = str(name).strip()
        endpoint = str(url).strip()
        if not key:
            raise ToolError("API name is required")
        if not endpoint.lower().startswith(("http://", "https://")):
            return ToolResult(
                content=f"refusing non-http URL: {endpoint[:80]}", ok=False
            )
        if not chain_is_public([endpoint]):
            return ToolResult(
                content=f"endpoint rejected (not publicly reachable or blocked): {endpoint[:80]}",
                ok=False,
            )
        path = Path(__file__).resolve().parent / "apis.json"
        registry = self._registered_apis()
        entry: dict[str, Any] = {
            "url": endpoint,
            "description": str(description or f"User-registered API ({key})")[:160],
        }
        if str(token).strip():
            entry["token"] = str(token).strip()
        registry[key] = entry
        try:
            path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            return ToolResult(content=f"could not save API: {exc}", ok=False)
        return ToolResult(content=f"API '{key}' registered ({endpoint}).")

    def _call_api(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
        headers: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a registered API by name with query parameters or headers (#23).

        Same guard as read_page: only public http(s) endpoints, so a registry
        typo can never turn the agent into an internal-network probe.
        """
        registry = self._registered_apis()
        entry = registry.get(str(name))
        if not entry:
            avail = ", ".join(sorted(registry)) or "none"
            return ToolResult(
                content=f"API '{name}' is not registered; available: {avail}"
            )
        url = str(entry.get("url", ""))
        if not url.lower().startswith(("http://", "https://")) or not chain_is_public(
            [url]
        ):
            return ToolResult(
                content=f"refusing non-public endpoint: {url[:80]}", ok=False
            )
        query = {str(k): str(v) for k, v in dict(params or {}).items()}
        req_headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        if entry.get("token") and "Authorization" not in req_headers:
            req_headers["Authorization"] = f"Bearer {entry['token']}"
        verb = str(method or "GET").upper()
        try:
            if verb == "POST":
                response = requests.post(
                    url,
                    json=query or None,
                    headers=req_headers,
                    timeout=WEATHER_TIMEOUT_S,
                )
            else:
                response = requests.get(
                    url, params=query, headers=req_headers, timeout=WEATHER_TIMEOUT_S
                )
            response.raise_for_status()
            body = response.text[:4000]
        except requests.RequestException as exc:
            return ToolResult(content=f"API call failed: {exc}", ok=False)
        return ToolResult(content=body or "(empty response)")

    def _browse_page(self, url: str, max_chars: int = 4000) -> ToolResult:
        """Visit a public URL using browser_mcp (Playwright) so JS pages work too."""
        target = str(url).strip()
        if not target.lower().startswith(
            ("http://", "https://")
        ) or not chain_is_public([target]):
            return ToolResult(
                content=f"refusing non-public URL: {target[:80]}", ok=False
            )
        script = Path(__file__).resolve().parent / "browser_mcp.py"
        if not script.is_file():
            return self._read_page(target, max_chars=max_chars)
        try:
            from mcp_client import McpClient

            with McpClient(
                "browser-mcp", [sys.executable, str(script)], timeout=30.0
            ) as client:
                nav = client.call("navigate", {"url": target})
                page = client.call("read_page", {})
            nav_txt = ((nav.get("content") or [{}])[0]).get("text", "")
            body = ((page.get("content") or [{}])[0]).get("text", "")
            limit = max(200, min(12000, int(max_chars)))
            text = (nav_txt + "\n\n" + body).strip()[:limit]
            return ToolResult(content=text or "Page opened but had no readable text.")
        except Exception:  # noqa: BLE001
            return self._read_page(target, max_chars=max_chars)

    # ------------------------------------------ approval gate (#21/#30)
    def _send_email(self, to: str, subject: str, body: str) -> ToolResult:
        """Send an email (#21) - queued behind the owner's approval (#30).

        Nothing sends unattended: the action lands in the approval queue and the
        owner answers it from the UI (Approve/Deny) or POST /api/approval.
        """
        if self._on_approval is None:
            raise ToolError("the approval gate is not wired in this build")
        approval_id = self._on_approval(
            "send_email", f"to={to} subject={str(subject)[:60]} body={str(body)[:200]}"
        )
        return ToolResult(
            content=f"email to {to} is queued as approval #{approval_id}; "
            "the owner must approve it before anything is sent"
        )

    def _run_code(self, code: str, timeout_s: float = 20.0) -> ToolResult:
        """Execute a Python snippet in a sandboxed subprocess (feature b).

        The agent can already read and write files; this closes the loop by letting
        it RUN what it wrote. Guards: no shell is involved (the code travels as one
        argument to python -c), the working directory is out/sandbox, the wall
        clock is capped at 20 s, and the output is capped. This runs on the
        owner's machine at the owner's request - the same trust level as the
        owner typing python themselves.
        """
        snippet = str(code).strip()
        if not snippet:
            return ToolResult(content="run_code: the snippet was empty", ok=False)
        try:
            budget = min(20.0, max(1.0, float(timeout_s)))
        except (TypeError, ValueError):
            budget = 20.0
        base = Path(__file__).resolve().parent
        sandbox = base / "out" / "sandbox"
        sandbox.mkdir(parents=True, exist_ok=True)
        try:
            done = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=budget,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"run_code: killed after {budget:.0f}s (still running)",
                ok=False,
            )
        out = (done.stdout or "").strip()[:4000]
        err = (done.stderr or "").strip()[:2000]
        parts = [f"exit {done.returncode}"]
        if out:
            parts.append("stdout:\n" + out)
        if err:
            parts.append("stderr:\n" + err)
        if not out and not err:
            parts.append("(no output)")
        return ToolResult(content="\n".join(parts), ok=done.returncode == 0)

    # ---------------------------------------------- audio overview (#24)
    def _audio_overview(self, path: str) -> ToolResult:
        """Nova and Piattos discuss a file like a podcast (#24).

        Reads the file (citations and all), builds a NOVA:/PIATTOS: script from
        the text, and hands it to the voice layer, which speaks each line in that
        persona's own voice.
        """
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (self._cwd / target).resolve()
        if not target.is_file():
            return ToolResult(content=f"file not found: {target}")
        if self._on_speak_script is None:
            raise ToolError("the two-voice layer is not wired in this build")
        try:
            document = filereader.read_document(target)
        except filereader.FileError as exc:
            return ToolResult(content=f"could not read {target.name}: {exc}", ok=False)
        excerpt = " ".join(document.excerpt(700).split())[:700]
        part_one = excerpt[: len(excerpt) // 2]
        part_two = excerpt[len(excerpt) // 2 :]
        script = "\n".join(
            [
                "NOVA: Here is the audio overview of " + target.name + ".",
                "PIATTOS: What is in it?",
                "NOVA: " + part_one,
                "PIATTOS: And the rest?",
                "NOVA: " + part_two,
            ]
        )
        spoken = self._on_speak_script(script)
        return ToolResult(
            content=f"Nova and Piattos discussed {target.name}; {spoken} lines spoken"
        )

    # ------------------------------- delegation (#49, CrewAI pattern)
    def _delegate_work(
        self, task: str, context: str = "", coworker: str = ""
    ) -> ToolResult:
        """Hand a task to a coworker (#49): the background task layer runs it."""
        brief = str(task).strip()
        if str(context).strip():
            brief += " | context: " + str(context).strip()[:200]
        if str(coworker).strip():
            brief += " | for: " + str(coworker).strip()[:40]
        return self._start_task(brief)

    def _ask_question(
        self, question: str, context: str = "", coworker: str = ""
    ) -> ToolResult:
        """Ask a coworker a question instead of guessing (#49).

        Synchronous on purpose: the answer is needed inside this turn, so it goes
        to the agent harness (its own tools, its own model) rather than a
        background thread.
        """
        brief = "Answer this question precisely: " + str(question).strip()
        if str(context).strip():
            brief += " | context: " + str(context).strip()[:200]
        if str(coworker).strip():
            brief += " | for: " + str(coworker).strip()[:40]
        return self._delegate_task(brief)

    # ------------------------------------------- agent-to-agent (A2A, one turn)
    def _list_agents(self) -> ToolResult:
        """Who is on the roster and which model carries each role (routing is data)."""
        path = Path(__file__).resolve().parent / "agents.json"
        try:
            roster = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return ToolResult(content=f"could not read the roster: {exc}", ok=False)
        if not isinstance(roster, dict) or not roster:
            return ToolResult(content="the roster is empty (agents.json)")
        lines = [
            f"{role}: {(cfg or {}).get('model') or '-'}"
            for role, cfg in sorted(roster.items())
        ]
        return ToolResult(content="Agent roster:\n" + "\n".join(lines))

    def _ask_agent(self, role: str, question: str, context: str = "") -> ToolResult:
        """Consult one roster role and return its answer inside this turn.

        Synchronous on purpose: the front agent needs the answer to compose its own
        reply, and the consult carries only the question plus a short context - not
        the conversation - so it stays cheap and the specialist stays focused.
        """
        if self._on_ask_agent is None:
            raise ToolError("the agent roster is not wired in this build")
        wanted = str(role).strip()
        asked = str(question).strip()
        if not wanted:
            raise ToolError("which agent? pass a role name, e.g. researcher")
        if not asked:
            raise ToolError("the question is empty")
        answer = str(self._on_ask_agent(wanted, asked, str(context).strip()))
        return ToolResult(content=answer[:4000])

    def _ask_jev(self, task: str) -> ToolResult:
        """Ask the Jev routing delegate which specialist is best for a task."""
        if self._on_ask_agent is None:
            raise ToolError("the agent roster is not wired in this build")
        asked = str(task).strip()
        if not asked:
            raise ToolError("the task is empty")
        answer = str(
            self._on_ask_agent(
                "jev", asked, "Pick the best agent for this task from the roster."
            )
        )
        return ToolResult(content=answer[:2000])

    # --------------------------------------- memory hygiene (#47/#62)
    def _forget(self, fact: str) -> ToolResult:
        """Remove matching facts from MEMORY.md (#47/#62)."""
        mem = Path(__file__).resolve().parent / "MEMORY.md"
        needle = str(fact).strip().lower()
        if not mem.is_file() or not needle:
            return ToolResult(content="memory is empty")
        try:
            lines = mem.read_text(encoding="utf-8").splitlines()
            kept = [row for row in lines if needle not in row.lower()]
            removed = len(lines) - len(kept)
            mem.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"could not update memory: {exc}") from exc
        return ToolResult(content=f"forgot {removed} fact(s) matching '{fact}'")

    def _cd(self, folder: str) -> ToolResult:
        """Change the working directory for subsequent file tools, like a terminal cd."""
        try:
            candidate = Path(folder).expanduser()
            if not candidate.is_absolute():
                candidate = (self._cwd / candidate).resolve()
            resolved = candidate.resolve()
            if not resolved.is_dir():
                return ToolResult(content=f"not a directory: {resolved}")
            self._cwd = resolved
            return ToolResult(content=f"cwd: {self._cwd}")
        except OSError as exc:
            return ToolResult(content=f"cd failed: {exc}")

    def _pwd(self) -> ToolResult:
        """Where the file tools currently are."""
        return ToolResult(content=f"cwd: {self._cwd}")

    def _tree(self, depth: str = "2") -> ToolResult:
        """A compact directory tree, like a terminal agent's folder view."""
        max_depth = max(1, min(4, int(depth) if str(depth).isdigit() else 2))
        lines: list[str] = []
        base = self._cwd

        def walk(folder: Path, prefix: str, level: int) -> None:
            try:
                entries = sorted(
                    folder.iterdir(), key=lambda e: (e.is_file(), e.name.lower())
                )
            except OSError:
                return
            for shown, entry in enumerate(entries):
                if shown >= 20:
                    lines.append(prefix + "... (more)")
                    break
                name = entry.name + ("/" if entry.is_dir() else "")
                lines.append(prefix + name)
                if (
                    entry.is_dir()
                    and not entry.name.startswith(".")
                    and level < max_depth
                ):
                    walk(entry, prefix + "  ", level + 1)

        walk(base, "", 1)
        return ToolResult(content=f"{base}" + chr(10) + chr(10).join(lines[:60]))

    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write a text/code file inside the allowed workspace."""
        raw = str(path).strip()
        if not raw:
            raise ToolError("path is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (self._cwd / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not filereader.is_allowed(candidate):
            raise ToolError(f"not allowed to write outside workspace: {candidate}")
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(str(content), encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"could not write {candidate.name}: {exc}") from exc
        return ToolResult(content=f"wrote {candidate} ({len(str(content))} chars)")

    def _list_files(self, folder: str = ".", pattern: str = "*") -> ToolResult:
        """List readable files and subfolders relative to current working directory."""
        try:
            rows = filereader.list_folder(
                folder, pattern, base_dir=self._cwd, include_dirs=True
            )
        except filereader.FileError as exc:
            raise ToolError(str(exc)) from exc
        if not rows:
            return ToolResult(
                content=f"no readable files in {folder} matching {pattern}"
            )
        lines = [
            f"{row['name']} ({row['kind']})"
            if row["kind"] == "dir"
            else f"{row['name']} ({row['kind']}, {row['mb']} MB)"
            for row in rows[:40]
        ]
        return ToolResult(
            content=f"{len(rows)} entries in {folder}:\n" + "\n".join(lines)
        )

    def _start_crew(
        self, task: str, roles: list[str], process: str = "sequential"
    ) -> ToolResult:
        """W2.1: run a crew of agents on a task (#49 swarm handoffs, real).

        roles come from agents.json; the engine starts the crew and each step
        runs on that role's model. The result is reported to the chat when the
        crew finishes.
        """
        if self._on_crew is None:
            raise ToolError("the crew layer is not wired in this build")
        clean_roles = [str(r).strip() for r in (roles or []) if str(r).strip()]
        if not clean_roles:
            raise ToolError("pick at least one role from agents.json")
        if str(process).lower() not in {"sequential", "hierarchical"}:
            raise ToolError("process must be sequential or hierarchical")
        try:
            message = self._on_crew(str(task), clean_roles, str(process).lower())
        except Exception as exc:  # the caller gets a tool error, not a crash
            raise ToolError(f"could not start the crew: {exc}") from exc
        return ToolResult(content=message)

    def _start_task(self, brief: str, model: str = "") -> ToolResult:
        """Hand work to the background task layer and return something sayable."""
        if self._on_task is None:
            raise ToolError("background tasks are not available in this build")
        try:
            message = self._on_task(brief, model)
        except Exception as exc:  # the caller gets a tool error, not a crash
            raise ToolError(f"could not start the task: {exc}") from exc
        return ToolResult(content=message)

    def _delegate_task(self, task: str, tools: str = "") -> ToolResult:
        """Run the agent harness on an open-ended task and return its answer.

        This is what makes the harness load-bearing instead of decorative: work the
        fixed tools cannot do leaves this process and is done by the agent harness,
        which brings its own tools and its own MCP servers.
        """
        task = task.strip()
        if not task:
            raise ToolError("the task is empty")
        command = harness_command(task, tools)
        if command is None:
            raise ToolError(
                "no agent harness available: set ME3_HARNESS_CMD, or ME3_HARNESS_SSH "
                "to run it on the Raspberry Pi"
            )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=HARNESS_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"the harness did not finish within {HARNESS_TIMEOUT_S:.0f} seconds"
            ) from exc
        elapsed = time.perf_counter() - started
        answer = (completed.stdout or "").strip()
        if not answer:
            detail = (completed.stderr or "").strip()[:200]
            raise ToolError(f"the harness returned nothing ({detail or 'no output'})")
        print(f"[harness] task finished in {elapsed:.1f}s via {command[0]}")
        return ToolResult(content=answer[:MAX_HARNESS_CHARS])

    def shutdown(self) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    def _get_time(self, timezone: str = "local") -> ToolResult:
        if timezone and timezone.lower() != "local":
            try:
                now = datetime.now(ZoneInfo(timezone))
            except ZoneInfoNotFoundError as exc:
                raise ToolError(f"unknown timezone '{timezone}'") from exc
        else:
            now = datetime.now()
        stamp = now.strftime("%A %d %B %Y, %H:%M")
        zone = now.tzname() or ""
        return ToolResult(content=f"{stamp} {zone}".strip())

    def _set_timer(self, seconds: int, label: str = "timer") -> ToolResult:
        duration = int(seconds)
        if duration <= 0:
            raise ToolError("duration must be positive")
        if duration > MAX_TIMER_SECONDS:
            raise ToolError("duration is longer than 24 hours")

        timer = threading.Timer(duration, self._fire_timer, args=(label,))
        timer.daemon = True
        timer.start()
        self._timers.append(timer)

        minutes, secs = divmod(duration, 60)
        human = f"{minutes} min {secs} s" if minutes else f"{secs} s"
        return ToolResult(content=f"Timer '{label}' started for {human}.")

    def _fire_timer(self, label: str) -> None:
        if self._on_timer is not None:
            self._on_timer(f"Your {label} is done.")

    def _set_lights(
        self, room: str, state: str, brightness: int | None = None
    ) -> ToolResult:
        wanted = str(state).lower()
        if wanted not in {"on", "off"}:
            raise ToolError("state must be 'on' or 'off'")
        level = 100 if wanted == "on" and brightness is None else brightness
        if wanted == "on" and level is not None and not 0 <= int(level) <= 100:
            raise ToolError("brightness must be between 0 and 100")

        self._lights[room] = {"state": wanted, "brightness": level}
        if wanted == "off":
            return ToolResult(content=f"{room} light is now off.")
        return ToolResult(content=f"{room} light is now on at {level}% brightness.")

    def _get_weather(self, city: str) -> ToolResult:
        if not city.strip():
            raise ToolError("city is required")
        # Host is a constant; the only variable part is the quoted city path.
        url = f"https://wttr.in/{quote(city, safe='')}?format=j1"
        try:
            response = requests.get(url, timeout=WEATHER_TIMEOUT_S)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return ToolResult(
                content=f"Weather lookup is unavailable right now ({exc}).", ok=False
            )
        except ValueError as exc:
            return ToolResult(
                content=f"Weather service returned junk ({exc}).", ok=False
            )

        current = (payload.get("current_condition") or [{}])[0]
        description = ((current.get("weatherDesc") or [{}])[0]).get("value", "unknown")
        temp = current.get("temp_C", "?")
        feels = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        return ToolResult(
            content=f"{city}: {description}, {temp}C (feels like {feels}C), humidity {humidity}%."
        )

    def _get_stock_price(self, symbol: str) -> ToolResult:
        """Fetch live stock price by ticker, company name, or PH subsidiary (Yahoo + PSE Phisix + web)."""
        raw = " ".join(str(symbol or "").split()).strip()
        if not raw:
            raise ToolError("symbol is required")
        low = raw.lower()
        ph_aliases: dict[str, tuple[str, str]] = {
            "gcash": (
                "GLO",
                "GCash (Mynt) is unlisted; parent Globe Telecom Inc. (PSE: GLO)",
            ),
            "mynt": (
                "GLO",
                "Mynt (GCash) is unlisted; parent Globe Telecom Inc. (PSE: GLO)",
            ),
            "globe": ("GLO", "Globe Telecom, Inc. (PSE: GLO)"),
            "globe telecom": ("GLO", "Globe Telecom, Inc. (PSE: GLO)"),
            "smart": (
                "TEL",
                "Smart Communications is unlisted; parent PLDT Inc. (PSE: TEL)",
            ),
            "smart communications": (
                "TEL",
                "Smart Communications is unlisted; parent PLDT Inc. (PSE: TEL)",
            ),
            "pldt": ("TEL", "PLDT Inc. (PSE: TEL)"),
            "ayala": ("AC", "Ayala Corporation (PSE: AC)"),
            "jollibee": ("JFC", "Jollibee Foods Corporation (PSE: JFC)"),
            "bdo": ("BDO", "BDO Unibank, Inc. (PSE: BDO)"),
            "bpi": ("BPI", "Bank of the Philippine Islands (PSE: BPI)"),
            "meralco": ("MER", "Manila Electric Company (PSE: MER)"),
            "sm": ("SM", "SM Investments Corporation (PSE: SM)"),
        }
        pse_sym = ""
        note = ""
        if low.endswith(".ps") or low.startswith("pse:"):
            pse_sym = (
                re.sub(r"^(pse:)|(\.ps)$", "", low, flags=re.IGNORECASE).strip().upper()
            )
            note = f"PSE: {pse_sym}"
        else:
            matched_ph: list[tuple[str, str]] = []
            seen_codes: set[str] = set()
            words_set = set(re.findall(r"[a-z0-9]+", low))
            for k, (sym_code, desc) in ph_aliases.items():
                if (
                    low == k or (" " in k and k in low) or k in words_set
                ) and sym_code not in seen_codes:
                    seen_codes.add(sym_code)
                    matched_ph.append((sym_code, desc))
            if len(matched_ph) > 1:
                parts_out: list[str] = []
                for sym_code, desc in matched_ph:
                    sub = self._get_stock_price(f"{sym_code}.PS")
                    if sub.ok:
                        parts_out.append(f"{desc}: {sub.content.split(': ', 1)[-1]}")
                if parts_out:
                    return ToolResult(content=" | ".join(parts_out))
            if matched_ph:
                pse_sym, note = matched_ph[0]
        hdrs = {"User-Agent": "Mozilla/5.0"}
        if pse_sym:
            try:
                presp = requests.get(
                    f"https://phisix-api3.appspot.com/stocks/{quote(pse_sym)}.json",
                    headers=hdrs,
                    timeout=WEATHER_TIMEOUT_S,
                )
                presp.raise_for_status()
                payload_ps = presp.json()
                stocks = payload_ps.get("stocks") or payload_ps.get("stock") or []
                if stocks:
                    st = stocks[0]
                    p_info = st.get("price") or {}
                    amt = p_info.get("amount")
                    cur = p_info.get("currency") or "PHP"
                    pct = (
                        st.get("percentChange")
                        if st.get("percentChange") is not None
                        else st.get("percent_change")
                    )
                    chg = (
                        f" ({'+' if float(pct) >= 0 else ''}{float(pct):.2f}%)"
                        if pct is not None
                        else ""
                    )
                    if amt is not None:
                        return ToolResult(
                            content=f"{note}: {float(amt):,.2f} {cur}{chg}."
                        )
            except (requests.RequestException, ValueError):
                pass
        ticker = raw.upper() if re.fullmatch(r"[A-Za-z0-9.\-^]{1,10}", raw) else ""
        if not ticker:
            try:
                sresp = requests.get(
                    f"https://query1.finance.yahoo.com/v1/finance/search?q={quote(raw)}&quotesCount=5&newsCount=0",
                    headers=hdrs,
                    timeout=WEATHER_TIMEOUT_S,
                )
                sresp.raise_for_status()
                quotes = sresp.json().get("quotes") or []
                eq = next(
                    (
                        q
                        for q in quotes
                        if q.get("quoteType") in {"EQUITY", "ETF", "INDEX"}
                    ),
                    quotes[0] if quotes else None,
                )
                if eq and eq.get("symbol"):
                    ticker = str(eq["symbol"])
            except (requests.RequestException, ValueError):
                ticker = ""
        if ticker:
            try:
                cresp = requests.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='^.-')}?interval=1d&range=1d",
                    headers=hdrs,
                    timeout=WEATHER_TIMEOUT_S,
                )
                cresp.raise_for_status()
                res = (cresp.json().get("chart") or {}).get("result") or []
                if res:
                    meta = res[0].get("meta") or {}
                    price = meta.get("regularMarketPrice")
                    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
                    cur = meta.get("currency") or "USD"
                    name = meta.get("shortName") or meta.get("longName") or ticker
                    if price is not None:
                        chg_txt = ""
                        if isinstance(prev, (int, float)) and prev:
                            diff = float(price) - float(prev)
                            pct = (diff / float(prev)) * 100.0
                            sign = "+" if diff >= 0 else ""
                            chg_txt = f" ({sign}{diff:.2f}, {sign}{pct:.2f}%)"
                        return ToolResult(
                            content=f"{name} ({ticker}): {float(price):.2f} {cur}{chg_txt}."
                        )
            except (requests.RequestException, ValueError):
                pass
        return self._web_search(f"{raw} stock price today", recency="day")

    def _web_search(self, query: str, recency: str = "any") -> ToolResult:
        """Search the web, preferring recent pages when asked for news."""
        window = "" if str(recency).lower() in {"", "any", "none"} else str(recency)
        try:
            hits, provider = web_search(
                query,
                max_results=3,
                brave_key=os.environ.get("BRAVE_API_KEY", ""),
                timelimit=window,
            )
        except (WebSearchError, requests.RequestException) as exc:
            return ToolResult(
                content=f"Web search is unavailable right now ({exc}).", ok=False
            )
        if not hits:
            return ToolResult(content=f"No web results for '{query}'.", ok=False)
        return ToolResult(content=format_results(hits, provider))

    def _read_page(self, url: str, max_chars: int = 4000) -> ToolResult:
        """Return readable text from one public web page."""
        try:
            text = fetch_page(url, max_chars=int(max_chars))
        except (WebSearchError, requests.RequestException) as exc:
            return ToolResult(
                content=(
                    f"That page could not be read ({exc}). Some sites block automated "
                    "readers; answer from the search snippets instead."
                ),
                ok=False,
            )
        return ToolResult(content=text or "That page had no readable text.")

    def _set_alarm(self, time: str, label: str = "alarm") -> ToolResult:
        wanted = str(time).strip()
        if not wanted:
            raise ToolError("time is required")
        self._device["alarm"] = {"time": wanted, "label": label}
        return ToolResult(content=f"Alarm set for {wanted} ({label}).")

    def _adjust_lights(
        self, room: str, direction: str, percent: int = 20
    ) -> ToolResult:
        want = str(direction).lower()
        if want not in {"dim", "brighten"}:
            raise ToolError("direction must be 'dim' or 'brighten'")
        step = int(percent)
        if step <= 0:
            raise ToolError("percent must be positive")
        current = int(self._lights.get(room, {}).get("brightness") or 100)
        level = max(0, current - step) if want == "dim" else min(100, current + step)
        self._lights[room] = {
            "state": "on" if level > 0 else "off",
            "brightness": level,
        }
        if level == 0:
            return ToolResult(content=f"{room} light is now off.")
        return ToolResult(content=f"{room} light is now at {level}% brightness.")

    def _adjust_temperature(self, direction: str, degrees: float = 1) -> ToolResult:
        want = str(direction).lower()
        if want not in {"warmer", "cooler"}:
            raise ToolError("direction must be 'warmer' or 'cooler'")
        delta = abs(float(degrees))
        current = float(self._device["temperature"])
        self._device["temperature"] = (
            current + delta if want == "warmer" else current - delta
        )
        return ToolResult(
            content=f"Thermostat set to {self._device['temperature']:.1f} degrees C."
        )

    def _media_control(self, action: str) -> ToolResult:
        want = str(action).lower()
        mapping = {
            "play": "playing",
            "pause": "paused",
            "next": "playing",
            "previous": "playing",
        }
        if want not in mapping:
            raise ToolError("action must be play, pause, next or previous")
        self._device["media"] = mapping[want]
        phrases = {
            "play": "Playing.",
            "pause": "Paused.",
            "next": "Skipped to the next track.",
            "previous": "Went back to the previous track.",
        }
        return ToolResult(content=phrases[want])

    def _adjust_volume(self, direction: str) -> ToolResult:
        want = str(direction).lower()
        if want not in {"up", "down"}:
            raise ToolError("direction must be 'up' or 'down'")
        level = int(self._device["volume"]) + (10 if want == "up" else -10)
        self._device["volume"] = max(0, min(100, level))
        return ToolResult(content=f"Volume is now {self._device['volume']} percent.")

    def _set_reminder(self, text: str, when: str = "") -> ToolResult:
        if not str(text).strip():
            raise ToolError("text is required")
        self._reminders.append({"text": str(text).strip(), "when": str(when).strip()})
        tail = f" for {when}" if when else ""
        return ToolResult(content=f"Reminder saved{tail}: {text}.")

    def _list_reminders(self) -> ToolResult:
        if not self._reminders:
            return ToolResult(content="There are no reminders.")
        lines = [
            f"{index}. {item['text']}" + (f" ({item['when']})" if item["when"] else "")
            for index, item in enumerate(self._reminders, start=1)
        ]
        return ToolResult(content="Reminders:" + chr(10) + chr(10).join(lines))

    def _make_call(self, contact: str) -> ToolResult:
        if not str(contact).strip():
            raise ToolError("contact is required")
        return ToolResult(
            content=f"Calling {contact}. (simulated - no telephony on this device)"
        )

    def _silence(self) -> ToolResult:
        self._device["media"] = "paused"
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
        return ToolResult(content="Stopped speaking and cleared active timers.")

    def _calculate(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"could not parse '{expression}'") from exc
        value = _eval_node(tree.body)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return ToolResult(content=f"{expression} = {value}")


def _function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
):
    """Build one function schema in the shared OpenAI/Ollama shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


def _eval_node(node: ast.AST) -> float:
    """Evaluate a tiny arithmetic subset of Python - no attribute or name escapes."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError("only numbers are allowed")
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError("unsupported operator")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError("unsupported unary operator")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _MATH_CONSTANTS:
            return _MATH_CONSTANTS[node.id]
        raise ToolError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCTIONS:
            raise ToolError("unsupported function")
        arguments = [_eval_node(arg) for arg in node.args]
        return float(_MATH_FUNCTIONS[node.func.id](*arguments))
    if isinstance(node, ast.Tuple):
        raise ToolError("tuples are not supported")
    raise ToolError("expression is too complex")


def tool_call_log(call_name: str, arguments: dict[str, Any], result: ToolResult) -> str:
    """One-line rendering used by the console log."""
    return f"{call_name}({json.dumps(arguments, default=str)}) -> {'ok' if result.ok else 'error'}"


def _check(box: ToolBox, title: str, want_ok: bool, name: str, **args: Any) -> bool:
    """Call one tool and report whether it behaved as intended.

    The title is separate from the tool arguments, because several tools take an
    argument literally called "label".
    """
    result = box.call(name, args)
    verdict = "PASS" if result.ok == want_ok else "FAIL"
    detail = result.content.replace(chr(10), " ")[:90]
    print(f"  [{verdict}] {title:32s} ok={result.ok} :: {detail}")
    return result.ok == want_ok


def _selftest() -> int:
    """Exercise every tool, including deliberately bad arguments."""
    box = ToolBox()
    print("tools:")
    checks = [
        _check(box, "get_time", True, "get_time"),
        _check(
            box,
            "set_lights on 40%",
            True,
            "set_lights",
            room="bedroom",
            state="on",
            brightness=40,
        ),
        _check(
            box,
            "set_lights bad state",
            False,
            "set_lights",
            room="bedroom",
            state="maybe",
        ),
        _check(box, "set_timer 5s", True, "set_timer", seconds=5, label="check"),
        _check(box, "set_timer negative", False, "set_timer", seconds=-1),
        _check(box, "set_timer junk seconds", False, "set_timer", seconds="soon"),
        _check(box, "calculate", True, "calculate", expression="(12 + 8) * 3"),
        _check(
            box, "calculate bad name", False, "calculate", expression="__import__('os')"
        ),
        _check(box, "unknown tool", False, "nope"),
        _check(
            box, "read_page LAN target", False, "read_page", url="http://192.168.1.1/"
        ),
        _check(
            box,
            "read_page junk max_chars",
            False,
            "read_page",
            url="https://example.com",
            max_chars="lots",
        ),
        _check(
            box, "web_search (live)", True, "web_search", query="Raspberry Pi 5 price"
        ),
    ]
    print(f"schemas exposed: {len(box.schemas())}")
    box.shutdown()
    ok = all(checks)
    print("RESULT:", "PASS" if ok else f"FAIL ({checks.count(False)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
