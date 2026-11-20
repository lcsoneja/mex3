"""A2A acceptance check: can the front agent message the other agents?

Proves, end to end and with the REAL models:
  1. the roster is readable as data (agents.json);
  2. ask_agent_turn consults one role and returns its answer inside the turn
     (this is what the front agent's ask_agent tool calls);
  3. the toolbox tools list_agents and ask_agent work through the same path the
     model uses, and a consult cannot act (it gets no tools);
  4. the agents MCP server speaks MCP and delivers the same capability to anything
     that can connect. It is driven with the repo's own client (mcp_client.py)
     against a temporary engine on a spare port, so the check is self-contained.

Text only; no audio is touched.
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import os
import sys
import threading
import time
import tkinter as tk
from http.server import ThreadingHTTPServer

sys.path.insert(0, ".")

import agents_mcp
from app_gui import VoiceAgentApp
from mcp_client import McpClient, McpProtocolError
from server import Session, Settings, _ConsultBox, _handler_factory, load_settings

PROBLEMS: list[str] = []
NOTES: list[str] = []


def note(text: str) -> None:
    NOTES.append(text)
    print("  ", text)


def check(condition: bool, message: str) -> None:
    if not condition:
        PROBLEMS.append(message)


def first_line(text: str) -> str:
    """The first line of a payload, or an explicit marker when it is empty."""
    lines = str(text).splitlines()
    return lines[0][:90] if lines else "(empty)"


def mcp_text(result: dict) -> str:
    """The text of an MCP tools/call result, or "" when it has none."""
    content = result.get("content")
    first = content[0] if isinstance(content, list) and content else {}
    return str(first.get("text") or "") if isinstance(first, dict) else ""


def mcp_call(client: McpClient, name: str, args: dict) -> str:
    """One MCP tool call: a failure becomes a recorded problem, not a crash."""
    try:
        return mcp_text(client.call(name, args))
    except (McpProtocolError, RuntimeError, OSError, ValueError) as exc:
        PROBLEMS.append(f"mcp {name} call failed: {type(exc).__name__}: {exc}")
        return ""


# --- 1. the roster is data ------------------------------------------------
try:
    roster_text = agents_mcp.list_agents()  # reads agents.json
except Exception as exc:
    PROBLEMS.append(f"could not read the roster: {type(exc).__name__}: {exc}")
    print("RESULT: FAIL")
    for problem in PROBLEMS:
        print("PROBLEM:", problem)
    raise SystemExit(1) from exc
note("roster: " + first_line(roster_text))
check("front:" in roster_text, f"roster does not name the front role: {roster_text[:120]}")

# --- 2/3. in-process: the tool the front agent actually calls --------------
try:
    # load_settings reads config.json and Session opens audio devices: a missing file
    # or a busy device must be a recorded failure, not an uncaught traceback.
    settings = load_settings()
    settings.tts_enabled = False  # a check must not speak
    session = Session(settings)
    root = tk.Tk()
    root.withdraw()
    app = VoiceAgentApp(root, session)
    root.update()
except Exception as exc:
    PROBLEMS.append(f"could not build the in-process session: {type(exc).__name__}: {exc}")
    print("RESULT: FAIL")
    for problem in PROBLEMS:
        print("PROBLEM:", problem)
    raise SystemExit(1) from exc

listed = session._toolbox.call("list_agents", {})
check(bool(listed.ok), f"list_agents tool failed: {listed.content[:160]}")
note("list_agents tool: " + first_line(str(listed.content)))

answer = session._toolbox.call(
    "ask_agent",
    {
        "role": "researcher",
        "question": "Reply with exactly one short sentence: what is 17 plus 25?",
        "context": "quick arithmetic",
    },
)
note(f"ask_agent tool ok={answer.ok}: {str(answer.content)[:120]!r}")
check(bool(answer.ok), f"ask_agent tool failed: {answer.content[:200]}")
check(len(str(answer.content)) > 3, "ask_agent returned nothing")

# the consult must not be able to act: no tools on the specialist's box
box = _ConsultBox()
check(box.schemas() == [], "the consult box exposes tools")
try:
    box.call("set_timer", {"seconds": 5})
    PROBLEMS.append("a consult was able to call a tool")
except RuntimeError:
    note("consult box: tools refused, as designed")
_ = app

try:
    session.stop_listening()
    root.destroy()
except Exception as exc:  # noqa: BLE001 - teardown may run after the window is gone
    PROBLEMS.append(f"in-process teardown failed: {type(exc).__name__}: {exc}")

# --- 4. the MCP server, driven with the repo's own client -----------------
port = 8899
temp_session: Session | None = None
httpd: ThreadingHTTPServer | None = None
try:
    # load_settings reads config.json and Session opens audio devices; a bad file or a
    # missing device must be a failed check, not an uncaught traceback.
    temp_settings: Settings = load_settings()
    temp_settings.tts_enabled = False
    temp_session = Session(temp_settings)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler_factory(temp_session))
except Exception as exc:
    PROBLEMS.append(
        f"could not start the temporary engine on port {port}: "
        f"{type(exc).__name__}: {exc}"
    )
    print("RESULT: FAIL")
    for problem in PROBLEMS:
        print("PROBLEM:", problem)
    raise SystemExit(1) from exc
assert httpd is not None and temp_session is not None  # the except above exits
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.6)

# The client spawns the server itself; the child inherits this env var, which is
# how it is pointed at the temporary engine instead of the usual port.
os.environ["ME3_ENGINE_URL"] = f"http://127.0.0.1:{port}"
# The client is a context manager: entering it spawns the server and runs the
# MCP initialize handshake, leaving it closes the child cleanly.
try:
    with McpClient("agents-mcp", [sys.executable, "agents_mcp.py"], timeout=120) as client:
        identity = client.server_info.get("serverInfo")
        server_name = str((identity or {}).get("name") or "?")
        note(f"mcp initialize: {server_name}")
        check(server_name == "me3-agents", f"unexpected mcp server name: {identity}")

        names = sorted(str(tool.get("name")) for tool in client.list_tools())
        note("mcp tools: " + ", ".join(names))
        check(names == ["ask_agent", "list_agents"], f"unexpected mcp tools: {names}")

        roster_payload = mcp_call(client, "list_agents", {})
        note("mcp list_agents: " + first_line(roster_payload))
        check(
            "critic:" in roster_payload,
            f"mcp list_agents missing roles: {roster_payload[:120]}",
        )

        remote_answer = mcp_call(
            client,
            "ask_agent",
            {"role": "summarizer", "question": "Answer in three words: name a primary colour."},
        )
        note(f"mcp ask_agent: {remote_answer[:120]!r}")
        check(
            len(remote_answer.strip()) > 2,
            f"mcp ask_agent returned nothing: {remote_answer[:160]}",
        )
except (McpProtocolError, RuntimeError, OSError, ValueError) as exc:
    # A server that will not speak MCP is a failed check, not a crashed check.
    PROBLEMS.append(f"the agents MCP server could not be driven: {type(exc).__name__}: {exc}")
finally:
    # The `with` block already closed the MCP child; only local cleanup is left.
    try:
        httpd.shutdown()
        temp_session.stop_listening()
    except Exception as exc:  # noqa: BLE001 - the temporary server may already be down
        PROBLEMS.append(f"temporary engine teardown failed: {type(exc).__name__}: {exc}")

print()
print("RESULT: " + ("FAIL" if PROBLEMS else "PASS"))
for problem in PROBLEMS:
    print("PROBLEM:", problem)
raise SystemExit(1 if PROBLEMS else 0)
