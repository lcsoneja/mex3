"""ME3 agents MCP server: the roster, reachable as tools by other agents.

The front agent inside the app already has `ask_agent`; this server exposes the same
capability over MCP, so any MCP-capable harness, editor or script can list the roster
and send one agent a message, an instruction or a question - and read the answer back.

It talks to the local engine on 127.0.0.1 only (the same control-plane rule sonj.py
follows). Text only: no audio is read or written by this server.

Run standalone (stdio, speaks MCP):
    python me3_laptop/agents_mcp.py
"""

from __future__ import annotations

import json
import os
import urllib.request

from mcp_core import Server, ToolError, log, make_tool

# The engine's address. Localhost by default (the app's own rule); ME3_ENGINE_URL
# lets the Pi deployment or a test point at another port without editing code.
BASE = os.environ.get("ME3_ENGINE_URL", "http://127.0.0.1:8765").rstrip("/")
TIMEOUT_S = 120


def _request(path: str, body: dict | None = None) -> dict:
    """One call to the local engine; a refusal is a tool error, not a crash."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except Exception as exc:
        # The engine may simply be down; the caller gets a plain refusal.
        raise ToolError(
            f"the engine is not reachable on {BASE} ({type(exc).__name__}); "
            "start it with 'sonj' or 'python me3_laptop/server.py'"
        ) from exc
    if not isinstance(payload, dict):
        raise ToolError("the engine answered with something that is not JSON")
    return payload


def list_agents() -> str:
    """Every roster role and the model that carries it."""
    payload = _request("/api/roles")
    roles = payload.get("roles") or {}
    if not isinstance(roles, dict) or not roles:
        return "no roles in agents.json"
    lines = [
        f"{role}: {(cfg or {}).get('model') or '-'}"
        for role, cfg in sorted(roles.items())
    ]
    return "Agent roster:\n" + "\n".join(lines)


def ask_agent(role: str, question: str, context: str = "") -> str:
    """Send one message/instruction to a named agent and return its answer."""
    if not str(role).strip():
        raise ToolError("role is required (see list_agents)")
    if not str(question).strip():
        raise ToolError("question is required")
    payload = _request(
        "/api/ask_agent",
        {"role": role, "question": question, "context": context},
    )
    if not payload.get("ok"):
        raise ToolError(str(payload.get("error") or "the consult failed"))
    return str(payload.get("reply") or "")


def build_server() -> Server:
    server = Server("me3-agents")
    server.add(
        make_tool(
            "list_agents",
            "List the agent roster: every role and the model each one uses.",
            {},
            list_agents,
        ),
        make_tool(
            "ask_agent",
            (
                "Send a message, instruction or question to ONE agent on the roster "
                "and return its answer. The specialist answers on its own model and "
                "does not see your conversation, so include the facts it needs."
            ),
            {
                "role": {"type": "string", "description": "role name, e.g. researcher"},
                "question": {"type": "string", "description": "the message or question"},
                "context": {
                    "type": "string",
                    "description": "short background the agent needs (optional)",
                },
            },
            ask_agent,
            required=["role", "question"],
        ),
    )
    return server


def main() -> int:
    log("agents-mcp tools: list_agents, ask_agent")
    return build_server().serve()


if __name__ == "__main__":
    raise SystemExit(main())
