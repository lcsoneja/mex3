"""A minimal, dependency-free MCP server core (stdio transport).

Implements the subset of the Model Context Protocol a tool server needs:
initialize, tools/list, tools/call, ping and the initialized notification.
Messages are newline-delimited JSON-RPC 2.0 on stdin/stdout, which is what MCP
stdio servers speak.

Everything human-readable goes to stderr: stdout is the protocol channel, so a
stray print would corrupt the stream. Hand-rolling this keeps the Raspberry Pi
deployment free of extra packages, and the ME3 brief asks for these servers to
be mine.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


class ToolError(RuntimeError):
    """A tool failed in a way worth reporting back as an error result."""


def log(message: str) -> None:
    """Diagnostics for humans; never touches stdout."""
    print(message, file=sys.stderr, flush=True)


@dataclass
class Tool:
    """One callable tool with its JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def make_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    handler: Callable[..., str],
    required: list[str] | None = None,
) -> Tool:
    """Build a Tool with an object schema, like the OpenAI/Ollama function shape."""
    return Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
        handler=handler,
    )


class Server:
    """Holds tools and speaks the MCP stdio protocol."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, Tool] = {}

    def add(self, *tools: Tool) -> None:
        for tool in tools:
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------ protocol
    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one JSON-RPC request; None means it was a notification."""
        method = str(request.get("method") or "")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            return self._ok(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            )

        if method in {"notifications/initialized", "initialized"}:
            return None

        if method == "ping":
            return self._ok(request_id, {})

        if method == "tools/list":
            return self._ok(request_id, {"tools": [tool.schema for tool in self._tools.values()]})

        if method == "tools/call":
            return self._call_tool(request_id, params)

        return self._error(request_id, -32601, f"method not found: {method}")

    def _call_tool(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        tool = self._tools.get(name)
        if tool is None:
            return self._error(request_id, -32602, f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            return self._error(request_id, -32602, "arguments must be an object")

        log(f"tool call: {name}({json.dumps(arguments, default=str)})")
        try:
            text = tool.handler(**arguments)
        except ToolError as exc:
            return self._tool_result(request_id, f"{name} failed: {exc}", is_error=True)
        except TypeError as exc:
            return self._tool_result(request_id, f"{name} got bad arguments: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the server
            log(traceback.format_exc())
            return self._tool_result(
                request_id, f"{name} crashed: {type(exc).__name__}: {exc}", is_error=True
            )
        return self._tool_result(request_id, str(text), is_error=False)

    def serve(self, stdin=None, stdout=None) -> int:
        """Read requests until stdin closes."""
        source = stdin or sys.stdin
        sink = stdout or sys.stdout
        log(f"{self.name} ready: {len(self._tools)} tool(s)")
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except ValueError as exc:
                log(f"ignoring unparseable line: {exc}")
                continue
            if not isinstance(request, dict):
                log("ignoring non-object message")
                continue
            response = self.handle(request)
            if response is None:
                continue
            sink.write(json.dumps(response, default=str) + "\n")
            sink.flush()
        log(f"{self.name} stdin closed, exiting")
        return 0

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @classmethod
    def _tool_result(cls, request_id: Any, text: str, is_error: bool) -> dict[str, Any]:
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
        if is_error:
            result["isError"] = True
        return cls._ok(request_id, result)
