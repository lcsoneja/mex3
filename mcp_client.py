"""A small synchronous MCP client for newline-delimited stdio servers."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

BASE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "2024-11-05"


class McpProtocolError(RuntimeError):
    """The server returned an invalid or unsuccessful JSON-RPC response."""


class McpClient:
    """Manage one MCP server process and exchange JSON-RPC messages over stdio."""

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        timeout: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> None:
        if isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty sequence of arguments")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.name = name
        self.command = tuple(str(part) for part in command)
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=20)
        self._request_lock = threading.Lock()
        self._next_id = 0
        self._started = False
        self._closed = False
        self._failed = False
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.env = dict(env) if env else {}
        self.server_info: dict[str, Any] = {}

    @property
    def returncode(self) -> int | None:
        """Child exit code, or None while it is still running."""
        return self.process.poll() if self.process is not None else None

    def start(self) -> dict[str, Any]:
        """Spawn the server, perform initialize, and send initialized notification."""
        if self._started:
            return self.server_info
        if self._closed:
            raise RuntimeError(f"{self.name}: client is closed")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(BASE),
            env={**os.environ, **self.env} if self.env else None,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.server_info = self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "me3-mcp-client", "version": "0.1.0"},
                },
            )
            self._notify("notifications/initialized")
            self._started = True
            return self.server_info
        except Exception:
            self.close()
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tools advertised by the initialized server."""
        result = self._request("tools/list")
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpProtocolError(f"{self.name}: tools/list result has no tools array")
        return [tool for tool in tools if isinstance(tool, dict)]

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a named tool and return its MCP result object."""
        if args is not None and not isinstance(args, dict):
            raise TypeError("tool arguments must be a dictionary")
        return self._request("tools/call", {"name": name, "arguments": args or {}})

    def close(self, timeout: float = 5.0) -> bool:
        """Close stdin and wait for a clean server exit, escalating if it hangs."""
        if self.process is None:
            self._closed = True
            return True
        if self.process.poll() is None:
            if self.process.stdin is not None:
                with suppress(OSError):
                    self.process.stdin.close()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._closed = True
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        return self.process.returncode == 0

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._request_lock:
            process = self.process
            if process is None or self._closed or self._failed:
                raise RuntimeError(f"{self.name}: client is not available")
            if process.stdin is None:
                raise RuntimeError(f"{self.name}: stdin pipe is unavailable")
            self._next_id += 1
            request_id = self._next_id
            request: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                request["params"] = params
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._failed = True
                raise RuntimeError(
                    f"{self.name}: could not write to server: {self._stderr_tail()}"
                ) from exc
            try:
                line = self._responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                self._failed = True
                raise TimeoutError(
                    f"{self.name}: timed out waiting for {method}"
                ) from exc
            if line is None:
                self._failed = True
                raise RuntimeError(
                    f"{self.name}: server closed stdout: {self._stderr_tail()}"
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self._failed = True
                raise McpProtocolError(
                    f"{self.name}: non-JSON stdout: {line.strip()[:120]!r}"
                ) from exc
            if not isinstance(response, dict) or response.get("id") != request_id:
                self._failed = True
                raise McpProtocolError(
                    f"{self.name}: unexpected JSON-RPC response: {response!r}"
                )
            if "error" in response:
                error = response["error"]
                raise McpProtocolError(f"{self.name}: {method} failed: {error}")
            result = response.get("result")
            if not isinstance(result, dict):
                self._failed = True
                raise McpProtocolError(
                    f"{self.name}: {method} returned no result object"
                )
            return result

    def _notify(self, method: str) -> None:
        process = self.process
        if process is None or process.stdin is None or self._closed:
            raise RuntimeError(f"{self.name}: stdin pipe is unavailable")
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            self._responses.put(None)
            return
        try:
            for line in process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def _stderr_tail(self) -> str:
        return " | ".join(self._stderr)


def discover_servers() -> list[dict[str, Any]]:
    """Return spawn configs for the ASR and TTS MCP servers present beside this file."""
    servers: list[dict[str, Any]] = []
    for name, script in (("asr", "asr_mcp.py"), ("tts", "tts_mcp.py")):
        path = BASE / script
        if path.is_file():
            servers.append(
                {"name": name, "command": [sys.executable, str(path.resolve())]}
            )
    return servers
