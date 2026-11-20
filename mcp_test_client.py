"""Drive the ME3 MCP servers over stdio and check the conversation.

Run: python me3_laptop/mcp_test_client.py [--speak]

Each server is spawned as a real child process, so this exercises the same path
a harness uses: initialize, tools/list, tools/call, including error handling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from console import use_utf8_stdout

BASE = Path(__file__).resolve().parent
TEST_WAV = BASE / "models" / "sherpa-onnx-whisper-tiny.en" / "test_wavs" / "0.wav"
NEWLINE = chr(10)


class McpProcess:
    """A child MCP server speaking newline-delimited JSON-RPC on stdio."""

    def __init__(self, script: str) -> None:
        self.script = script
        self.proc = subprocess.Popen(
            [sys.executable, str(BASE / script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(BASE),
        )
        self.next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            payload["params"] = params
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"{self.script}: pipes are not available")
        self.proc.stdin.write(json.dumps(payload) + NEWLINE)
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.script} closed stdout. stderr tail: {self.stderr_tail()}")
        try:
            response = json.loads(line)
        except ValueError as exc:
            # Non-JSON on stdout is how an MCP server usually breaks: a stray
            # print corrupts the protocol channel.
            raise RuntimeError(
                f"{self.script} sent non-JSON on stdout: {line.strip()[:120]!r}"
            ) from exc
        if response.get("id") != self.next_id:
            raise RuntimeError(f"{self.script} answered out of order: {response}")
        return response

    def notify(self, method: str) -> None:
        if self.proc.stdin is None:
            return
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + NEWLINE)
        self.proc.stdin.flush()

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Call a tool; returns (text, is_error)."""
        response = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        result = response.get("result") or {}
        content = result.get("content") or []
        text = str(content[0].get("text", "")) if content else ""
        return text, bool(result.get("isError"))

    def tool_names(self) -> list[str]:
        response = self.request("tools/list")
        tools = (response.get("result") or {}).get("tools") or []
        return [str(tool.get("name")) for tool in tools]

    def handshake(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "me3-check", "version": "0.1"},
            },
        )
        self.notify("notifications/initialized")
        return response.get("result") or {}

    def stderr_tail(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            self.proc.terminate()
            _, err = self.proc.communicate(timeout=10)
        except (subprocess.TimeoutExpired, ValueError):
            return ""
        return (err or "").strip().splitlines()[-1] if err else ""

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((label, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f"  :: {detail}" if detail else ""))


def check_asr() -> None:
    print("ASR MCP server (me3_laptop/asr_mcp.py):")
    server = McpProcess("asr_mcp.py")
    try:
        info = server.handshake()
        server_info = info.get("serverInfo") or {}
        check("initialize handshake", server_info.get("name") == "me3-asr", str(server_info))
        check("protocol version returned", bool(info.get("protocolVersion")), str(info.get("protocolVersion")))

        names = server.tool_names()
        check("tools/list exposes transcribe", "transcribe" in names, ", ".join(names))

        text, is_error = server.tool("asr_info")
        check("asr_info reports the route", "route:" in text and not is_error, text.splitlines()[0] if text else "")

        if TEST_WAV.exists():
            transcript, is_error = server.tool("transcribe", {"audio_path": str(TEST_WAV), "backend": "whisper"})
            check(
                "transcribe a real clip",
                bool(transcript) and not is_error,
                transcript[:70],
            )
        else:
            check("transcribe a real clip", False, f"missing sample {TEST_WAV}")

        text, is_error = server.tool("transcribe", {"audio_path": "/does/not/exist.wav"})
        check("missing file reported as a tool error", is_error, text[:60])

        response = server.request("tools/call", {"name": "nope", "arguments": {}})
        error = (response.get("error") or {}).get("message", "")
        check("unknown tool is a protocol error", "unknown tool" in error, error)
    finally:
        server.close()


def check_tts(speak_aloud: bool) -> None:
    print(NEWLINE + "TTS MCP server (me3_laptop/tts_mcp.py):")
    server = McpProcess("tts_mcp.py")
    try:
        info = server.handshake()
        check("initialize handshake", (info.get("serverInfo") or {}).get("name") == "me3-tts")

        names = server.tool_names()
        check(
            "tools/list exposes speak and synthesize",
            {"speak", "synthesize", "stop"}.issubset(set(names)),
            ", ".join(names),
        )

        text, is_error = server.tool("tts_info")
        check("tts_info reports the backend", "backend:" in text and not is_error, text.splitlines()[0])

        out_path = BASE / "out" / "mcp_tts_check.wav"
        out_path.unlink(missing_ok=True)
        text, is_error = server.tool("synthesize", {"text": "MCP synthesis check.", "path": str(out_path)})
        size = out_path.stat().st_size if out_path.exists() else 0
        check("synthesize writes audio", size > 1000 and not is_error, f"{size} bytes at {out_path.name}")

        if speak_aloud:
            text, is_error = server.tool("speak", {"text": "Speaking check from the MCP server."})
            check("speak returns without error", not is_error, text[:60])
        text, is_error = server.tool("stop")
        check("stop is safe when idle", not is_error, text[:60])

        voices, is_error = server.tool("voices")
        check("voices answers", bool(voices) and not is_error, voices.splitlines()[0][:60])
    finally:
        server.close()


def check_tasks() -> None:
    print(NEWLINE + "Task MCP server (me3_laptop/tasks_mcp.py):")
    server = McpProcess("tasks_mcp.py")
    try:
        info = server.handshake()
        check("initialize handshake", (info.get("serverInfo") or {}).get("name") == "me3-tasks")

        names = server.tool_names()
        expected = {"set_lights", "adjust_lights", "set_alarm", "media_control", "adjust_volume", "set_reminder", "silence", "set_timer", "get_time"}
        check("ME2 device commands exposed", expected.issubset(set(names)), f"{len(names)} tools")

        text, is_error = server.tool("tasks_info")
        check("tasks_info maps ME2 intents", "ME2 intent coverage" in text and not is_error)

        for label, tool, args in (
            ("lights on at 40%", "set_lights", {"room": "bedroom", "state": "on", "brightness": 40}),
            ("dim the lights", "adjust_lights", {"room": "bedroom", "direction": "dim", "percent": 20}),
            ("set an alarm", "set_alarm", {"time": "07:00", "label": "wake up"}),
            ("next track", "media_control", {"action": "next"}),
            ("volume up", "adjust_volume", {"direction": "up"}),
            ("save a reminder", "set_reminder", {"text": "call mom", "when": "7 pm"}),
        ):
            reply, is_error = server.tool(tool, args)
            check(label, not is_error and bool(reply), reply[:60])

        listing, is_error = server.tool("list_reminders")
        check("list_reminders reads back", "call mom" in listing and not is_error, listing.replace(NEWLINE, " | ")[:70])

        reply, is_error = server.tool("set_lights", {"room": "bedroom", "state": "maybe"})
        check("bad argument is a tool error, not a crash", is_error, reply[:60])

        reply, is_error = server.tool("silence")
        check("silence works", not is_error, reply[:60])
    finally:
        server.close()


def main() -> int:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="ME3 MCP server check")
    parser.add_argument("--speak", action="store_true", help="also play audio out loud")
    args = parser.parse_args()

    check_asr()
    check_tts(args.speak)
    check_tasks()

    failed = [label for label, passed, _ in RESULTS if not passed]
    print(NEWLINE + f"checks: {len(RESULTS)}   failures: {len(failed)}")
    for label in failed:
        print(f"  FAILED: {label}")
    print("RESULT:", "PASS" if not failed else f"FAIL ({len(failed)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
