"""Acceptance check for the real stdio MCP client and ASR server.

Run from the Exercise 3 root with: python me3_laptop/mcp_client_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
TEST_WAV = BASE / "out" / "selftest_tts.wav"
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((label, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f"  :: {detail}" if detail else ""))


def result_text(result: dict[str, Any]) -> str:
    content = result.get("content") or []
    return str(content[0].get("text", "")) if content else ""


def check_timeout(client_type: Any) -> None:
    client = client_type(
        "timeout-probe",
        [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.stdin.readline()"],
        timeout=0.1,
    )
    timed_out = False
    failure_detail = ""
    try:
        try:
            client.start()
        except TimeoutError:
            timed_out = True
        except (OSError, RuntimeError) as exc:
            failure_detail = f"{type(exc).__name__}: {exc}"
    finally:
        graceful = client.close()
    check("request timeout is enforced", timed_out, failure_detail)
    check("timeout probe shuts down cleanly", graceful, f"return code {client.returncode}")


def main() -> int:
    try:
        client_module = importlib.import_module("mcp_client")
    except ImportError as exc:  # report import failure as a regular check
        check("MCP client module imports", False, f"{type(exc).__name__}: {exc}")
        return report()

    McpClient = client_module.McpClient
    discover_servers = client_module.discover_servers
    check("MCP client module imports", True)

    configs = discover_servers()
    server_names = {str(config.get("name")) for config in configs}
    check("ASR and TTS servers are discoverable", {"asr", "tts"}.issubset(server_names), ", ".join(sorted(server_names)))
    asr_config = next((config for config in configs if config["name"] == "asr"), None)
    check("ASR server spawn config is available", asr_config is not None, str(asr_config))
    if asr_config is None:
        return report()
    check("transcription sample exists", TEST_WAV.is_file(), str(TEST_WAV))
    if not TEST_WAV.is_file():
        return report()

    client = McpClient(**asr_config)
    try:
        try:
            initialized = client.start()
            server_info = initialized.get("serverInfo") or {}
            check("initialize handshake", server_info.get("name") == "me3-asr", str(server_info))

            tools = client.list_tools()
            tool_names = {str(tool.get("name")) for tool in tools}
            check("tools/list exposes required ASR tools", {"transcribe", "asr_info"}.issubset(tool_names), ", ".join(sorted(tool_names)))

            info = client.call("asr_info", {})
            info_text = result_text(info)
            check("asr_info returns backend details", bool(info_text.strip()) and not info.get("isError"), info_text[:120])

            transcription = client.call("transcribe", {"path": str(TEST_WAV)})
            transcript = result_text(transcription)
            check("transcribe returns a non-empty transcript", bool(transcript.strip()) and not transcription.get("isError"), transcript[:120])
        except (OSError, RuntimeError) as exc:
            check("ASR MCP client flow", False, f"{type(exc).__name__}: {exc}")
    finally:
        graceful = client.close()
        check("clean shutdown", graceful, f"return code {client.returncode}")

    check_timeout(McpClient)
    return report()


def report() -> int:
    failed = [label for label, passed, _detail in RESULTS if not passed]
    print(f"checks: {len(RESULTS)}   failures: {len(failed)}")
    print("RESULT:", "PASS" if not failed else f"FAIL ({len(failed)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
