"""Prepare the deploy file list and scp commands for the SONJ Pi profile.

Run from any directory with: python me3_laptop/install_pi.py [--host user@pi]
This reports the exact sources and transfer commands; it does not copy files or secrets.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEFAULT_HOST = os.environ.get("SONJ_PI_HOST", "lcsoneja@192.168.1.75")

RUNTIME_FILES = (
    "agent.py",
    "asr.py",
    "asr_mcp.py",
    "audio.py",
    "filereader.py",
    "llm.py",
    "mcp_core.py",
    "router.py",
    "server.py",
    "sonj.py",
    "tasks.py",
    "tasks_mcp.py",
    "toolbox.py",
    "tts.py",
    "tts_mcp.py",
    "vad.py",
    "websearch.py",
    "workflowview.py",
)
CONFIG_FILES = (
    "agents.json",
    "config.json",
    "pi_profile.json",
    "profiles.json",
)
SUPPORT_DIRS = ("skills", "static")
MODEL_ITEMS = (
    "models/silero_vad.onnx",
    "models/sherpa-onnx-moonshine-base-en-int8",
    "models/sherpa-onnx-whisper-tiny.en",
    "models/vits-piper-en_US-lessac-high",
)


def _paths(names: tuple[str, ...]) -> list[Path]:
    return [ROOT / name for name in names]


def _print_scp_command(paths: list[Path], destination: str, recursive: bool = False) -> None:
    flag = " -r" if recursive else ""
    sources = " ".join(shlex.quote(str(path.relative_to(PROJECT).as_posix())) for path in paths)
    print(f"scp{flag} {sources} {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare SONJ Pi deployment transfer commands")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH destination (default: %(default)s)")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._@:-]+", args.host):
        parser.error("--host must be a user@host or host name without shell syntax")

    groups = {
        "runtime and profile files": _paths(RUNTIME_FILES + CONFIG_FILES),
        "support directories": _paths(SUPPORT_DIRS),
        "Pi ASR/VAD/Piper models": _paths(MODEL_ITEMS),
    }
    missing = [path for paths in groups.values() for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing deploy source: {path.relative_to(PROJECT).as_posix()}")
        return 1

    staged = [path for paths in groups.values() for path in paths]
    if any("kokoro" in path.name.lower() for path in staged):
        print("refusing deploy list containing a Kokoro model")
        return 1

    print(f"SONJ Pi deploy file list ({len(staged)} entries; no Kokoro models):")
    for title, paths in groups.items():
        print(f"\n{title}:")
        for path in paths:
            print(f"  {path.relative_to(PROJECT).as_posix()}")

    host = args.host
    print("\nTransfer commands (run from the project root):")
    print(f"ssh {host} 'mkdir -p ~/me3/models'")
    code_and_config = groups["runtime and profile files"]
    _print_scp_command(code_and_config, f"{host}:~/me3/")
    _print_scp_command(groups["support directories"], f"{host}:~/me3/", recursive=True)
    _print_scp_command(groups["Pi ASR/VAD/Piper models"], f"{host}:~/me3/models/", recursive=True)
    print("\nopenrouter.key is intentionally not included; transfer credentials separately if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
