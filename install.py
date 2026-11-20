"""SONJ4 cross-platform installer (Windows / Linux / Raspberry Pi / macOS).

Creates a local .venv, installs the Python dependencies, downloads the offline
speech models, and runs the built-in doctor.

On a Raspberry Pi / Linux with --pi it ALSO (via sudo, once):
  * apt-installs the audio system libraries (libportaudio2, libsndfile1) + tools,
  * installs Node.js 20 and the Pi CLI (@earendil-works/pi-coding-agent),
  * points config.json at the light Pi models so ./sonj4.sh just works.

Usage:
    python install.py                 # Windows / macOS / Linux laptop
    python install.py --pi            # Raspberry Pi: one-command full setup
    python install.py --no-models     # deps only; download models later
    python install.py --check         # only run the doctor, install nothing

After it finishes on a Pi, the only interactive step left is signing in:
    pi          # one-time sign-in (GitHub Copilot / your provider)
    ./sonj4.sh  # start SONJ4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
PI_VOICE = "Neural: en_US-lessac-high"


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        print(f"  could not run: {exc}")
        return 1


def _has_sudo() -> bool:
    return bool(shutil.which("sudo"))


def apt_install(pkgs: list[str]) -> bool:
    if not _has_sudo():
        print("  sudo not available; install these manually: " + " ".join(pkgs))
        return False
    if run(["sudo", "apt-get", "update", "-y"]) != 0:
        print("  warning: apt-get update failed; trying install anyway")
    return run(["sudo", "apt-get", "install", "-y", *pkgs]) == 0


def _node_major() -> int:
    try:
        out = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return int(out.stdout.strip().lstrip("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def ensure_node() -> bool:
    if _node_major() >= 20:
        print("  Node.js >= 20 already present")
        return True
    if not IS_LINUX or not _has_sudo():
        print(
            "  install Node.js 20+ manually, then: npm install -g @earendil-works/pi-coding-agent"
        )
        return False
    print("  installing Node.js 20 (NodeSource) ...")
    setup = Path("/tmp/nodesource_setup.sh")
    if (
        run(
            ["curl", "-fsSL", "https://deb.nodesource.com/setup_20.x", "-o", str(setup)]
        )
        == 0
        and run(["sudo", "-E", "bash", str(setup)]) == 0
        and run(["sudo", "apt-get", "install", "-y", "nodejs"]) == 0
        and _node_major() >= 20
    ):
        return True
    print("  NodeSource failed; falling back to apt nodejs+npm")
    apt_install(["nodejs", "npm"])
    return _node_major() >= 18


def install_pi_cli() -> bool:
    if shutil.which("pi"):
        print("  Pi CLI already installed")
        return True
    if not shutil.which("npm"):
        print("  npm missing; cannot install Pi CLI automatically")
        return False
    pkg = "@earendil-works/pi-coding-agent"
    if run(["npm", "install", "-g", pkg]) == 0:
        return True
    if _has_sudo() and run(["sudo", "npm", "install", "-g", pkg]) == 0:
        return True
    print("  warning: could not install Pi CLI globally; run: npm install -g " + pkg)
    return False


def patch_pi_config() -> None:
    """Point config.json at the light Pi models so ./sonj4.sh works out of the box."""
    cfg = ROOT / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8-sig")) if cfg.exists() else {}
    except (OSError, ValueError):
        data = {}
    data.update(
        {
            "asr_backend": "moonshine",
            "tts_voice": PI_VOICE,
            "nova_voice": PI_VOICE,
            "piattos_voice": PI_VOICE,
            "idle_unload_s": 60,
        }
    )
    try:
        cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("  config.json set to the light Pi models (moonshine ASR + Piper voice)")
    except OSError as exc:
        print(f"  warning: could not write config.json: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Install SONJ4 on this computer")
    ap.add_argument(
        "--pi", action="store_true", help="Raspberry Pi: full one-command setup"
    )
    ap.add_argument("--no-models", action="store_true", help="skip model downloads")
    ap.add_argument("--skip-venv", action="store_true", help="use the current Python")
    ap.add_argument(
        "--python", default=sys.executable, help="Python used to create .venv"
    )
    ap.add_argument("--check", action="store_true", help="only run the doctor")
    args = ap.parse_args()

    if args.check:
        return run([args.python, str(ROOT / "sonj.py"), "doctor"])

    print(f"SONJ4 installer - {sys.platform} - Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        print(
            "warning: Python 3.10-3.12 is the supported range for sherpa-onnx wheels."
        )

    # --- Pi / Linux system prerequisites (audio libs + Node + Pi CLI) ---
    node_ok = True
    cli_ok = True
    if args.pi and IS_LINUX:
        print("\n[1/5] System audio libraries + tools (sudo) ...")
        apt_install(
            [
                "libportaudio2",
                "libsndfile1",
                "python3-venv",
                "curl",
                "tar",
                "git",
                "ca-certificates",
            ]
        )
        print("\n[2/5] Node.js + Pi CLI ...")
        node_ok = ensure_node()
        cli_ok = install_pi_cli() if node_ok else False

    # --- Python venv + dependencies ---
    print("\n[3/5] Python environment + dependencies ...")
    py = args.python
    if not args.skip_venv:
        if not venv_python().exists():
            print("creating .venv ...")
            try:
                venv.create(str(ROOT / ".venv"), with_pip=True)
            except OSError as exc:
                print(f"could not create .venv ({exc}); using current Python")
                py = args.python
            else:
                py = str(venv_python())
        else:
            py = str(venv_python())

    if run([py, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        print("warning: pip upgrade failed; continuing")
    rc = run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if rc != 0:
        print(
            "ERROR: core dependencies failed to install. Check your network / Python version."
        )
        return rc
    extras = ["ddgs", "pywebview"]
    if IS_WIN:
        extras.append("pywin32")
    if run([py, "-m", "pip", "install", *extras]) != 0:
        print(
            "warning: optional extras failed; SONJ still runs with local neural voice + browser window."
        )

    # --- Models ---
    print("\n[4/5] Offline speech models ...")
    if not args.no_models:
        profile = "pi" if args.pi else "laptop"
        if run([py, str(ROOT / "setup_models.py"), "--profile", profile]) != 0:
            print(
                "warning: some models did not download; rerun setup_models.py when online."
            )
    if args.pi:
        patch_pi_config()

    # --- Doctor ---
    print("\n[5/5] Health check ...")
    run([py, str(ROOT / "sonj.py"), "doctor"])

    print("\n" + "=" * 60)
    print("SONJ4 install finished.")
    if args.pi:
        print(f"  Node.js ready : {'yes' if node_ok else 'NO - install Node 20+'}")
        print(
            f"  Pi CLI ready  : {'yes' if cli_ok else 'NO - npm install -g @earendil-works/pi-coding-agent'}"
        )
        print("\nOnly TWO steps left (the sign-in is interactive):")
        print("  1) pi            <- one-time sign-in to your model provider")
        print("  2) ./sonj4.sh    <- start SONJ4 (http://127.0.0.1:8766)")
    else:
        print("Start SONJ4:")
        print("  sonj4.bat   (Windows)   or   ./sonj4.sh   (Linux/macOS)")
        print("For the LLM agents, install + sign in to the Pi CLI:")
        print("  npm install -g @earendil-works/pi-coding-agent   then   pi")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
