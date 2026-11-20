"""SONJ4 cross-platform installer (Windows / Linux / Raspberry Pi / macOS).

Creates a local .venv, installs the Python dependencies, downloads the offline
speech models, and runs the built-in doctor. No administrator rights required.

Usage:
    python install.py                 # full laptop install (venv + deps + models)
    python install.py --pi            # Raspberry Pi profile (smaller models)
    python install.py --no-models     # deps only; download models later
    python install.py --check         # only run the doctor, install nothing

After it finishes, start SONJ with:
    Windows:          sonj4.bat
    Linux / macOS:    ./sonj4.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        print(f"  could not run: {exc}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Install SONJ4 on this computer")
    ap.add_argument("--pi", action="store_true", help="Raspberry Pi model profile")
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

    if not args.no_models:
        profile = "pi" if args.pi else "laptop"
        if run([py, str(ROOT / "setup_models.py"), "--profile", profile]) != 0:
            print(
                "warning: some models did not download; rerun setup_models.py when online."
            )

    run([py, str(ROOT / "sonj.py"), "doctor"])

    print("\nDone. Start SONJ:")
    if IS_WIN:
        print("  sonj4.bat            (or: python sonj.py)")
    else:
        print("  ./sonj4.sh           (or: python3 sonj.py)")
    print(
        "\nFor specialist agents (Back/Researcher/Doctor), install the Pi CLI and sign in:"
    )
    print("  npm install -g @earendil-works/pi-coding-agent")
    print("See docs/INSTALL.md for credentials, models, and troubleshooting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
