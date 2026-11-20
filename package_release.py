"""Build a clean, shippable SONJ4 release zip (no secrets, no runtime output).

Excludes API keys, out/ sessions/logs, model weights (unless --with-models),
caches, and local environments. Writes a MANIFEST and SHA-256 checksums so a
recipient can verify the archive.

Usage:
    python package_release.py                 # dist/sonj4-<version>.zip (no models)
    python package_release.py --with-models   # include downloaded model weights
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

EXCLUDE_DIRS = {
    "out",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "dist",
    ".pi",
    ".git",
    "node_modules",
}
EXCLUDE_SUFFIX = {".key", ".pyc", ".log", ".bak", ".zip"}
EXCLUDE_NAMES = {"openrouter.key", "elevenlabs.key", "config.json.bak", ".env"}


def included(path: Path, with_models: bool) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return False
    if not with_models and "models" in parts:
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIX:
        return False
    if path.name.startswith(".") and path.name not in {".gitignore"}:
        return False
    return path.is_file()


def main() -> int:
    ap = argparse.ArgumentParser(description="Package a clean SONJ4 release")
    ap.add_argument("--with-models", action="store_true", help="include model weights")
    args = ap.parse_args()

    version = (
        (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if (ROOT / "VERSION").is_file()
        else "0.0.0"
    )
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"sonj4-{version}.zip"

    files = sorted(p for p in ROOT.rglob("*") if included(p, args.with_models))
    if not files:
        print("nothing to package")
        return 1

    manifest_lines = [f"SONJ4 {version}", ""]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, f"sonj4/{rel}")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest_lines.append(f"{digest}  {rel}")
        zf.writestr("sonj4/MANIFEST.sha256", "\n".join(manifest_lines) + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(files)} files, {size_mb:.1f} MB)")
    print(
        "excluded: API keys, out/, caches, .venv"
        + ("" if args.with_models else ", models/")
    )
    print("\nRecipient installs with:  python install.py   (see docs/INSTALL.md)")
    if not args.with_models:
        print("Models are downloaded on first install by setup_models.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
