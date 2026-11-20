"""Download the offline ASR / VAD / TTS / voiceprint assets SONJ3 uses.

Assets come from the k2-fsa/sherpa-onnx GitHub release buckets. Downloads use
curl and extraction uses tar, both present on Windows 10+, Linux, and macOS.
Already-present assets are skipped, so rerunning is cheap.

Profiles:
    laptop   VAD + Parakeet ASR + Whisper fallback + Piper voice + Kokoro + voiceprint
    pi       VAD + Whisper + Moonshine + Piper voice (small; no Kokoro/Parakeet)
    minimal  VAD + Whisper + Piper voice

Usage:
    python setup_models.py                 # laptop profile
    python setup_models.py --profile pi
    python setup_models.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
ASR_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
TTS_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
# NOTE: upstream's release tag is misspelled "speaker-recongition-models".
SPK_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
)

# name -> (url, kind)  kind: "file" (single .onnx) or "archive" (.tar.bz2 -> folder)
ASSETS: dict[str, tuple[str, str]] = {
    "vad": (f"{ASR_BASE}/silero_vad.onnx", "file"),
    "parakeet": (
        f"{ASR_BASE}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "archive",
    ),
    "whisper": (f"{ASR_BASE}/sherpa-onnx-whisper-tiny.en.tar.bz2", "archive"),
    "moonshine": (f"{ASR_BASE}/sherpa-onnx-moonshine-base-en-int8.tar.bz2", "archive"),
    "piper": (f"{TTS_BASE}/vits-piper-en_US-lessac-high.tar.bz2", "archive"),
    "kokoro": (f"{TTS_BASE}/kokoro-int8-multi-lang-v1_1.tar.bz2", "archive"),
    "voiceprint": (
        f"{SPK_BASE}/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "file",
    ),
}

PROFILES: dict[str, list[str]] = {
    "laptop": ["vad", "parakeet", "whisper", "piper", "kokoro", "voiceprint"],
    "pi": ["vad", "whisper", "moonshine", "piper"],
    "minimal": ["vad", "whisper", "piper"],
}

# Where single-file assets land inside models/.
FILE_DEST = {
    "vad": MODELS / "silero_vad.onnx",
    "voiceprint": MODELS
    / "speaker"
    / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
}


def archive_dir(name: str) -> Path:
    stem = ASSETS[name][0].rsplit("/", 1)[-1].removesuffix(".tar.bz2")
    return MODELS / stem


def already_have(name: str) -> bool:
    kind = ASSETS[name][1]
    if kind == "file":
        p = FILE_DEST[name]
        return p.exists() and p.stat().st_size > 0
    return archive_dir(name).is_dir()


def curl(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.name}")
        return True
    print(f"  get   {dest.name} ...", flush=True)
    try:
        result = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(dest), url],
            capture_output=True,
            check=False,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  FAIL  {dest.name}: {exc}")
        return False
    if result.returncode != 0:
        print(f"  FAIL  {dest.name}: {result.stderr.strip()[:200]}")
        return False
    print(f"  ok    {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def extract(archive: Path) -> bool:
    try:
        result = subprocess.run(
            ["tar", "-xjf", str(archive), "-C", str(MODELS)],
            capture_output=True,
            check=False,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  FAIL  extract {archive.name}: {exc}")
        return False
    if result.returncode != 0:
        print(f"  FAIL  extract {archive.name}: {result.stderr.strip()[:200]}")
        return False
    return True


def fetch(name: str) -> bool:
    if already_have(name):
        print(f"  have  {name}")
        return True
    url, kind = ASSETS[name]
    if kind == "file":
        return curl(url, FILE_DEST[name])
    archive = MODELS / url.rsplit("/", 1)[-1]
    if curl(url, archive) and extract(archive):
        archive.unlink(missing_ok=True)
        print(f"  ok    {name} extracted")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download SONJ3 offline speech models")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    ap.add_argument(
        "--only", nargs="*", choices=sorted(ASSETS), help="fetch just these assets"
    )
    ap.add_argument("--list", action="store_true", help="list assets and exit")
    args = ap.parse_args()

    if args.list:
        for name, (url, kind) in ASSETS.items():
            print(f"  {name:11s} {kind:8s} {url}")
        print("profiles:", {k: v for k, v in PROFILES.items()})
        return 0

    MODELS.mkdir(parents=True, exist_ok=True)
    wanted = args.only or PROFILES[args.profile]
    print(f"SONJ3 models - profile={args.profile} - {len(wanted)} assets")
    ok = True
    for name in wanted:
        ok &= fetch(name)
    print("\nverify:")
    for path in sorted(MODELS.rglob("*")):
        if path.is_file() and path.suffix in {".onnx", ".txt"}:
            print(f"  {path.relative_to(MODELS)}  {path.stat().st_size / 1e6:.1f} MB")
    if not ok:
        print(
            "\nSome assets failed. Rerun when online; SONJ falls back to whatever is present."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
