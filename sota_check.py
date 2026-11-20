"""Check the hosted SOTA speech endpoints before relying on them.

A demo that fails at the model call is worse than one that never claimed SOTA
support, so this probes both services and reports which request shape each one
accepts, and how fast it answers.

Run:
    python me3_laptop/sota_check.py
    ME3_ASR_URL=http://10.153.60.233:8000 ME3_TTS_URL=http://10.153.60.233:8001 \
        python me3_laptop/sota_check.py

Exit status is non-zero when either service is unusable, so it can gate a demo.
It exercises the real adapters inside asr_mcp/tts_mcp rather than a copy of them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from console import use_utf8_stdout

BASE = Path(__file__).resolve().parent
CLIP = BASE / "models" / "sherpa-onnx-whisper-tiny.en" / "test_wavs" / "0.wav"
OUT_DIR = BASE / "out"


def probe_models(url: str, key: str, timeout: float = 20.0) -> tuple[bool, str]:
    """GET /v1/models: proves reachability and shows what the server hosts."""
    import requests

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = requests.get(f"{url.rstrip('/')}/v1/models", headers=headers, timeout=timeout)
        response.raise_for_status()
        rows = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001 - any failure here is the diagnosis
        return False, f"unreachable ({type(exc).__name__}: {str(exc)[:80]})"
    ids = [str(row.get("id", "?")) for row in rows][:4]
    return True, "models: " + (", ".join(ids) if ids else "(none listed)")


def probe_asr(url: str, key: str) -> tuple[bool, str]:
    """Run the real ASR adapter against the endpoint."""
    import asr_mcp

    if not CLIP.exists():
        return False, f"missing test clip {CLIP}"
    started = time.perf_counter()
    try:
        text = asr_mcp.transcribe(audio_path=str(CLIP), sota=True)
    except Exception as exc:  # noqa: BLE001 - report whatever the adapter reports
        return False, f"failed ({type(exc).__name__}: {str(exc)[:110]})"
    elapsed = (time.perf_counter() - started) * 1000
    if not text:
        return False, "returned an empty transcript"
    mode = getattr(asr_mcp, "_remote_mode", "?")
    return True, f"{elapsed:.0f} ms via {mode} -> {text[:60]!r}"


def probe_tts(url: str, key: str) -> tuple[bool, str]:
    """Run the real TTS adapter against the endpoint."""
    import tts_mcp

    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / "sota_check.wav"
    target.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        path = Path(tts_mcp.synthesize("This is the SOTA check.", path=str(target)))
    except Exception as exc:  # noqa: BLE001 - report whatever the adapter reports
        return False, f"failed ({type(exc).__name__}: {str(exc)[:110]})"
    elapsed = (time.perf_counter() - started) * 1000
    if not path.exists():
        return False, "reported success but wrote no audio"
    return True, f"{elapsed:.0f} ms -> {path.stat().st_size} bytes"


def main() -> int:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="probe the hosted SOTA endpoints")
    parser.add_argument("--asr-url", default=os.environ.get("ME3_ASR_URL", ""))
    parser.add_argument("--tts-url", default=os.environ.get("ME3_TTS_URL", ""))
    parser.add_argument("--asr-key", default=os.environ.get("ME3_ASR_KEY", ""))
    parser.add_argument("--tts-key", default=os.environ.get("ME3_TTS_KEY", ""))
    args = parser.parse_args()

    # The adapters read their configuration at import time, so set it first.
    os.environ["ME3_ASR_URL"] = args.asr_url
    os.environ["ME3_ASR_KEY"] = args.asr_key
    os.environ["ME3_TTS_URL"] = args.tts_url
    os.environ["ME3_TTS_KEY"] = args.tts_key

    results: list[tuple[str, bool, str]] = []

    print(f"ASR endpoint: {args.asr_url or '(not set - offline model will be used)'}")
    if args.asr_url:
        reachable, detail = probe_models(args.asr_url, args.asr_key)
        results.append(("asr reachable", reachable, detail))
        print(f"  [{'PASS' if reachable else 'FAIL'}] {detail}")
        if reachable:
            ok, detail = probe_asr(args.asr_url, args.asr_key)
            results.append(("asr transcribe", ok, detail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")

    print()
    print(f"TTS endpoint: {args.tts_url or '(not set - local voice will be used)'}")
    if args.tts_url:
        reachable, detail = probe_models(args.tts_url, args.tts_key)
        results.append(("tts reachable", reachable, detail))
        print(f"  [{'PASS' if reachable else 'FAIL'}] {detail}")
        if reachable:
            ok, detail = probe_tts(args.tts_url, args.tts_key)
            results.append(("tts synthesize", ok, detail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")

    failed = [name for name, ok, _ in results if not ok]
    print()
    if not args.asr_url and not args.tts_url:
        print("Neither endpoint is configured, so there is nothing to probe.")
        print("Set ME3_ASR_URL and ME3_TTS_URL (VPN required for the class DGX).")
        return 1
    print("RESULT:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
