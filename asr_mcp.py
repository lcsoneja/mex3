"""ME3 ASR MCP server: speech to text, running on the device.

A sibling of tts_mcp.py with the same stdio protocol (mcp_core.Server), so the
existing MCP protocol client tests it unchanged. Backends, first available wins:

1. remote  - ME3_ASR_URL, an OpenAI-compatible /audio/transcriptions endpoint
2. local   - sherpa-onnx whisper or moonshine from models/ (the engine's ASR)

Run standalone (stdio):
    python me3_laptop/asr_mcp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr import AsrError, create_asr, discover_models
from mcp_core import Server, ToolError, log, make_tool

REMOTE_URL = os.environ.get("ME3_ASR_URL", "")
REMOTE_KEY = os.environ.get("ME3_ASR_KEY", "")
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Whisper decodes at most 30 s and sherpa aborts on oversized pushes - the same
# failure that killed the engine silently (circular-buffer overflow in
# server_err.txt). Clamp at the last point before the native library.
MAX_AUDIO_S = 25.0

_engine: object | None = None
_engine_name = ""


def active_backend() -> str:
    if REMOTE_URL:
        return "remote (" + REMOTE_URL + ")"
    models = [str(m) for m in discover_models(MODELS_DIR)]
    return "local (" + (models[0] if models else "none") + ")"


def _clamp(audio: list[float]) -> list[float]:
    limit = int(MAX_AUDIO_S * 16000)
    return audio[-limit:] if len(audio) > limit else audio


def _get_engine():
    global _engine, _engine_name
    if _engine is not None:
        return _engine, _engine_name
    if REMOTE_URL:
        raise ToolError("remote backend handled in transcribe()")
    models = [str(m) for m in discover_models(MODELS_DIR)]
    if not models:
        raise ToolError("no ASR model under models/ - run me3_laptop/setup_models.py")
    _engine = create_asr(MODELS_DIR, backend=models[0])
    _engine_name = models[0]
    return _engine, _engine_name


def _remote_endpoint() -> str:
    # Audit rule: validate the scheme with a parser against an explicit allowlist
    # before any request is constructed, so a misconfigured value cannot become an
    # unexpected outbound target.
    import urllib.parse

    parsed = urllib.parse.urlparse(REMOTE_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("ME3_ASR_URL must be an http(s) URL with a host")
    return REMOTE_URL.rstrip("/") + "/audio/transcriptions"


def _remote_transcribe(path: Path) -> str:
    import requests

    endpoint = _remote_endpoint()
    data = path.read_bytes()
    try:
        resp = requests.post(
            endpoint,
            data=data,
            headers={
                "Authorization": "Bearer " + REMOTE_KEY,
                "Content-Type": "audio/wav",
            },
            timeout=120,
        )
        if resp.status_code >= 400:
            raise ToolError("remote ASR HTTP " + str(resp.status_code))
        import json

        payload = json.loads(resp.text)
    except requests.RequestException as exc:
        raise ToolError("remote ASR failed: " + type(exc).__name__) from exc
    return str(payload.get("text", "")).strip()


def _read_audio(path: str) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise ToolError("file not found: " + str(source))
    try:
        audio, rate = sf.read(str(source), dtype="float32")
    except Exception as exc:
        raise ToolError("could not read audio: " + type(exc).__name__) from exc
    if rate != 16000:
        import math

        # linear resample to the engine's 16 kHz
        count = math.ceil(len(audio) * 16000.0 / rate)
        idx = np.minimum((np.arange(count) * rate) / 16000.0, len(audio) - 1)
        audio = np.asarray(audio, dtype="float32")[idx.astype("int64")]
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    return np.asarray(mono, dtype="float32").reshape(-1)


def transcribe(
    path: str = "",
    audio_path: str = "",
    backend: str = "",
    sota: bool = False,
) -> str:
    """Transcribe one audio file to text (accepts path or audio_path)."""
    target = (path or audio_path).strip()
    if not target:
        raise ToolError("path is required")
    audio = _read_audio(target)
    if REMOTE_URL:
        return _remote_transcribe(Path(target))
    if sota:
        log("ME3_ASR_URL not set; using local ASR for sota=True")
    if backend and backend in discover_models(MODELS_DIR):
        engine = create_asr(MODELS_DIR, backend=backend)
    else:
        engine, _name = _get_engine()
    if len(audio) > int(MAX_AUDIO_S * 16000):
        log("clamped audio to " + str(MAX_AUDIO_S) + "s")
        audio = audio[-int(MAX_AUDIO_S * 16000) :]
    from typing import Any, cast

    recogniser = cast(Any, engine)
    try:
        return str(recogniser.transcribe(audio)).strip()
    except AsrError as exc:
        raise ToolError("transcription failed: " + str(exc)) from exc


def build_server() -> Server:
    server = Server("me3-asr")

    def tool_transcribe(path: str = "", audio_path: str = "", backend: str = "") -> str:
        target = (path or audio_path).strip()
        if not target:
            raise ToolError("path is required")
        return transcribe(path=target, backend=backend)

    def tool_info() -> str:
        return (
            "route: "
            + active_backend()
            + " | backend: "
            + active_backend()
            + " | max "
            + str(MAX_AUDIO_S)
            + "s | 16 kHz mono recommended"
        )

    server.add(
        make_tool(
            "transcribe",
            "Transcribe a WAV file to text (SOTA ASR: local sherpa-onnx or a remote endpoint).",
            {
                "path": {
                    "type": "string",
                    "description": "absolute or engine-relative path to a wav file",
                },
                "audio_path": {
                    "type": "string",
                    "description": "alias of path for backwards compatibility",
                },
                "backend": {
                    "type": "string",
                    "description": "optional ASR backend (whisper or moonshine)",
                },
            },
            tool_transcribe,
            required=[],
        ),
        make_tool(
            "asr_info",
            "Which ASR backend is active and what it accepts.",
            {},
            tool_info,
        ),
    )
    return server


def main() -> int:
    log("asr-mcp backend: " + active_backend())
    return build_server().serve()


if __name__ == "__main__":
    raise SystemExit(main())
