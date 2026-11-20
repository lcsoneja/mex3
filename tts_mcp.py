"""ME3 TTS MCP server: text to speech, running on the device.

The server is mine and runs on the Pi; the voice may be local or a hosted SOTA
endpoint, which the brief allows. Backend chain, first one available wins:

1. remote  - ME3_TTS_URL, an OpenAI-compatible /audio/speech endpoint (DGX vLLM)
2. piper   - offline, the usual choice on a Raspberry Pi (ME3_PIPER_MODEL)
3. espeak  - offline fallback if espeak-ng is installed
4. SAPI    - Windows built-in voice (what the laptop uses)

Run standalone (stdio, speaks MCP):
    python me3_laptop/tts_mcp.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from mcp_core import Server, ToolError, log, make_tool
from tts import NEURAL_PREFIX, NeuralTts, SapiTts, TtsError, list_neural_voices

OUT_DIR = Path(__file__).resolve().parent / "out"
REMOTE_URL = os.environ.get("ME3_TTS_URL", "")
REMOTE_KEY = os.environ.get("ME3_TTS_KEY", "")
REMOTE_VOICE = os.environ.get("ME3_TTS_VOICE", "alloy")
PIPER_MODEL = os.environ.get("ME3_PIPER_MODEL", "")

# Playback process, tracked so stop() can cut speech off mid-sentence.
_player: subprocess.Popen | None = None
_sapi_engine: SapiTts | None = None
_neural_engine: NeuralTts | None = None


def _sapi() -> SapiTts:
    """Create the Windows voice on first use (and keep it warm)."""
    global _sapi_engine
    if _sapi_engine is None:
        _sapi_engine = SapiTts()
    return _sapi_engine


def active_backend() -> str:
    """Which backend will actually be used, in priority order."""
    if REMOTE_URL:
        return f"remote ({REMOTE_URL})"
    neural = list_neural_voices()
    if neural:
        return f"neural ({neural[0].removeprefix(NEURAL_PREFIX)})"
    if shutil.which("piper") and PIPER_MODEL:
        return f"piper ({PIPER_MODEL})"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    if sys.platform == "win32":
        return "windows SAPI"
    return "none"


def synthesize(text: str, voice: str = "", rate: int = 0, path: str = "") -> str:
    """Render speech to a WAV file and return its path."""
    if not text.strip():
        raise ToolError("text is required")
    destination = Path(path) if path else OUT_DIR / "tts_out.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if REMOTE_URL:
        _synthesize_remote(text, voice or REMOTE_VOICE, destination)
    elif list_neural_voices():
        _synthesize_neural(text, voice, rate, destination)
    elif shutil.which("piper") and PIPER_MODEL:
        _synthesize_piper(text, destination)
    elif shutil.which("espeak-ng") or shutil.which("espeak"):
        _synthesize_espeak(text, voice, rate, destination)
    elif sys.platform == "win32":
        try:
            _sapi().to_wav(text, destination)
        except TtsError as exc:
            raise ToolError(f"SAPI failed: {exc}") from exc
    else:
        raise ToolError(
            "no TTS backend: set ME3_TTS_URL, install piper (with ME3_PIPER_MODEL) ",
            "or install espeak-ng",
        )

    if not destination.exists():
        raise ToolError("synthesis produced no audio")
    return str(destination)


def _speech_endpoint(base: str) -> str:
    """Accept a base URL or a full route, so configuration stays forgiving.

    Without this, pointing ME3_TTS_URL at a host and port posted to / and the server
    answered 404 - found by probing a stand-in server before the real one existed.
    """
    trimmed = base.rstrip("/")
    route = "/v1/audio/speech"
    return trimmed if trimmed.endswith(route) else trimmed + route


def _synthesize_remote(text: str, voice: str, destination: Path) -> None:
    import requests

    headers = {"Authorization": f"Bearer {REMOTE_KEY}"} if REMOTE_KEY else {}
    try:
        response = requests.post(
            _speech_endpoint(REMOTE_URL),
            headers=headers,
            json={"model": os.environ.get("ME3_TTS_MODEL", "tts-1"), "input": text, "voice": voice},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ToolError(f"remote TTS failed: {exc}") from exc
    destination.write_bytes(response.content)


def _synthesize_neural(text: str, voice: str, rate: int, destination: Path) -> None:
    """Offline neural voice (sherpa-onnx): the natural option on laptop and Pi.

    Priority order puts this above piper and espeak because it sounds markedly
    more human, and it is skipped automatically when no model is present.
    """
    global _neural_engine
    if _neural_engine is None:
        _neural_engine = NeuralTts(voice=voice or None, rate=rate)
    _neural_engine.set_rate(rate)
    _neural_engine.to_wav(text, destination)


def _synthesize_piper(text: str, destination: Path) -> None:
    binary = shutil.which("piper")
    if not binary:
        raise ToolError("piper disappeared from PATH")
    result = subprocess.run(
        [binary, "--model", PIPER_MODEL, "--output_file", str(destination)],
        input=text.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise ToolError(f"piper failed: {result.stderr.decode('utf-8', 'replace')[:160]}")


def _synthesize_espeak(text: str, voice: str, rate: int, destination: Path) -> None:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise ToolError("espeak disappeared from PATH")
    words_per_minute = 175 + rate * 20
    command = [binary, "-w", str(destination), "-s", str(words_per_minute)]
    if voice:
        command += ["-v", voice]
    command.append(text)
    result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if result.returncode != 0:
        raise ToolError(f"espeak failed: {result.stderr.decode('utf-8', 'replace')[:160]}")


def _play(path: Path) -> None:
    """Play a WAV, keeping the process handle so stop() can end it."""
    global _player
    player = None
    for candidate in ("aplay", "paplay", "afplay"):
        player = shutil.which(candidate)
        if player:
            break
    if player is None:
        raise ToolError("no audio player found (aplay/paplay/afplay)")
    _player = subprocess.Popen([player, str(path)])


def speak(text: str, voice: str = "", rate: int = 0) -> str:
    """Say something out loud on this device."""
    if sys.platform == "win32" and not REMOTE_URL:
        try:
            tts = _sapi()
            tts.set_rate(rate)
            if voice:
                tts.set_voice(voice)
            tts.say(text)
        except TtsError as exc:
            raise ToolError(f"SAPI failed: {exc}") from exc
        return f"speaking {len(text)} characters via Windows SAPI"

    path = Path(synthesize(text, voice=voice, rate=rate))
    _play(path)
    return f"speaking {len(text)} characters via {active_backend()}"


def stop() -> str:
    """Cut speech off immediately (the barge-in path)."""
    global _player
    stopped = []
    if sys.platform == "win32":
        try:
            _sapi().stop()
            stopped.append("SAPI")
        except TtsError as exc:
            log(f"SAPI stop failed: {exc}")
    if _player is not None and _player.poll() is None:
        _player.terminate()
        stopped.append("playback")
        _player = None
    return "stopped " + (", ".join(stopped) if stopped else "nothing (was not speaking)")


def voices() -> str:
    """Voices the active backend offers."""
    if sys.platform == "win32" and not REMOTE_URL:
        try:
            return "\n".join(_sapi().list_voices()) or "no SAPI voices"
        except TtsError as exc:
            raise ToolError(f"could not list voices: {exc}") from exc
    if REMOTE_URL:
        return f"remote endpoint, voice selected by name (default {REMOTE_VOICE})"
    return "this backend has no voice list"


def tts_info() -> str:
    """Which text-to-speech route is active."""
    return (
        f"backend: {active_backend()}\n"
        f"remote url: {REMOTE_URL or 'not set'}\n"
        f"piper model: {PIPER_MODEL or 'not set'}\n"
        f"output dir: {OUT_DIR}"
    )


def build_server() -> Server:
    server = Server("me3-tts", version="0.1.0")
    server.add(
        make_tool(
            "speak",
            "Speak text out loud on this device.",
            {
                "text": {"type": "string", "description": "what to say"},
                "voice": {"type": "string", "description": "voice name, backend specific"},
                "rate": {"type": "integer", "description": "-6 slow to 6 fast, default 0"},
            },
            speak,
            required=["text"],
        ),
        make_tool(
            "synthesize",
            "Render speech to a WAV file without playing it (used by the latency benchmark).",
            {
                "text": {"type": "string", "description": "what to say"},
                "voice": {"type": "string", "description": "voice name, backend specific"},
                "rate": {"type": "integer", "description": "-6 slow to 6 fast, default 0"},
                "path": {"type": "string", "description": "where to write the WAV"},
            },
            synthesize,
            required=["text"],
        ),
        make_tool(
            "stop",
            "Stop speaking immediately.",
            {},
            stop,
        ),
        make_tool("voices", "List the voices available on this device.", {}, voices),
        make_tool("tts_info", "Report the active text-to-speech route.", {}, tts_info),
    )
    return server


if __name__ == "__main__":
    raise SystemExit(build_server().serve())
