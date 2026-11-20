"""Headless feature check for the console surfaces and the 2026-11-08 features.

Covers what selftest.py (audio/ASR/VAD/TTS) and bargecheck.py (interruption decision)
do not: the HTTP API the console depends on, the persona voice settings, the Thinking
stream, and the stop-word configuration.

Needs the engine on 127.0.0.1:8765 for the API half; the pure-function half runs alone.
It never POSTs a turn, never opens the microphone, and writes settings back to the
values it read, so running it cannot change behaviour.

Run:  python me3_laptop/featurecheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from tts import (
    KOKORO_PERSONA_PREFIX,
    KOKORO_VOICE_IDS,
    NeuralTts,
    list_neural_voices,
)

BASE = "http://127.0.0.1:8765"
failures = 0


def report(title: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {title:58s} {detail}")


def get(path: str):
    try:
        r = requests.get(BASE + path, timeout=10)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def api_checks() -> None:
    code, health = get("/api/health")
    if code != 200:
        report("engine reachable (API half skipped)", False, f"GET /api/health -> {code} {str(health)[:60]}")
        return
    report("engine reachable", True, str(health)[:40])

    code, state = get("/api/state")
    report("GET /api/state", code == 200 and isinstance(state, dict), f"{code}")
    if not isinstance(state, dict):
        return
    for key in ("status", "level", "speaking", "thinking", "turns", "tasks", "settings", "tts_voices", "session_dir"):
        report(f"state exposes '{key}'", key in state, "")
    settings = state.get("settings", {})
    for key in ("nova_voice", "piattos_voice", "barge_triggers", "tts_voice", "barge_rms"):
        report(f"settings expose '{key}'", key in settings, str(settings.get(key))[:40])
    voices = state.get("tts_voices", [])
    report("persona voices offered", any(v.startswith(KOKORO_PERSONA_PREFIX) for v in voices),
           f"{len(voices)} voices")
    report("thinking stream is a list", isinstance(state.get("thinking"), list),
           f"{len(state.get('thinking') or [])} entries")
    for path in ("/api/sessions", "/api/files", "/api/map"):
        status, _body = get(path)
        report(f"GET {path}", status == 200, f"{status}")
    # idempotent settings round-trip: write back exactly what is configured
    try:
        r = requests.post(BASE + "/api/settings", json={
            "nova_voice": settings.get("nova_voice"),
            "piattos_voice": settings.get("piattos_voice"),
            "barge_triggers": settings.get("barge_triggers"),
        }, timeout=10)
        report("POST /api/settings round-trip", r.status_code == 200 and r.json().get("ok") is not False, f"{r.status_code}")
    except Exception as exc:  # noqa: BLE001
        report("POST /api/settings round-trip", False, f"{type(exc).__name__}: {exc}")


def pure_checks() -> None:
    report("kokoro voice table is non-empty", bool(KOKORO_VOICE_IDS), str(len(KOKORO_VOICE_IDS)) + " voices")
    for name, sid in KOKORO_VOICE_IDS.items():
        report(f"sid for {name}", NeuralTts._parse_sid(KOKORO_PERSONA_PREFIX + name) == sid, f"sid={sid}")
    try:
        voices = list_neural_voices()
        report("list_neural_voices includes personas",
               any(v.startswith(KOKORO_PERSONA_PREFIX) for v in voices), f"{len(voices)} entries")
    except Exception as exc:  # noqa: BLE001
        report("list_neural_voices includes personas", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("featurecheck: console API + 2026-11-08 features")
    print("pure functions (no engine needed):")
    pure_checks()
    print("engine API:")
    api_checks()
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
