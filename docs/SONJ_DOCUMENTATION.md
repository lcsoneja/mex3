# SONJ4 — Whole Documentation

SONJ4 is a local-first, full-duplex voice agent. One engine owns the microphone,
speakers, conversation, non-LLM Decision Classifier, and 7 always-warm Pi panes.

## 1. Architecture (1 + 1 + 1 + 5)

```
mic -> VAD (Silero) -> ASR (sherpa-onnx)
  -> Decision Classifier (<100 ms, non-LLM, +10% Front bias; choices: front | back | doctor)
  -> FRONT (1 coordinator & voice companion)
       ├─ quick (<2s) turns & device_tool -> direct reply -> TTS
       ├─ long (>2s) build/research/verify/summarize -> delegate_task -> Back Pool (back-1..back-5)
       └─ diagnostics/logs/settings -> ask_agent/delegate -> Doctor (1)
```

- **Front (1)**: owner's voice companion & coordinator. Replies <2s, never runs slow tools directly.
- **Decision Classifier (1, non-LLM)**: `sonj4/classifier.py` sits between user and Front. Returns typed `Decision(route="front"|"back"|"doctor")` in <100 ms with a **+0.10 additive bias to Front**.
- **Doctor (1)**: read-only code/log inspection + `doctor_app_control` (15s auto-confirm pop-up) + in-engine `doctor.py` detectors & catalogue repairs.
- **Back Pool (5 always-warm)**: `back-1` .. `back-5` absorb Back + Researcher + Critic + Summarizer + automation. Idle-pool scheduler runs up to 5 tasks concurrently; peers coordinate via `ask_agent` / `post_community` in `community · agents talking`.

## 2. HTTP API (local only)

- `/api/classify` (POST): `{text}` -> non-LLM `<100ms` `Decision` JSON.
- `/api/community` (POST): `{from_role, to_role, message}` -> independent `community · agents talking` entry.
- `/api/delegate` (POST): dispatches to idle worker in `back-1..back-5` (or `doctor`).
- `/api/ask_agent` (POST): synchronous A2A consult (`front`/`back-k`/`doctor`, depth guarded `<=2`).
- `/api/send`, `/api/state`, `/api/health`, `/api/session/new`, `/api/session/load`, `/api/restart`, `/api/doctor/control`.
