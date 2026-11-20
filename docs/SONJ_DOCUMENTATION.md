# SONJ4 — Whole Documentation

SONJ4 is a local-first, full-duplex voice agent running on Windows 10/11, Linux, Raspberry Pi 4/5, and macOS. One engine owns the microphone, acoustic cleanup, VAD/ASR/TTS pipeline, non-LLM Decision Classifier, and 7 always-warm Pi panes (`front`, `back-1..back-5`, `doctor`).

## 1. Architecture (`1 + 1 + 1 + 5`)

```text
mic -> Adaptive NLMS AEC (TTS ref) -> Telecon Wiener NS -> Silero VAD -> ASR (sherpa-onnx)
  -> Decision Classifier (<100 ms, non-LLM, +10% Front bias; choices: front | back | doctor)
  -> FRONT (1 coordinator & voice companion, Nova / Piattos)
       ├─ immediate turns -> direct natural reply (<= 2 sentences, no markdown/URLs aloud) -> TTS
       ├─ multi-step build/research/verify/summarize -> delegate_task -> Back Pool (back-1..back-5)
       └─ diagnostics/logs/settings -> ask_agent/delegate -> Doctor (1, 15s confirm gate)
```

- **Front (1)**: `sonj4/prompts/front_system_v4.txt` — task-agnostic voice companion & coordinator. Answers immediately or delegates long work unless **Front Only** is ON.
- **Decision Classifier (1, non-LLM)**: `sonj4/classifier.py` sits between user input and Front. Returns typed `Decision(route="front"|"back"|"doctor")` in `<100 ms` with a **+0.10 additive bias to Front**, tracking rolling `p50 / p95 / max ms` and route shares for the **Performance** view.
- **Doctor (1)**: read-only inspection + `doctor_app_control` (15 s confirmation pop-up) + `doctor.py` detectors and catalogue repairs.
- **Back Pool (5 always-warm)**: `back-1..back-5` run build, web research, verification, summarization, and desktop/browser MCP tasks concurrently (`max_parallel=5`) with peer A2A coordination shown in `community · agents talking`.

## 2. Audio Front-End & Topbar Toggles

| Control | Default | Behavior |
|---|---|---|
| `■ Stop Reply` | Always ready | Calls `POST /api/barge` immediately to stop TTS playback, abort the active LLM stream, drop stale audio, and return to listening (keeps hands-free mic ON). |
| `🎙 Auto-Interrupt` (`instant_barge`) | **OFF** | When **OFF** (default), talking over Nova does not cut her off unless you say a stop word (`stop`, `wait`, `cancel`, `pause`, `quiet`, `enough`, `shush`) or click `■ Stop Reply`. Toggle **ON** so any real speech immediately interrupts TTS + thinking LLM. |
| `👤 Front Only` (`front_only`) | **OFF** | When **OFF** (default), full team (`front` + `back-1..back-5` + `doctor`) runs. Toggle **ON** so only Front (Nova) answers directly; delegation is blocked. |
| `🔇 TTS Echo Cancel` (`echo_cancellation`) | **ON** | `nlms-aec` (`sonj4/aec.py`): aligns mic blocks against the TTS/filler playback reference (`0–600 ms`, 8-tap NLMS) to cancel Nova's own speaker echo while preserving double-talk interruptions. |
| `🌊 Noise Suppress` (`noise_suppression`) | **ON** | `wiener-ns` (`sonj4/aec.py`): 256-sample overlap-add STFT Wiener spectral suppressor reducing steady fan/AC/room noise before VAD & ASR. |

## 3. Live Panes, Workflow & Performance

- **Inspector Panes (Chat right column):** `pipeline` (8 live stages including `AEC/NS`, `gates`, `classifier`), `reasoning trace`, `engine event log`, `status`, `agents` (inline model/thinking + round table), `workflow · live` (`mic -> nlms-aec + wiener-ns -> VAD/ASR -> classifier -> front -> back-1..5 / doctor -> TTS`), `voices`, and `mind map`.
- **Lower Panes:** `community · agents talking` (with live classifier pill) + one live terminal per warm Pi pane (`front`, `back-1..back-5`, `doctor`).
- **Performance View:** Live **Decision Classifier** KPI strip, latency sparkline (`chart-clf`), and Front/Back/Doctor share bar alongside ASR / LLM TTFT / TTFA / E2E charts, per-feature RAM & CPU table, gate funnel, and bottleneck review.

## 4. HTTP API (local only, default `:8766`)

- `/api/classify` (`POST`): `{text}` -> non-LLM `<100ms` `Decision` JSON.
- `/api/performance/distribution` & `/api/performance/review` (`GET`): live classifier, audio DSP, stage blocks, and memory telemetry.
- `/api/barge` (`POST`): immediate mid-turn stop (TTS + filler + Front LLM abort, keeps mic active).
- `/api/settings` (`POST`): live toggle updates (`instant_barge`, `front_only`, `echo_cancellation`, `noise_suppression`, `wake_gate`, model/voice).
- `/api/delegate`, `/api/ask_agent`, `/api/community`, `/api/send`, `/api/state`, `/api/health`, `/api/docs`, `/api/pipeline`, `/api/session/new`, `/api/restart`, `/api/doctor/control`.
