# SONJ4 — Local-First Full-Duplex Voice Agent

SONJ4 consolidates the voice agent into **1 Front (`Nova`) + 1 non-LLM Decision Classifier (`<100 ms`, `+10%` Front bias) + 1 Doctor + 5 always-warm Back workers (`back-1..back-5`)**, paired with pure-NumPy **TTS-Reference Echo Cancellation (`nlms-aec`)** and **Telecon Noise Suppression (`wiener-ns`)** for Windows 10/11, Linux, Raspberry Pi 4/5, and macOS.

> **Docs & install:** [`docs/INSTRUCTIONS.md`](docs/INSTRUCTIONS.md) · [`docs/Instructions.txt`](docs/Instructions.txt) · [`docs/BENCHMARK.md`](docs/BENCHMARK.md) · [`docs/INSTALL.md`](docs/INSTALL.md) · [`docs/HANDOFF.md`](docs/HANDOFF.md) · [`docs/RECENT_CHANGES.md`](docs/RECENT_CHANGES.md) · [`docs/SONJ_DOCUMENTATION.md`](docs/SONJ_DOCUMENTATION.md)

## Quick Start

```bash
python install.py          # Windows / Linux / macOS  (add --pi on Raspberry Pi 4/5)
sonj4.bat                  # Windows (runs on :8766; sonj3 on :8765 stays untouched)
./sonj4.sh                 # Linux / macOS / Raspberry Pi
```

## Audio & Agent Controls (Topbar Defaults)

| Button | Default | Purpose |
|---|---|---|
| `■ Stop Reply` | Ready | Immediately cancels TTS playback, aborts the active Front LLM stream, drops queued audio, and keeps the hands-free mic active (`POST /api/barge`). |
| `🎙 Auto-Interrupt` | **off** | **OFF** by default: stop words (`stop`, `wait`, `cancel`, `pause`, `quiet`, `enough`, `shush`) or `■ Stop Reply` cut a reply. Toggle **ON** so any speech interrupts mid-reply or mid-thought. |
| `👤 Front Only` | **OFF** | **OFF** by default: full team routes multi-step work to `back-1..back-5` and diagnostics to `doctor`. Toggle **ON** to force only Front (Nova) to answer every request directly. |
| `🔇 TTS Echo Cancel` | **ON** | `nlms-aec`: uses Nova's own TTS/filler playback as a reference to subtract speaker echo before VAD & ASR while preserving overlapping user voice (double-talk). |
| `🌊 Noise Suppress` | **ON** | `wiener-ns`: telecon spectral noise suppressor that removes steady fan/AC/room noise before VAD & ASR. |

## Architecture & Panes

```text
mic -> nlms-aec (TTS ref) -> wiener-ns -> Silero VAD -> asr-mcp (Parakeet / Moonshine)
  -> Decision Classifier (<100ms, +10% Front bias)
  -> Front (Nova, prompts/front_system_v4.txt) -> tts-mcp (Piper / Kokoro) -> speaker
       ├─ back-1..back-5 (5 always-warm workers + tasks-mcp / computer / browser)
       └─ doctor (read-only diagnostics + 15s confirmation gate)
```

- **Performance View:** Tracks **Decision Classifier** latency (`p50`, `p95`, `max`, `<100ms` budget), Front/Back/Doctor routing share, per-turn ASR/TTFT/TTFA/E2E charts, and RAM/CPU distribution.
- **Live Workflow & Inspector Panes:** Live `pipeline`, `reasoning trace`, `event log`, `agents`, `workflow · live` (`mic -> AEC/NS -> VAD/ASR -> classifier -> front -> back-1..5 / doctor -> TTS`), `voices`, and `mind map`, plus lower split agent terminals.

## Verification

```bash
python deckcheck.py                     # 16 unit + DSP checks (prompt v4, MD/URL sanitize, NLMS AEC, Wiener NS, 5-task Front-Only)
python deckcheck.py --live --port 8766  # 22+ Playwright live UI + zero h-scroll + Classifier perf + Stop + 5-task Front-Only QA
python classifiercheck.py               # 27/27 Decision Classifier checks (<100ms, +10% Front bias)
python bargecheck.py                    # 16/16 echo guard & Auto-Interrupt ON/OFF checks
python announcecheck.py                 # 9/9 task announcement & TTS fallback checks
python sonj4_eval20.py --unit           # 20/20 consolidated harness checks
```
