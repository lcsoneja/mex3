# SONJ4 — Handoff

Version: see `VERSION` (`4.0.0`). Read with `docs/INSTALL.md` and `docs/SONJ_DOCUMENTATION.md`.

## Consolidated Agent & Audio Architecture

| Component | Count | Type | Role |
|---|---:|---|---|
| **AEC + NS** | 2 DSP stages | Pure NumPy (`aec.py`) | `nlms-aec` (TTS reference cancellation, ON default) + `wiener-ns` (telecon room noise suppression, ON default) |
| **Decision Classifier** | 1 | Non-LLM (`classifier.py`) | Sits between user and Front; `<100ms` budget; `+10%` Front bias; `p50/p95/max` stats in Performance view |
| **Front (Nova)** | 1 | LLM Pi pane | Task-agnostic voice companion (`prompts/front_system_v4.txt`, max 2 sentences, no MD/URLs aloud) |
| **Doctor** | 1 | LLM Pi pane + `doctor.py` | Read-only diagnostics, logs, settings proposals (15 s confirm), catalogue repairs |
| **Back Pool** | 5 | Always-warm LLM Pi panes (`back-1..back-5`) | Unified generalist workers: build, research, verify, summarize, MCP, peer A2A |

## Default Control States

- `instant_barge` (`Auto-Interrupt`): **`false`** (stop words or `■ Stop Reply` interrupt; toggle ON for voice-over cut-in)
- `front_only` (`Front Only`): **`false`** (full 1+1+1+5 team active; toggle ON to restrict answers to Front only)
- `echo_cancellation` (`TTS Echo Cancel`): **`true`** (`nlms-aec`)
- `noise_suppression` (`Noise Suppress`): **`true`** (`wiener-ns`)

## Verification Suite

- `python deckcheck.py` / `python deckcheck.py --live --port 8766` — prompt v4, MD/URL speech sanitizer, NLMS AEC (`23.9 dB` ERLE), Wiener NS (`8.6 dB`), zero horizontal scrollbars, Classifier performance UI, half-way `■ Stop Reply`, topbar toggles, and 5-task Front-Only cycle.
- `python classifiercheck.py` — DecisionClassifier contract, `<100ms` latency, `+10%` Front bias, 20 routes (`27/27`).
- `python bargecheck.py` & `python bargelatencycheck.py` — echo rejection, `Auto-Interrupt` ON/OFF gating, real ASR floor.
- `python announcecheck.py` (`9/9`) & `python sonj4_eval20.py` (`20/20` unit, `21/21` live).
