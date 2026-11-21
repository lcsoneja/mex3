# SONJ4 — Handoff

Version: see `VERSION` (`4.0.0`). Read with [`docs/INSTRUCTIONS.md`](INSTRUCTIONS.md), [`docs/Instructions.txt`](Instructions.txt), [`docs/BENCHMARK.md`](BENCHMARK.md), `docs/INSTALL.md`, `docs/SONJ_DOCUMENTATION.md`, and [`docs/RECENT_CHANGES.md`](RECENT_CHANGES.md) (exact diff & agent handoff log since `7a4310c`).

## Consolidated Agent & Audio Architecture

| Component | Count | Type | Role |
|---|---:|---|---|
| **AEC + NS** | 2 DSP stages | Pure NumPy (`aec.py`) | `nlms-aec` (TTS reference cancellation, ON default) + `wiener-ns` (telecon room noise suppression, ON default) |
| **Decision Classifier** | 1 | Non-LLM (`classifier.py`) | Sits between user and Front; `<100ms` budget; `+10%` Front bias; `p50/p95/max` stats in Performance view |
| **Front (Nova)** | 1 | LLM Pi pane | Task-agnostic voice companion (`prompts/front_system_v4.txt`, max 2 sentences, no MD/URLs aloud) |
| **Doctor** | 1 | LLM Pi pane + `doctor.py` | Read-only diagnostics, logs, settings proposals (15 s confirm), catalogue repairs |
| **Back Pool** | 5 | Always-warm LLM Pi panes (`back-1..back-5`) | Unified generalist workers: build, research, verify, summarize, MCP (exclusive `github-mcp`), peer A2A |

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

## Raspberry Pi 5, Bookmarks, Stock Quotes & Back-Only GitHub MCP

- **Public MEX3 repository:** `https://github.com/lcsoneja/mex3` (`MEX3_Deck.pdf` updated for both Win11 & RPi 5 8GB, unified prompt v4).
- **MCP Servers (`mcp_servers.json`):** `asr-mcp`, `tts-mcp`, `browser-mcp`, `computer-mcp`, `tasks-mcp`, `agents-mcp`, plus Back-only **`github-mcp`** (`github_mcp.py`, blocked on `front`, exclusive to `back-1..back-5` with `github_auth.py` token stored outside git in `~/.sonj4-github-token`, `chmod 600`; sign in via Settings UI or `sonj4 github-login`).
- **Pi 5 Tuned Defaults & Bookmarks (`Demo Purpose`, `High Speed`, `High Thinking`, `Text Only`):** `moonshine` (`thr=2`, `195 ms` p50 ASR), `en_US-lessac-medium` / `en_US-amy-low` (`thr=2`, `355–388 ms` TTFB), `local-onnx` classifier (`models/classifier/intent.onnx`, `0.38 ms` p95), local Ollama disabled on Raspberry Pi, and `get_stock_price` supporting global Yahoo tickers + PSE Phisix (including GCash → `GLO` and Smart → `TEL`).
