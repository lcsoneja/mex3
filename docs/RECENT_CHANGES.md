# SONJ4 — Recent Changes & Agent Handoff Log (`7a4310c` → `c5f4b98`)

This document summarizes everything added and modified since the `7a4310c` Raspberry Pi one-command install baseline so an incoming coding agent (or RPi setup agent) can inspect exact files, runtime defaults, API additions, behavior changes, and verification commands without re-reading full commit diffs.

---

## 1. Quick Orientation for an Incoming Agent

- **Repo & branch:** `https://github.com/lcsoneja/sonj4` (`main`)
- **Primary deliverables added/updated:**
  1. **Task-agnostic Nova prompt v4 & speech sanitizer** (`prompts/front_system_v4.txt`, `server.py`, `agent.py`)
  2. **Half-way conversation interruption & TTS stop fixes** (`server.py`, `tts.py`, `static/index.html`, `bargecheck.py`)
  3. **Adaptive TTS-Reference Echo Canceller + Telecon Noise Suppressor** (`aec.py`, `server.py`, `syshealth.py`)
  4. **Topbar toggle controls with instant tooltips + zero horizontal scrollbar UI** (`static/index.html`, `config.json`, `server.py`)
  5. **Decision Classifier performance telemetry + updated live Workflow & Docs** (`classifier.py`, `syshealth.py`, `static/index.html`, `docs/`, `README.md`)
  6. **Completed 2-page 16:9 MEX3 Deck** (`MEX3_Deck.pdf`, matching `machine_exercise_2page_3.pdf`)
  7. **Automated unit, DSP, Playwright UI, and 5-task Front-Only QA suite** (`deckcheck.py`)

---

## 2. Exact File-by-File Changes

### Prompts & Speech Safety
- **`prompts/front_system_v4.txt`** *(new)*: Versioned Front Agent (`Nova`) system prompt. Removed all forbidden hardcoded example tasks (`weather`, `time`, `lights`, `timer`, `alarm`, `calculate`, `snake`, `gazette`, `music`, `volume`). Enforces max 2 short natural sentences, stop-and-listen on interruption, confirmation before costly/external/irreversible actions, MCP tool use (`asr-mcp`, `tts-mcp`, `tasks-mcp`), no markdown/URLs read aloud, and Solo Front awareness.
- **`agent.py` & `server.py` (`_role_system_prompt("front")`, `DEFAULT_SYSTEM_PROMPT`)**: Synced to match `prompts/front_system_v4.txt`.
- **`server.py` (`sanitize_for_speech`)**: Strips markdown formatting (`**`, `*`, `` ` ``, headings, bullets), converts `[label](url)` to `label`, and strips raw `http(s)://` / `www.` URLs before `SpeechPump.push()`, `voice.say()`, and task-completion announcements.
- **`server.py` (`_ask_front_pi`, `_record_turn`)**: Logs `context_chars` and `context_tokens_est` per turn.

### Half-Way Interruption / Stop Fixes & Defaults
- **`server.py` & `tts.py`:**
  - `Settings` defaults set to **`instant_barge: bool = False`** (`Auto-Interrupt: off`) and **`front_only: bool = False`** (`Front Only: OFF`), matching `config.json`.
  - Added monotonic `_interrupt_seq` in `Session.interrupt()` and `Session.ask()` so an interrupted turn stops `FillerBank`, `SpeechPump`, `tts()`, `Player`, and `pi_abort("front")` immediately **and skips any fallback LLM retry or delayed non-streamed TTS**.
  - Added live mid-thought VAD interruption (`_think_speech_ms >= 190 ms`) while `status == "thinking"` when `instant_barge=True`.
  - Relaxed `_barge_gates()` (`normal`: `0.06 RMS / 220 ms`, `relaxed`: `0.04 / 150 ms`, `strict`: `0.10 / 380 ms`), capped `_echo_peak_rms` adaptive threshold (`min(0.14, _echo_peak_rms * 1.05)`) with faster decay (`0.96`), and expanded `BARGE_TRIGGER_WORDS` with `"wait"`, `"pause"`, `"hold on"`.
  - Updated `_hears_own_voice()`: when `instant_barge=True` and `not wake_to_interrupt`, any real multi-word non-echo speech cuts immediately while single artifacts (e.g., `"of"`), noise hallucinations, and self-echo stay blocked.
  - Updated `NeuralTts._stream` / `NeuralTts.stop` / `CloudTts.stop` (`tts.py`) to poll `rendered.get(timeout=0.05)` and join `timeout=0.2` so playback stops promptly on barge-in.
- **`static/index.html` (`#btn-barge`):** Fixed bug where clicking `■ Stop Reply` sent `/api/stop` (kill mic) instead of `/api/barge` whenever `s.recording` was `True` during TTS playback. `#btn-barge` now always calls `POST /api/barge` unless idle in explicit `ptt` mode.

### Acoustic Echo Cancellation (`nlms-aec`) & Telecon Noise Suppression (`wiener-ns`)
- **`aec.py`:**
  - Upgraded `EchoSuppressor` (`name = "nlms-aec"`): searches `0–600 ms` delay (`STRIDE=32`, 2 ms grid) against the live TTS/filler reference ring (`AEC_TAIL_S = 0.55`), solves a regularized 8-tap linear room reflection filter (`_nlms_cancel`), subtracts speaker echo (`23.9 dB` measured ERLE), and preserves overlapping user speech (`double-talk`).
  - Added `NoiseSuppressor` (`name = "wiener-ns"`): pure-NumPy 50%-overlap STFT (`frame_len=512`, `hop=256`, `<0.3 ms/block`) decision-directed Wiener spectral suppressor reducing steady fan/AC/room noise (`8.6 dB` measured attenuation) before VAD & ASR.
- **`server.py` (`Settings.noise_suppression = True`, `Session._ns`, `_listen_loop`, `Session.state`):**
  - Runs `_echo.process(block)` and `_ns.process(block)` **live during `speaking_now == True`** (before `_speech_run` & `_judge_interruption`) as well as idle listening (before `SileroVad.accept` & ASR).
  - Exposes `echo_engine`, `echo_score`, `echo_erle_db`, `ns_engine`, `ns_reduction_db` in `/api/state`.
- **`syshealth.py` & `config.json`:** Registered `noise_suppression` in `_SETTINGS_DOCS` and `config.json` (`true`).

### Decision Classifier Performance Telemetry, Live Panes & Zero-HScroll UI
- **`classifier.py`:** Added rolling latency ring (`_latencies`, `maxlen=60`) and `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `recent_latencies`, `deadline_ms`, `front_bias_pct`, `onnx_ready` inside `DecisionClassifier.stats`.
- **`syshealth.py` (`feature_distribution`, `performance_review`, `app_introspection`, `inspect_architecture`):**
  - Added `Decision Classifier (rules-v1)`, `TTS Echo Canceller (nlms-aec)`, and `Telecon Noise Suppressor (wiener-ns)` to `features`, plus `classifier_perf`, `gate_funnel`, `snr_info`, `avg_tok_per_s`, `model_scorecard`, and `pi_headroom`.
  - Updated `app_introspection` and `inspect_architecture` to SONJ4's `1 + 1 + 1 + 5` roster (`front`, `classifier`, `doctor`, `back-1..back-5`).
- **`static/index.html`:**
  - Removed horizontal scrollbars across viewports (`1440×900` & `1280×800`) via `flex-wrap: wrap` on `.topbar`, `.phead`, `.statusbar`, `max-width: 100%` / `overflow-x: hidden` on layout containers, and wrapping `.md pre` / scrollable `.md table`.
  - Added topbar toggles with `0 ms` CSS hover tooltips (`data-tip`) and dynamic `title` strings:
    - `🎙 Auto-Interrupt: off` (`#btn-instant-barge`, default OFF)
    - `👤 Front Only: OFF` (`#btn-front-only`, default OFF)
    - `🔇 TTS Echo Cancel: ON` (`#btn-aec`, default ON)
    - `🌊 Noise Suppress: ON` (`#btn-ns`, default ON)
    - Plus clear tooltips on `■ Stop Reply`, `⏵ Listen`, `wake`, `+ New Chat`, `model`, `↻ Restart`, `◐`.
  - Added **`Decision Classifier Performance`** card (`#perf-clf-card`, `#perf-clf-kpis`, `#chart-clf`, `#perf-clf-bar`) to **Performance** view.
  - Updated **`workflow · live`** (`#tab-workflow`), **`pipeline`** (`#tab-pipeline`, 8 stages preserved), **`mind map`** (`#tab-mindmap`), **`agents`** (`#tab-agents` + `#db-roster` + `buildRoundTable`), and **`#lpane-state-community`** to reflect SONJ4's `nlms-aec + wiener-ns` cleanup, `classifier`, `front (Nova)`, `back-1..back-5`, and `doctor` (removing legacy `researcher`/`critic`/`summarizer`/`router` references).

### Documentation & MEX3 Deck
- **`MEX3_Deck.pdf`:** Built matching the exact `960 × 540 pt` (16:9, 2-page) format of `machine_exercise_2page_3.pdf` with vector architecture diagram, versioned prompt v4, context engineering, 7-row MCP/Model Serving table, benchmark table, demo script, results, and challenges/references.
- **`docs/SONJ_DOCUMENTATION.md`, `docs/HANDOFF.md`, `README.md`, `server.py` (`_docs_summary`):** Updated to reflect SONJ4 audio cleanup (`nlms-aec`, `wiener-ns`), topbar defaults (`Auto-Interrupt: off`, `Front Only: off`), live workflow/performance panes, and verification suite.

---

## 3. Commit History (`7a4310c` → `c5f4b98`)

| Commit | Summary |
|---|---|
| `b52ba72` | Task-agnostic Nova prompt v4, MD/URL speech sanitizer, context logging, initial `MEX3_Deck.pdf` |
| `806af6d` | Exact `960×540` 16:9 `MEX3_Deck.pdf`, half-way `■ Stop Reply` & mid-thought/TTS stop fixes, dynamic tooltips, 19/19 UI + 5-task Front-Only QA |
| `ef8a671` | Pure-NumPy `nlms-aec` + `wiener-ns` audio front-end, live playback AEC/NS, `#btn-aec`/`#btn-ns` toggles (`22/22` QA) |
| `4c746da` | Zero horizontal-scroll UI, `Auto-Interrupt` & `Front Only` OFF defaults, Decision Classifier performance view, updated panes/workflow/docs & `MEX3_Deck.pdf` |
| `fce3123` | Pinned open-mic echo tests in `bargecheck.py` with `instant_barge=True` while default stays `False` |
| `c5f4b98` | Synced formatted `server.py`, `syshealth.py`, `deckcheck.py` & verified `MEX3_Deck.pdf` on `main` |

---

## 4. Verification Status & Remaining On-Device Work

- **Verified on Windows 11 x64 (`:8766`):**
  - `python deckcheck.py --live --port 8766` → **`26/26 passed`** (Prompt v4, MD/URL sanitizer, `instant_barge=False` & `front_only=False` defaults, `nlms-aec` `23.9 dB` ERLE + double-talk preservation, `wiener-ns` `8.6 dB` room-noise reduction, Decision Classifier telemetry, zero horizontal scrollbars at `1440×900` & `1280×800`, topbar toggles & tooltips, mid-turn `■ Stop Reply`, 5 live Front-Only tasks with 0 delegations, live Workflow/Performance/Docs panes).
  - `python classifiercheck.py` → **`27/27 passed`** (`0.01 ms` p95)
  - `python bargecheck.py` → **`all checks passed`**
  - `python announcecheck.py` → **`9/9 passed`**
  - `python sonj4_eval20.py --unit` → **`20/20 passed`**
  - `ruff check` & `py_compile` clean across modified Python modules.
- **Pending on Raspberry Pi 4/5 hardware (for RPi setup agent):**
  1. `git pull origin main` inside `~/sonj4` (or fresh `git clone https://github.com/lcsoneja/sonj4.git`)
  2. `python3 install.py --pi` (installs Moonshine + Piper + Silero VAD profile for RPi 4 GB)
  3. Run `./sonj4.sh`, then verify live on-device audio & benchmarks (`python3 deckcheck.py`, `python3 benchmark.py --runs 5`, and optional `python3 sonj4_eval20.py --live --port 8766`) and record RPi model, cooling, and hardware latency measurements back into Results.
