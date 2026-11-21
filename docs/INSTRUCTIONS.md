# SONJ4 — Consolidated Instructions & Requirements

> Original verbatim course & owner notes: [`docs/Instructions.txt`](Instructions.txt)  
> Install & Raspberry Pi setup: [`docs/INSTALL.md`](INSTALL.md) · Classwork benchmark: [`docs/BENCHMARK.md`](BENCHMARK.md) · Recent diffs: [`docs/RECENT_CHANGES.md`](RECENT_CHANGES.md) · System manual: [`docs/SONJ_DOCUMENTATION.md`](SONJ_DOCUMENTATION.md)

---

## 1. Core Assignment (Machine Exercise 3)

Build a **full-duplex, voice-enabled AI agent running on a Raspberry Pi 4 or 5** (4 GB friendly):
1. Install an AI agent harness (`pi` CLI via RPC + custom MCP client) with **stdio MCP servers** built and run on-device.
2. Build local MCP servers for **SOTA ASR** (`asr_mcp.py`: Parakeet / Moonshine / Whisper via `sherpa-onnx`) and **TTS** (`tts_mcp.py`: Piper / Kokoro) plus task/browser/computer tools (`tasks_mcp.py`, `browser_mcp.py`, `computer_mcp.py`).
3. Run **on-device VAD** (`Silero VAD`) so speech interaction is natural, conversational, and interruptible (full-duplex).
4. Commercial LLM APIs are permitted for reasoning (`GitHub Copilot` / `OpenRouter`), while audio, VAD, AEC/NS, Decision Classifier, and MCP servers run locally.
5. Measure latency with a **standardized monotonic clock** (`perf_counter`, stage boundaries: VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker) and deliver the **2-page MEX3 Deck** (`MEX3_Deck.pdf`, matching `960×540` 16:9 template).

---

## 2. Standing Product Rules & Personas

- **CLI & Standalone App:** Easy `sonj` CLI (`sonj`, `sonj status`, `sonj ask`, `sonj doctor`); standalone console (`app_window.py` / WebView2 or Tk) that shuts down cleanly with zero leaked child processes. Never replace `sonj3` (`:8765`); SONJ4 runs on **`:8766`**.
- **Two Personas:** **Nova** (warm female voice, default) and **Piattos** (natural male voice, switchable on request); custom agents configurable via `agents.json`.
- **Two-Layer Multi-Agent Architecture (`1 + 1 + 1 + 5`):**
  - **Front (1, Nova):** Conversational coordinator (`<2s` replies, max 2 short natural sentences).
  - **Non-LLM Decision Classifier (1):** `<100 ms` budget with **`+10%` Front bias** (`classifier.py`), routing `front | back | doctor`.
  - **Back Pool (5 always-warm):** `back-1..back-5` handle coding, web search, research, verification, summarization, and desktop/browser MCP work concurrently.
  - **Doctor (1):** Read-only diagnostics, log inspection, and 15 s confirmation-gated settings/repairs (`doctor.py`).
- **Self-Healing & Verification:** Auto-diagnose errors without self-coding (`DoctorGate`), and verify every change with runnable checks.

---

## 3. Current SONJ4 & MEX3 Prompt / UI / Audio Specifications

1. **Task-Agnostic Front Prompt (`prompts/front_system_v4.txt`):**
   - Do **not** hardcode specific example tasks (`weather`, `time`, `lights`, `timer`, `alarm`, `calculate`, etc.).
   - Keep responses natural (`max 2 sentences`), confirm before costly/external/irreversible actions, prefer MCP tools rather than guessing, and **never read markdown syntax or URLs aloud** (enforced both in prompt and `sanitize_for_speech()` in `server.py`).
2. **Topbar Controls, Defaults & Instant Tooltips (`static/index.html`, `config.json`):**
   - **`■ Stop Reply` (`POST /api/barge`):** Immediately cuts TTS, filler, and active Front LLM stream without turning off the hands-free mic (`_interrupt_seq` prevents fallback retries).
   - **`🎙 Auto-Interrupt` (`instant_barge`):** **Default `OFF`** (`false`). When OFF, stop words (`stop`, `wait`, `cancel`, `pause`, `quiet`, `enough`, `shush`) or `■ Stop Reply` interrupt; when toggled ON, any real speech interrupts mid-reply or mid-thought.
   - **`👤 Front Only` (`front_only`):** **Default `OFF`** (`false`). When toggled ON, **only Front (Nova)** answers all requests directly and delegation to `back-1..back-5` / `doctor` is blocked.
   - **`🔇 TTS Echo Cancel` (`echo_cancellation`, `nlms-aec`):** **Default `ON`** (`true`). Uses active TTS/filler playback reference to subtract speaker echo live while preserving overlapping user speech (double-talk).
   - **`🌊 Noise Suppress` (`noise_suppression`, `wiener-ns`):** **Default `ON`** (`true`). Telecon spectral noise suppressor running before VAD & ASR.
   - Every topbar control must show a specific plain-English ON/OFF tooltip (`title` + 0 ms `data-tip`) and wrap cleanly with **zero horizontal scrollbars**.
3. **Observability, Workflow & Performance:**
   - **Performance view:** Includes live **Decision Classifier** KPIs (`calls`, `p50/p95/max ms`, route shares, `<100ms` chart) alongside ASR / TTFT / TTFA / E2E charts and RAM/CPU distribution.
   - **Live Workflow & Panes:** Shows real-time `mic -> nlms-aec + wiener-ns -> VAD/ASR -> classifier -> front -> back-1..5 / doctor -> TTS`.
   - **MEX3 Deck (`MEX3_Deck.pdf`):** 2-page `960 × 540 pt` (16:9) deck following the exact template sections (1 System Diagram, 2 Prompt, 3 Context Engineering, 4 & 6 MCP Services & Model Serving, 5 Evaluation & Benchmark, Demo, Results, Challenges & References).

---

## 4. Acceptance Checks

```bash
python deckcheck.py --live --port 8766   # Prompt v4, MD/URL sanitize, NLMS AEC, Wiener NS, defaults OFF, Classifier perf, zero h-scroll, Stop Reply, 5 Front-Only tasks (26 checks)
python classifiercheck.py                # 27 Decision Classifier checks (<100ms, +10% Front bias)
python bargecheck.py                     # 16 echo rejection & Auto-Interrupt checks
python announcecheck.py                  # 9 task-completion announcement checks
python sonj4_eval20.py --unit            # 20 consolidated harness checks
python benchmark.py --runs 5             # Stage benchmark (VAD -> ASR -> LLM -> TTS)
python benchrun.py --turns 20 --url http://127.0.0.1:8766  # Standardized classwork latency method (n=20)
python duplex_test.py                    # Turn-taking, barge-in, quiet noise & echo rejection
```
