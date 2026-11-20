# SONJ4 — local-first full-duplex voice agent

SONJ4 consolidates the agent team to **1 Front + 1 non-LLM Decision Classifier (<100 ms, +10% Front bias) + 1 Doctor + 5 always-warm Back pool workers (`back-1..back-5`)**, running on **Windows 10/11, Linux, Raspberry Pi OS, and macOS**.

> **Install & handoff:** see [`docs/INSTALL.md`](docs/INSTALL.md), [`docs/HANDOFF.md`](docs/HANDOFF.md), and [`docs/SONJ_DOCUMENTATION.md`](docs/SONJ_DOCUMENTATION.md).

## Quick start

```bash
python install.py          # Windows / Linux / macOS  (add --pi on Raspberry Pi)
# then (runs on :8766 by default so sonj3 on :8765 is untouched):
sonj4.bat                  # Windows
./sonj4.sh                 # Linux / macOS / Pi
```

Verify with `python classifiercheck.py`, `python announcecheck.py`, and `python sonj4_eval20.py`.

## Two desktop builds

| Launcher | Window | What you see |
|---|---|---|
| **`ME3 Agent.bat`** (recommended) | Native window (pywebview + Edge WebView2), no browser chrome | The full **web console**: Chat, Sessions, Files, Sources, Agents, Skills, MCP, Thinking, Settings, the Subagents / Workflow / Mind-map inspector, Listen / Stop and the context meter |
| `python me3_laptop/app_gui.py` | Tkinter window | The compact standalone build: model dropdown, tool cards, latency strip |

`ME3 Agent.bat` starts the HTTP engine itself when nothing is listening, opens
the window, and stops the engine again when the window closes. It falls back
from pywebview to an Edge/Chrome app-mode window, then to the default browser.

Check which window it would use, without opening anything:

```bat
python me3_laptop/app_window.py --check
```

From a terminal, the same window opens with the **`sonj`** command:

```bat
sonj                 :: open the window, then keep a live status dash in this terminal
sonj dash            :: live one-view dashboard here (Ctrl+C quits, --once = one frame)
sonj dash --once     :: print a single status frame and exit
sonj status --json   :: engine, audio, agents, memory
sonj ask "what time is it?"
sonj doctor          :: can this machine run the agent?
```

`sonj` prefers the native console window and falls back to the Tk build when
the WebView2 runtime is missing; it never opens a browser tab.

## The dashboard (one view)

The console opens on **Dashboard** - a single scrollable page instead of hopping
between rail views:

| Card | What it is |
|---|---|
| **Status** | main directory, model, tokens used, spend, memory, active agents, session |
| **Reasoning trace** | the agents' live thinking, in place (like pi.dev) - router, consults, tools |
| **Agents** | every role's **model** and **thinking** level as inline dropdowns - no trip to Settings |
| **Crew studio** | describe the crew ("AI thoughts"), agents → tasks → tools canvas, a Resources panel to pick agents/tasks/tools, an execution timeline, and Export |
| **Execution** | the event timeline behind the latest turn |

The **Subagents** tab in the Chat view carries the same editable roster, so model
and thinking can be changed from the conversation too. The main directory also
shows as a chip in the Chat header.

## Run it (current SONJ3 build)

```bash
python install.py               # one-time install (venv + deps + models + doctor)
python sonj.py                  # start the engine + console
python sonj.py doctor           # health check
python announcecheck.py         # task results reach chat AND speech
```

Launchers: `sonj3.bat` (Windows) or `./sonj3.sh` (Linux/macOS/Pi). Audio plays
through your machine; no cloud service is required unless you pick a cloud LLM /
ASR / TTS provider. See `docs/INSTALL.md` for credentials and troubleshooting.

### What is in the window

| Area | Contents |
|---|---|
| Left panel | Provider + model dropdown, OpenRouter key box, speech-recognition model, voice, rate, TTS and barge-in switches, barge-in threshold, device names |
| Right panel | The conversation: your turns, **each tool call with its arguments, result and duration**, the reply, and a latency strip (asr / llm / tts / total) |
| Buttons | **Hold to talk** (records while held), **Listen continuously** (VAD-driven), **Interrupt now** (manual barge-in), **New chat** |

### Then try these

1. **Hold to talk** and say *"what time is it?"* - watch the `get_time` tool card appear.
2. Say *"turn on the bedroom light at 40 percent"* - the `set_lights` card shows the arguments it chose.
3. Say *"set a timer for 30 seconds"* and wait - the app announces completion by itself.
4. Click **Listen continuously** and just talk; it detects when you stop.
5. While it is speaking, **talk over it** - it should stop mid-sentence.
6. Change the model in the dropdown, or paste an OpenRouter key and switch provider.

## Pipeline

```text
mic -> VAD (silero) -> ASR (sherpa-onnx) -> LLM + tools -> TTS (SAPI) -> speakers
                                  |
                        barge-in: speech while speaking -> TTS.stop()
```

| Stage | File | Notes |
|---|---|---|
| VAD | `vad.py` | Silero via sherpa-onnx, 512-sample windows |
| ASR | `asr.py` | Whisper tiny.en or Moonshine base, fully offline |
| LLM + tools | `llm.py`, `toolbox.py`, `agent.py` | Tool-calling loop, max 4 rounds |
| TTS | `tts.py` | Windows SAPI, asynchronous so it can be cut off |
| Audio I/O | `audio.py` | sounddevice; abortable playback |
| Desktop app | `app_gui.py` | Tkinter window (the app) |
| Engine | `server.py` | The audio/agent engine plus an optional HTTP mode |

Tools the model can call: `get_time`, `set_timer`, `set_lights`, `get_weather`,
`calculate`, **`web_search`**, **`read_page`**. `set_lights` is simulated on the
laptop; on the Pi it becomes a real GPIO call.

### Web search

No API key needed. Providers are tried in order and the app reports which one
answered:

| Provider | Needs | Notes |
|---|---|---|
| `ddgs` | `pip install ddgs` | General web search, handles DuckDuckGo's anti-bot token |
| DuckDuckGo raw endpoints | nothing | Often blocked with HTTP 202 - kept as a fallback |
| Wikipedia | nothing | Reliable for facts, weak for news |
| Brave | `BRAVE_API_KEY` | Best quality when you have a key |

`web_search` takes a `recency` argument (`day`/`week`/`month`/`year`), so "what is
the latest news about X" returns fresh pages. `read_page` fetches one URL and
returns readable text; it refuses non-public hosts so the model cannot probe your
LAN.

Try: *"search the web for the current price of a Raspberry Pi 5"*, then
*"what is the latest news about voice agents?"*.

## Checks you can run

```bat
python me3_laptop/selftest.py                 :: ASR, VAD, TTS round trip (no mic needed)
python me3_laptop/app_gui.py --selftest       :: builds the window, renders a turn, checks widgets
python me3_laptop/toolbox.py                  :: exercises every tool, including bad arguments
python me3_laptop/websearch.py --selftest     :: reports which search providers actually answer
python me3_laptop/memcheck.py                 :: measures memory use and the idle release
python me3_laptop/modelcheck.py               :: every model answers; leaves nothing in RAM
python me3_laptop/llmcheck.py                 :: shows the raw tool-calling exchange with Ollama
```

Optional headless mode (used by the automated tests; not the app):

```bat
python me3_laptop/server.py --no-browser --no-tts
python me3_laptop/apitest.py --speak          :: drives the running app over HTTP
```

## Settings and secrets

- The **OpenRouter key is session-only**: held in memory, never written to
  `config.json`, gone when the app closes.
- If no key is typed in, the app **finds one the machine already has**, in this
  order: the `OPENROUTER_API_KEY` environment variable, a local
  `me3_laptop/openrouter.key` file, then the Pi CLI's own config
  (`~/.pi/agent/models.json`). Nothing is copied or duplicated - the key is read
  at startup and only held in memory. The log records `OpenRouter key found on
  this machine (memory only, never saved)` when it happens.
- Everything else (provider, model, voice, thresholds) persists in
  `me3_laptop/config.json`.

## Verified behaviour

| Check | Result |
|---|---|
| Offline ASR | Moonshine 0.18 s, Whisper tiny.en 0.29 s for 6.6 s audio (~0.03x realtime) |
| Silero VAD | 1 segment, 5.83 s speech detected in 21 ms |
| TTS round trip | SAPI -> WAV -> ASR: "Set a timer for five minutes please." -> "...for 5 minutes please." |
| Tool calling | `set_lights`, `get_time`, `set_timer` all call correctly and the reply matches the tool result |
| SAPI off the main thread | measured `tts=3126 ms` on the worker thread |
| Desktop window | `--selftest` builds the window and renders a turn with its tool card |
| Tools | `toolbox.py`: 12 checks pass, including bad arguments (bad state, negative timer, junk numbers, LAN URL) |
| Web search | live queries answered: price lookup and news lookup, with per-provider fallback |

## Latency benchmark (classwork)

`benchmark.py` is the standardized method: fixed boundaries (end of speech → VAD
close → ASR → **first LLM token** → TTS audio ready), real human input clips that
are identical on every machine, no microphone required, reported as p50 / p95 /
max. It also folds in real conversation turns and every barge-in decision from
`out/app.log`.

```bat
python me3_laptop/benchmark.py --runs 5 --report me3_laptop/out/benchmark_report.md
```

Measured on 2026-11-02 with offline Whisper and a free OpenRouter model:

| Metric (ms) | Laptop p50 | Raspberry Pi p50 |
|---|---|---|
| VAD close | 22 | 100 |
| ASR | 316 | 1292 |
| **TTFT** | **8385** | **8277** |
| TTS render | 16 | 35 |
| **To first word** | **8723** | **12239** |

The model is 85-95% of the latency and the figure is almost identical on both
machines, because it happens in the cloud. Full method and discussion:
`BENCHMARK.md`.

### Full-duplex behaviour (no microphone needed)

`duplex_test.py` injects audio into the real pipeline and measures the other half
of the problem - turn taking, barge-in, noise rejection and echo rejection:

```bat
python me3_laptop/duplex_test.py --report me3_laptop/out/duplex_report.md
```

4 scenarios pass on both the laptop and the Raspberry Pi; barge-in is decided in
about 0.9 s (350 ms of that is the speech required to qualify), turn-taking and
echo guards both verified. This is what makes the agent interruptible rather than
walkie-talkie, measured rather than asserted.

## MCP servers (the graded ME3 deliverable)

The brief asks for **MCP servers for SOTA ASR and TTS, built and run by you on the
Pi** (the model behind them may be a hosted API). Three servers are implemented
here, with no external MCP package: the protocol is a small stdio JSON-RPC core
(`mcp_core.py`), so the Pi needs no extra installs.

| Server | File | Tools | Route |
|---|---|---|---|
| **ASR** | `asr_mcp.py` | `transcribe` (path or base64 WAV), `asr_info` | offline sherpa-onnx by default, or a remote endpoint via `ME3_ASR_URL` |
| **TTS** | `tts_mcp.py` | `speak`, `synthesize`, `stop`, `voices`, `tts_info` | remote (`ME3_TTS_URL`) → piper (`ME3_PIPER_MODEL`) → espeak-ng → Windows SAPI |
| **Tasks** (optional) | `tasks_mcp.py` | 17 device commands | reuses `toolbox.py`, so there is one implementation |

The task server covers the ME2 command vocabulary, and `tasks_info` prints the
mapping: lights on/off/dim/bright, timer, alarm, thermostat warmer/cooler, media
play/pause/next/previous, volume up/down, reminders, call, search, silence.

Run one by hand (it speaks MCP on stdin/stdout; logs go to stderr):

```bat
python me3_laptop/asr_mcp.py
```

Check all three over the wire, exactly as a harness would:

```bat
python me3_laptop/mcp_test_client.py           :: 25 protocol checks
python me3_laptop/mcp_test_client.py --speak   :: also plays audio
```

Registering with an MCP-capable harness (for example the Pi's `pi` CLI or
pipecat) is a config entry per server:

```json
{
  "mcpServers": {
    "me3-asr":   { "command": "python3", "args": ["/home/lcsoneja/me3/asr_mcp.py"] },
    "me3-tts":   { "command": "python3", "args": ["/home/lcsoneja/me3/tts_mcp.py"] },
    "me3-tasks": { "command": "python3", "args": ["/home/lcsoneja/me3/tasks_mcp.py"] }
  }
}
```

Environment variables select the routes: `ME3_ASR_URL` / `ME3_ASR_KEY` /
`ME3_ASR_BACKEND`, `ME3_TTS_URL` / `ME3_TTS_KEY` / `ME3_TTS_VOICE`,
`ME3_PIPER_MODEL`. With none set the servers run fully offline.

## Interface

The transcript is a chat: your turns are dark bubbles on the right, the
assistant's are green on the left, and every tool call is a purple card showing
the arguments it chose, the result, and how long it took. Under each turn is a
line of timings (asr / llm / tts / total, plus the model that answered).

The **state indicator** in the top bar shows what the app is doing, and turns red
and pulses while your voice is being captured:

| Indicator | Meaning |
|---|---|
| grey `idle` | not listening |
| green `listening` | microphone open, waiting for speech |
| **red `RECORDING`** | your voice is being captured right now (the window title also shows `RECORDING`) |
| amber `transcribing` / `thinking` | speech-to-text, then the model |
| blue `speaking` | the reply is being spoken |

The window sizes itself to the screen (a fixed size ran off the bottom of a
150%-scaled display and hid the talk buttons), the talk buttons and text box sit
at the bottom, and the control column scrolls.

Run `python me3_laptop/app_gui.py --demo` to see the chat layout with sample
turns and no microphone.

## Free models (no credit spend)

The OpenRouter dropdown is filled from the live catalogue and shows **only free,
tool-capable models** - currently 14 of them, Nemotron first - so a small credit
cap cannot be spent by accident. The default is:

```text
nvidia/nemotron-3.5-lightning:free      verified: 1.27 s per call, tool-reliable
```

Verified with `python me3_laptop/agentcheck.py --model nvidia/nemotron-3.5-lightning:free`,
which checks the failure mode that matters: a model calls a tool and then denies
it did anything.

| Model | Call latency | Tool call |
|---|---|---|
| `nvidia/nemotron-3.5-lightning:free` | 1.27 s | ok |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 2.01 s | ok |
| `openrouter/free` | 3.92 s | ok |
| `google/gemma-4-26b-a4b-it:free` | - | 429 rate limited |

Free models are rate limited upstream, so a 429 in a turn means "wait a few
seconds or pick another free model", not a broken app.

`openrouter_order` is empty by default: free models must be routed to their own
provider. Set it (for example `"Wafer"`) only if you switch to a paid model and
see slow routing.

## The API key

The key is saved to `me3_laptop/openrouter.key` (mode 0600, never printed, never
written to `config.json` or the logs) and is found automatically at startup.
Discovery order: `OPENROUTER_API_KEY` env var, then that file, then the Pi CLI's
`~/.pi/agent/models.json`. Delete the file to stop reusing the key.

## Memory behaviour (idle models are freed)

Two things can hold memory: the language model inside Ollama, and the app's own
offline speech models. Both are released when they are not in use.

| Control | Where | Effect |
|---|---|---|
| **Model memory after a reply** | Memory box | `unload immediately (0s)` frees the model the moment a reply finishes; `keep 5 min` (default) keeps it warm for a conversation; `keep 30 min` is fastest but holds the most RAM |
| **Free memory now** | Memory box | Forces the release immediately |
| Idle watchdog | automatic | After 5 minutes without a turn the app frees its ASR/VAD models **and** asks Ollama to evict every resident model. It also logs it: `released idle memory after 312s idle: asr:moonshine, vad, ollama:qwen3.8-27b-iq3` |
| Live readout | Memory box | `app memory: … MB` and `models in RAM: …` straight from the process and from Ollama |

Everything reloads lazily on the next turn, so releasing costs a reload, never a
restart.

Measured by `python me3_laptop/memcheck.py`:

```text
1. app started              app memory: 106 MB    models in RAM: qwen3.8-27b-iq3
2. speech models loaded     app memory: 376 MB
3. one conversation turn    app memory: 376 MB
4. forced idle release      freed: asr:moonshine, vad, ollama:qwen3.8-27b-iq3
                            app memory: 131 MB    models in RAM: none
5. reload after release     ASR reloaded: ok
```

`modelcheck.py` proves every model with `keep_alive="0s"` and unloads whatever it
loaded before it exits, so checking the models does not leave gigabytes behind.

## Known limits

- **Local model speed:** a 9B local model answers in roughly 8-30 s on this
  laptop - fine for testing, too slow to feel conversational. Use OpenRouter
  (about 1 s) for smooth conversation, and for the Pi demo.
- **Tool calling needs a capable model.** Verified: `rafw007/qwen35-claude-coder:9b`.
  `llama3.2:3b` calls tools but tends to ignore the result, so it is not the default.
- **Echo:** the laptop mic hears the laptop speakers. Raise the barge-in
  threshold if the agent interrupts itself; headphones remove the problem.
- **Windows SAPI** is used for speech because it needs no download and no key.
  It is the first thing to swap for a real TTS on the Pi.
- **MCP servers are separate processes.** `asr_mcp.py`, `tts_mcp.py` and
  `tasks_mcp.py` speak JSON-RPC over stdio through `mcp_core.py`, and the protocol
  test client passes 25/25 checks. The same servers run on the Raspberry Pi.

## Natural voice (offline neural TTS)

The Voice dropdown now carries two kinds of voice: Windows SAPI, and offline
neural voices served by sherpa-onnx - the same library the ASR already uses, so no
new dependency is needed on the Raspberry Pi.

| Voice | Time to first sentence | Speed |
|---|---|---|
| `Microsoft David` (SAPI) | instant | robotic |
| **`Neural: en_US-lessac-high`** | **548 ms** | 4.27 s of speech rendered in 739 ms (RTF 0.17) |
| `Neural: kokoro-...` | ~1 s | higher quality, roughly 5x slower |

Two findings worth keeping:

* **int8 is the slow path on this CPU.** The int8 build of the *same* piper voice
  measured RTF 1.65 against 0.29 for fp32 - about six times slower - so the loader
  prefers non-int8 weights and hides an int8 folder when an fp32 sibling exists.
* **Speech is streamed one sentence at a time**, so the first sentence plays while
  the next is still being synthesised. That matters more for perceived latency
  than raw speed does.

Measure it: `python me3_laptop/benchmark.py --runs 5`, with the neural voice
selected in the app.

## Load-bearing agent harness

The installed agent harness does real work instead of sitting in the background:
`delegate_task` hands an open-ended job to `pi` running headlessly (`pi --print`)
and returns what it produced, so the 17 fixed device tools are not the ceiling.

Measured: the harness answered in 6.2 s and the agent read its poem back.

| Variable | Effect |
|---|---|
| `ME3_HARNESS_CMD` | harness command (default: `pi` found on PATH) |
| `ME3_HARNESS_SSH` | run it on the Pi instead, e.g. `lcsoneja@192.168.1.75` |
| `ME3_HARNESS_TIMEOUT_S` | deadline before the tool reports failure (default 180) |

On Windows the launcher resolves `pi.CMD` and runs it through `cmd.exe`, because
CreateProcess cannot start a batch file directly.


## Latency fast paths

Three additions make the assistant answer at conversational speed. All are on by default and
can be switched off individually in `config.json` to measure their effect.

| Setting | What it does | Measured effect |
|---|---|---|
| `filler_enabled` + `filler_delay_ms` | Plays a pre-rendered acknowledgement ("Okay.", "Hmm.", "Let me check.") if the real answer is not ready. Clips are rendered once in the active voice at startup, and one output stream is kept open because opening a WASAPI stream per clip costs 190-600 ms. | **first audio 512 / 1374 ms p50/p95** (stable playback) against 4774 / 5773 ms with no acknowledgement; the answer itself follows at 2433 ms. A persistent output stream reaches 260 ms but is **off by default**: Windows MME wedges a stream cut off mid-clip and closing it during a write segfaulted the process. Enable with `ME3_FILLER_STREAM=1` once that is properly solved |
| `fast_tool_replies` | For a single definite device command, the tool's own sentence is spoken instead of paying for a second LLM round. | removes ~1.6-2.2 s per command turn |
| `router_enabled` + `router_model` | A local model classifies the turn (command / chat / complex) and picks the acknowledgement. Keyword routing is used on the critical path; the model is only consulted when a turn needs a routing hint. | routing 0 ms (keywords) vs 450-930 ms (local model) |

Both arms of the measurement, on the same machine:

```text
fast paths OFF   first word 5.5-7.8 s   LLM 4589 ms p50   total 7839 ms p50
fast paths ON    first audio 263 ms     LLM 1919 ms p50   total 4807 ms p50
```

Reproduce it (no microphone or speaker needed - speech is simulated):

```bash
python turn_bench.py --label before --no-fast
python turn_bench.py --label after
```

The design reasoning, the human-conversation references and three local models that were tried
and rejected as agents are in `../LATENCY_HUMAN_TARGET.md`.


## Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama not reachable` | Start the Ollama app, or switch provider to openrouter. |
| No sound | Check the `out:` device line in the left panel; raise the speaking rate. |
| Nothing transcribed | Check the `in:` device; hold **Hold to talk** longer, or use continuous mode. |
| Agent answers itself | Raise the barge-in threshold, or use headphones. |
| Reply ignores what the tool did | Pick a stronger model in the dropdown (see limits above). |
