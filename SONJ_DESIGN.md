# SONJ — product design (ME3 delivery vehicle)

_Design by the multi-agent company; status claims are checked against the repository, not assumed._

## 1. What we are building

**SONJ** is one product with **two faces over one engine**:

| Face | Who it is for | Shape | Runs on |
|---|---|---|---|
| **`sonj` CLI** | you, over SSH, on the Pi | lightweight, scriptable, ~200 ms to start; can also *open* the GUI | anywhere |
| **SONJ Console (GUI)** | you, at the desk | the pi.dev-style browser console already built: chat, thinking, subagents, skills, MCP, settings | laptop now, Pi via `sonj ui --host` |

The CLI is not a lesser product: it is how the Pi stays usable when a browser is too heavy (Pi 4 GB).
The GUI is the *inspection* surface: it shows what the agents are doing instead of claiming it.

**Non-negotiable:** both faces read and write the **same engine state** (one session log, one task
board, one settings file), so a turn started in the GUI is visible in the CLI and vice versa.

---

## 2. Architecture — two layers, one engine

```
┌────────────────────────── LAYER 1 · SPEAKER ───────────────────────────┐
│  mic → VAD → ASR → router → ack clip → TTS(Nova|Piattos) → speakers     │
│  barge-in (stop words) · echo guard · phantom-speaking guard            │
│  RULE: never blocks. Owns all audio. Target: first sound < 300 ms.      │
└───────────────┬────────────────────────────────────────────────────────┘
                │  speech out ↑        intents + status ↓
┌───────────────┴─────────── LAYER 2 · TASK COMPANY ─────────────────────┐
│  Orchestrator (the "company"): routes work to specialist agents         │
│  ├ Nova      — speaker/front desk: fast answers, device commands        │
│  ├ Piattos   — foreman: long tasks, code, multi-step plans              │
│  ├ Researcher— web search, reading, citations                           │
│  ├ Builder   — writes files, code, documents                            │
│  ├ Reviewer  — checks the Builder's work (independent)                  │
│  └ Auditor   — verifies claims against evidence (independent)           │
│  + MCP servers (ASR, TTS, tools) + Skills + tool registry               │
└───────────────┬────────────────────────────────────────────────────────┘
                │
┌───────────────┴───────────────────── CORE ENGINE ───────────────────────┐
│  state() · sessions/events.jsonl · task board · settings · ledger       │
│  supervisor (reconnect, self-clean) · health endpoints · bench harness  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this shape:** the latency targets are only reachable if Layer 1 never waits for Layer 2.
A command that needs no thought is answered in Layer 1 (local rules/VCM path) in well under 300 ms;
anything that needs work is handed to Layer 2, acknowledged instantly ("on it"), and **spoken when
ready** — the interface between layers is a queue of «say this when you can», never a blocking call.

---

## 3. The multi-agent company (operating model)

| Role | Voice | Job | Handoff trigger | Handoff target |
|---|---|---|---|---|
| **Nova** | af_heart (default) | greet, take the request, answer instantly, run device commands | request needs tools, research, code, or >1 step | Piattos |
| **Piattos** | am_michael (on call) | own long tasks, delegate to specialists, report back | a specialist can do it faster in parallel | Researcher / Builder |
| **Researcher** | silent (reports) | web + file reading, always cites sources | findings ready | Reviewer or Nova |
| **Builder** | silent | writes code/documents, never edits outside an approved path | artefact written | Reviewer |
| **Reviewer** | silent | independent check of the artefact | pass or defect list | Piattos |
| **Auditor** | silent | verifies claims against evidence; may refuse | verdict | Piattos |

Rules the company obeys (learned in this project, not invented):
1. **One writer per file/worktree.** Reviewers and auditors never write.
2. **A task is Done only when its check ran.** The ledger (`REQUESTS.md`) is the truth.
3. **Handoffs are bounded** — max depth 3, no loops, unknown target = refuse and report.
4. **Every handoff is visible** in the transcript and the task board (agent, reason, time).
5. **Speech never blocks work, and work never blocks speech.**

---

## 4. GUI (SONJ Console)

Already built and working (pi.dev-shaped): rail (Chat · Sessions · Files · Agents · Skills · MCP ·
Thinking · Settings), two-sided message pane, inline tool cards, thinking stream, subagent/workflow/
mind-map tabs, model picker, speaker waveform, slash commands.

**Additions this design requires:**
| Pane | Addition | Why |
|---|---|---|
| **Task board** | one card per agent run with its **own conversation thread** (prompt → messages → tools → report) | the owner asked for sub-agents with their own conversation |
| **Sources** | drag-and-drop files, checkboxes, citations; feeds the notebook chat | feature (i): copilot-like sources |
| **Personas** | Nova / Piattos cards + custom persona creator (name, voice, instructions, tool scope) | the owner's persona requirement |
| **Reactions** | like/✗ on any turn or agent report; stored per turn | feature (j) |
| **Health** | supervisor status: uptime, restarts, memory, last error, self-clean actions | so a problem is visible before it becomes silence |

---

## 5. CLI — `sonj`

Installed as a single command; every subcommand returns a process exit code and, with `--json`, a
machine-readable result, so the Pi can be driven from scripts and over SSH.

```
sonj                     # start the engine and open the GUI (default: friendly)
sonj ui  [--host 0.0.0.0] [--port 8765]
sonj listen [--persona nova|piattos]   # hands-free voice session (Layer 1 only)
sonj ask  "turn on the bedroom light"  # one turn, prints the answer
sonj say  "text"                       # speak without a turn (voice test)
sonj task "summarise ~/docs/report.pdf"  # hand to Layer 2; --wait to block, --json for status
sonj status [--json]                   # engine, audio, agents, memory, last error
sonj agents                            # the company: roles, models, running tasks
sonj personas [--add name --voice kokoro:af_bella]
sonj mcp list | call <server> <tool> [json]
sonj skills list | show <name> | run <name>
sonj bench [--turns 20] [--json]       # the standardized latency method, one command
sonj doctor                            # self-check: devices, models, disk, RAM, permissions
sonj logs [--follow] [--since 10m]     # reads the same event stream the GUI shows
sonj stop | restart
```

Design rules: plain text by default, `--json` for automation; nothing interactive that cannot be
driven by flags (SSH-friendly); every command is safe to re-run; `sonj doctor` must succeed before
any deployment is called done.

---

## 6. Pi 4 (4 GB) profile

The laptop build is not the Pi build. On the Pi, memory and startup time are the constraints.

| Piece | Laptop today | Pi profile | Why |
|---|---|---|---|
| TTS | Kokoro int8 multi-voice (~100 s of MB + slow load) | **Piper VITS** (single voice, small, fast) | fits 4 GB; less RAM, faster start; persona variety comes from two piper voices or two kokoro voices loaded one at a time |
| ASR | Whisper tiny.en | **Moonshine base int8** or whisper tiny.en int8 | already supported; pick by measured RAM/accuracy |
| VAD | silero (sherpa-onnx) | same | tiny |
| GUI | browser console | **served on demand** (`sonj ui --host`); default is CLI | a Chromium tab is the single biggest RAM consumer on a Pi |
| Models in RAM | one idle-unload policy | idle unload after 60 s | keeps the Pi responsive |
| Startup | ~12 s engine boot | target < 5 s; models load lazily | CLI must feel instant |

Memory budget to defend: engine + VAD + ASR (int8) + piper + Python **≤ 1.2 GB**, leaving headroom
for the OS and the LLM client. Measured with `sonj status --json` (reports RSS) on the device.

Packaging: `install.sh IS NOT NEEDED IN E3`… replaced by — **`sonj install-pi`** which writes a
systemd unit (`sonj.service`, restart=always, backoff 5 s), enables it, and prints the `sonj doctor`
result. Reconnect is therefore a service property, not an application feature.

---

## 7. Robustness — reconnect and self-cleaning

| Failure seen in this project | Mechanism | Trigger |
|---|---|---|
| Engine died silently (native ASR abort) | **input clamp to 25 s** + watchdog restart with backoff | process exit |
| Reply cut by its own echo | content check against the sentence being spoken | every barge judgement |
| "speaking" flag stuck with no audio | clip-completion clears the flag; supervisor asserts `speaking` false when no stream is open | 5 s without audio while `speaking` |
| Mic wedged (device busy / no data) | open/close cycle; after 3 failures, switch input device and log | 3 s with no blocks |
| Model OOM on the Pi | unload on idle; on MemoryError unload everything, reload the smallest ASR, retry once | MemoryError |
| Cloud model unreachable | fallback list (already configured), then speak a short apology and keep listening | HTTP error / timeout |
| Session left mid-turn | on restart, mark the turn failed in the event log and resume listening | startup |

Rule: **every self-clean action appears in the Thinking pane and in `sonj logs`** — silent recovery is
indistinguishable from a bug.

---

## 8. Latency plan (how the targets are approached)

| Target | Mechanism | Which layer |
|---|---|---|
| < 300 ms **first sound** | pre-rendered acknowledgement clips (already ~0.4 ms to play); greeting cached | Layer 1 |
| < 250 ms **first response** when immediate | local intent fast path: device commands answered from the tool result text, no LLM round | Layer 1 |
| < 75 ms **barge-in detection** | level gate on 20 ms frames with a 75 ms speech run, before any ASR | Layer 1 |
| < 100 ms **reaction** | stop-word path short-circuits content check for the trigger list; audio stops on the first match | Layer 1 |
| End-to-end < 300 ms | **only reachable for local intents**; for LLM answers the honest figure is "acknowledged in ~300 ms, answered when ready" | decision needed |

The benchmark (`sonj bench`) reports: acknowledgement latency p50/p95, first-response p50/p95,
turn total, barge-in detection, and the share of turns answered locally vs by the cloud model.

---

## 9. Phases and gates

| Phase | Deliverable | Gate (must pass) |
|---|---|---|
| **P1** | `sonj` CLI skeleton + `doctor` + `status --json` | doctor green on laptop; status reports RSS and devices |
| **P2** | Pi 4 build (piper + tiny ASR + CLI default) | one spoken turn on the Pi; RSS ≤ 1.2 GB |
| **P3** | MCP servers on the Pi (ASR + TTS) | the 25-check protocol client passes **on the Pi** |
| **P4** | Harness consumes MCP; task board + per-agent threads in the GUI | a task dispatched to Piattos shows its own conversation, start to report |
| **P5** | Persona switching + custom personas; sources (drag-drop) | Nova default, Piattos reachable by name, one custom persona created and used |
| **P6** | Standardized benchmark + report | `sonj bench --turns 20` reproducible; p50/p95 table in the submission |

Stretch after P6: coding sandbox, API connectors, messaging/email, likes, NotebookLM-style Audio
Overview with Nova + Piattos.

---

## 10. Open decisions (owner input needed)

1. **Latency semantics** — is the 300 ms target *first sound* (achievable with acknowledgements) or
   *complete answer* (requires local intents for everything)? This changes P1–P2 scope.
2. **Default face** — GUI-first on the laptop, CLI-first on the Pi? (Recommendation: yes.)
3. **Persona scope** — should Piattos be a *different personality with its own tools* or the same
   brain with a male voice? (Recommendation: same brain, different voice and style, until P5.)
4. **Submission feature freeze** — confirm the must-have subset (suggest a0, a1, a, f, h, i) so the
   open-ended list cannot threaten the deadline.
