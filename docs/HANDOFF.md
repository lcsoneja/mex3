# SONJ4 — Handoff

Version: see `VERSION` (4.0.0). Read with `docs/INSTALL.md`.

## Consolidated Agent Architecture

| Component | Count | Type | Role |
|---|---:|---|---|
| **Front** | 1 | LLM Pi pane | Voice companion & coordinator (<2s turns, never slow work) |
| **Decision Classifier** | 1 | Non-LLM (`classifier.py`) | Sits between user and Front; <100ms hard deadline; +10% Front bias; choices: `front` \| `back` \| `doctor` |
| **Doctor** | 1 | LLM Pi pane + `doctor.py` | App diagnostics, logs, settings proposals (15s confirm), fixed catalogue repairs |
| **Back Pool** | 5 | Always-warm LLM Pi panes (`back-1..back-5`) | Unified generalist workers: build, web research, critic/verify, summarize, desktop/browser MCP, peer A2A |

## Key Modules

- `classifier.py` — `DecisionRequest` -> `Decision` (`<100ms`, Stage 1 rules + Stage 2 ONNX slot under `models/classifier/`, `+0.10` Front bias).
- `server.py` — Session, audio loop, Decision Classifier integration, 5-worker idle scheduler, always-warm pool (`warm_all_panes()`), task announce/TTS pipeline (`_deliver_task_report` + `_announce_pending_once`).
- `tasks.py` — Background task/crew queue (`max_parallel: 5`).
- `.pi/extensions/sonj_bridge.ts` — Bridge tools for Front (`classify_decision`, `delegate_task`, `ask_agent`, `start_crew`, `device_tool`), Back (`device_tool`, `mcp_call`, `ask_agent`, `post_community`), and Doctor (`doctor_app_control`, `ask_agent`, `post_community`).
- `sonj4_eval20.py` — 20-task evaluation suite (6 Front, 6 Back, 3 Doctor, 5 Coordination/Community independence).

## Verification Suite

- `python classifiercheck.py` — DecisionClassifier contract, <100ms latency, +10% Front bias, 20 routes (27 checks).
- `python announcecheck.py` — Task persistence & speech regression (9 checks).
- `python sonj4_eval20.py` — Full 20-task SONJ4 harness (`--live` against SONJ4 on default port `--port 8766`).
- `python bargecheck.py` & `python doctorcheck.py`.
