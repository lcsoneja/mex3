# SONJ4 — Install Guide

SONJ4 runs on **Windows 10/11, Linux, Raspberry Pi OS, and macOS**. Speech
recognition, synthesis, VAD, and the `<100ms` Decision Classifier run locally;
only the LLM panes and optional web search use the network.

## Quick Install (All Platforms)

From `sonj4/`:

```bash
python install.py            # Windows / Linux / macOS (laptop profile)
python install.py --pi       # Raspberry Pi OS (lighter models)
```

Start SONJ4 (defaults to port `8766`, leaving `sonj3` on `8765` untouched):
- Windows: `sonj4.bat` (or `python sonj.py`)
- Linux / macOS / Pi: `./sonj4.sh` (or `python3 sonj.py`)

## Prerequisites & Credentials
- **Python 3.10–3.12**, `pip`, `curl`, `tar`, mic + speakers (~2 GB disk).
- **Pi CLI** (for Front/Doctor/Back LLM panes): `npm install -g @earendil-works/pi-coding-agent` then `pi` to sign in (GitHub Copilot default).
- Optional OpenRouter key: `OPENROUTER_API_KEY` env var or `sonj4/openrouter.key` (excluded from release zips).

## Verify
```bash
python sonj.py doctor
python classifiercheck.py
python announcecheck.py
python sonj4_eval20.py
```
