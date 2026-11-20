# SONJ4 — Install Guide

SONJ4 runs on **Windows 10/11, Linux, Raspberry Pi OS, and macOS**. Speech
recognition, synthesis, VAD, and the `<100ms` Decision Classifier run locally;
only the LLM panes and optional web search use the network.

## Raspberry Pi — one command

On the Pi, `install.py --pi` does almost everything for you (it uses `sudo` once
for the system audio libraries and Node.js):

```bash
git clone https://github.com/lcsoneja/sonj4.git   # sign in to GitHub if asked
cd sonj4
python3 install.py --pi
```

That installs the audio system libraries, Node.js 20 + the Pi CLI, the Python
environment, the light speech models, and points `config.json` at them. Then the
**only two steps left** (the sign-in is interactive):

```bash
pi          # one-time sign-in to your model provider
./sonj4.sh  # start SONJ4  ->  http://127.0.0.1:8766
```

> **Not a programmer?** Just tell your Raspberry Pi's coding agent this one sentence:
>
> *"Clone my GitHub repo `lcsoneja/sonj4`, go into the folder, and run
> `python3 install.py --pi` to set everything up; when it finishes, tell me in plain
> language whether it worked and the one thing I still need to do."*
>
> (If the repo is private, the Pi will ask you to sign in to GitHub once.)

## Windows / macOS / Linux laptop

```bash
python install.py            # creates .venv, installs deps + models, runs doctor
# then:
sonj4.bat                    # Windows
./sonj4.sh                   # Linux / macOS
```

For the LLM agents on a laptop, install + sign in to the Pi CLI yourself:
`npm install -g @earendil-works/pi-coding-agent` then `pi`.

SONJ4 defaults to port **8766**, so it never replaces SONJ3 on 8765.

## Prerequisites & credentials
- **Python 3.10–3.12**, `pip`, `curl`, `tar`, mic + speakers (~2 GB disk).
- On a Pi, `install.py --pi` handles the system libraries and Node for you.
- **Model provider:** the Pi CLI sign-in (`pi`) uses GitHub Copilot by default.
  Optional OpenRouter key: `OPENROUTER_API_KEY` env var or `sonj4/openrouter.key`
  (never committed / shipped).

## Verify
```bash
python sonj.py doctor
python classifiercheck.py
python announcecheck.py
python sonj4_eval20.py
```

## Notes
- The Pi profile (light models: moonshine ASR + Piper voice) is applied both by
  `install.py --pi` (writes `config.json`) and at runtime when `SONJ_PROFILE=pi`.
- Verified on Windows; the Pi/Linux path is implemented but should be confirmed on
  the actual device (`python sonj.py doctor`).
