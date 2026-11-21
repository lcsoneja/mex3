# SONJ4 — Classwork Latency Benchmark & Standardized Method

> **Classwork deliverable** (`Instructions.txt`: *"Build a latency benchmark (classwork): a standardized latency measurement method."*).  
> Full stage benchmark runner: [`benchmark.py`](../benchmark.py) · Live API runner: [`benchrun.py`](../benchrun.py) · Turn-path A/B: [`turn_bench.py`](../turn_bench.py) · Full duplex test: [`duplex_test.py`](../duplex_test.py) · Classifier check: [`classifiercheck.py`](../classifiercheck.py) · Barge-in check: [`bargelatencycheck.py`](../bargelatencycheck.py) · Deck Section 5: [`MEX3_Deck.pdf`](../MEX3_Deck.pdf)

---

## 1. Standardized Monotonic-Clock Method & Timing Boundaries

All measurements use a single monotonic clock (`time.perf_counter()`) with fixed boundaries:

```
VAD end (end of speech) -> ASR final transcript -> LLM 1st token (TTFT) -> TTS 1st audio (TTFA / TTFB) -> Speaker playback
```

| Metric | Boundary / Definition | Source / Engine Field | Target |
|---|---|---|---|
| **Decision Classifier** | Text -> `Decision(route)` (rules-v1 + optional ONNX) | `classifier.stats.p50_latency_ms`, `p95_latency_ms` | **< 100 ms** |
| **ASR final** | VAD end -> transcript text | `timings.asr_ms` | **< 300 ms** |
| **LLM TTFT** | Transcript -> 1st streamed token | `timings.first_token_ms` | < 1500 ms |
| **TTS TTFB / TTFA** | Text -> 1st audio chunk | `timings.ttfa_ms`, `tts.first_audio_at` | **< 600 ms** |
| **E2E / Ack** | VAD end -> 1st audible sound (ack or answer) | `timings.e2e_ms`, `timings.filler_ms` | Ack **< 300 ms** |
| **Barge-in** | User speech -> TTS + LLM stream cut | `bargelatencycheck.py` machinery / real ASR | Gate **< 75 ms** |
| **WER** | ASR word accuracy on reference test clips | `asr_bench` / `benchmark.py` | **< 5%** |

- **Statistics:** Report **p50** (median) and **p95** (nearest rank `ceil(0.95 * n)`), plus **max** and valid sample size `n` per stage. Exclude unscored warm-up turns; missing/zero stamps are excluded rather than counted as zero latency.
- **Standard sample size:** `n >= 20` measured turns for acceptance runs (`benchrun.py --turns 20 --url http://127.0.0.1:8766`); `n = 5` passes for scripted stage runs (`benchmark.py --runs 5`).

---

## 2. Run Commands (Laptop & Raspberry Pi)

```bash
# 1. Scripted stage benchmark (real human / synthesized WAVs through VAD -> ASR -> LLM -> TTS, no mic needed)
python benchmark.py --runs 5 --report out/benchmark_report.md

# 2. Standardized live turn benchmark against running SONJ4 on :8766 (n >= 20 turns)
python benchrun.py --turns 20 --url http://127.0.0.1:8766

# 3. Turn-path A/B with fast acknowledgements & streamed voice
python turn_bench.py --turns 6

# 4. Full-duplex turn-taking, barge-in, quiet-noise, and echo rejection
python duplex_test.py --report out/duplex_report.md

# 5. Non-LLM Decision Classifier (<100 ms) & barge-in latency checks
python classifiercheck.py
python bargelatencycheck.py
python bargecheck.py
```

---

## 3. Verified SONJ4 Benchmark Results (Windows 11 x64 Host, 2026-11-20/21)

*Environment: Windows 11 x64, 33.8 GB RAM, Python 3.12, `sherpa-onnx` 1.13.8, Parakeet 0.6B ASR + Piper lessac TTS, GitHub Copilot API (`gpt-5.4-nano` / `gpt-6-luna`).*

| Metric | Measured Mean / Median | p95 / Max | Status vs Target |
|---|---|---|---|
| **Decision Classifier** | **0.01 ms p95** (1,000 rules runs) | **0.035 ms max** | **MET** (`< 100 ms`) |
| **AEC + NS DSP** | **23.9 dB ERLE** (`nlms-aec`) / **8.6 dB NS** (`wiener-ns`) | `dt_rms = 0.198` (double-talk) | **MET** (`> 6 dB`) |
| **ASR final** | **192 ms** (114 ms p50, `n=22` live) | **488 ms p95** | Mean **MET** (`< 300 ms`) |
| **TTS TTFB (Piper)** | **496 ms** (`n=5`) | **546 ms max** | **MET** (`< 600 ms`) |
| **E2E / Ack** | **259 ms ack** / 3.58 s full answer (`n=25–35`) | 268 ms ack p95 / 21.3 s max | Ack **MET** (`< 300 ms`) |
| **Barge-in gate** | **0.7 ms machinery median** | **2.7 ms max** (real ASR floor ~1.78–2.0 s) | Gate **MET** (`< 75 ms`); real ASR content floor disclosed |
| **WER** | **1.5% Parakeet** · **3.4% Moonshine** | 0–9% per clip | **MET** (`< 5%`) |
| **LLM TTFT** | 2,955 ms (cloud API) | 7,950 ms p95 | Cloud API bound |

---

## 4. Fixed Prompt Set & Full-Duplex Tests

- **`benchrun.py` fixed 10 prompts** (2 passes for `n = 20`):
  1. What time is it in Manila?
  2. Calculate 17 times 23.
  3. Turn on the bedroom lights at 40 percent.
  4. Set the kitchen lights to off.
  5. Dim the living room lights by 20 percent.
  6. Set the bedroom temperature one degree cooler.
  7. Pause the music.
  8. Turn the volume down by 10 percent.
  9. Brighten the living room lights by 10 percent.
  10. What time is it?
- **`duplex_test.py` 4 scenarios:** (1) turn taking, (2) barge-in while speaking, (3) quiet room noise rejection, (4) self-echo rejection.

---

## 5. Raspberry Pi 4/5 (4 GB) On-Device Run (Pending Hardware Verification)

The RPi 4/5 profile (`install.py --pi`) pins lightweight local models (`Moonshine ASR` + `Piper lessac TTS` + `Silero VAD`). The benchmark scripts are portable and run on-device without microphone injection:

```bash
# On Raspberry Pi 4/5 after git pull origin main & python3 install.py --pi:
./sonj4.sh   # starts engine on :8766
python3 benchmark.py --runs 5 --report out/benchmark_report.md
python3 benchrun.py --turns 20 --url http://127.0.0.1:8766
python3 duplex_test.py
```

> Record RPi model (Pi 4 vs Pi 5), RAM, active cooling, network, and date in `out/benchmark_report.md` and update Section 5 of `MEX3_Deck.pdf` once measured on hardware.
