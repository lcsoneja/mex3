"""ME3 latency benchmark - the standardized measurement method (classwork).

The point of a standardized method is that two submissions can be compared, so
this fixes every boundary and reports percentiles rather than a single number:

    end of speech -> VAD closed the turn -> ASR text -> LLM first token (TTFT)
                  -> TTS audio ready -> response complete

The scripted mode needs no microphone: it synthesizes each test utterance with
the local TTS first, then feeds that audio through the real pipeline, so the
input is identical on every machine and every run.

The live mode reads me3_laptop/out/app.log, which records one line per real turn
(asr/llm/tts/total) and every barge-in decision, so real conversations can be
reported with the same definitions.

Run:
    python me3_laptop/benchmark.py                    :: scripted, 3 phrases, 2 runs
    python me3_laptop/benchmark.py --runs 5 --report out/benchmark_report.md
    python me3_laptop/benchmark.py --live-log out/app.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr import AsrError, create_asr
from audio import load_wav_16k
from console import use_utf8_stdout
from llm import LlmError, create_llm, find_openrouter_key
from tts import TtsError
from tts_mcp import active_backend, synthesize
from vad import SileroVad, VadError

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
OUT_DIR = BASE / "out"
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"

# Fixed utterances: short, one per ME2 command family, and deliberately plain so
# synthesis is reproducible.
UTTERANCES: list[str] = [
    "what time is it",
    "turn on the bedroom light",
    "set a timer for twenty seconds",
]
SYSTEM_PROMPT = (
    "You are a voice assistant. Reply in one short spoken sentence, no markdown."
)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, so a small sample still gives a sane number."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def stats(values: list[float]) -> str:
    """p50 / p95 / max in milliseconds."""
    if not values:
        return "-"
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    return f"{p50:.0f} / {p95:.0f} / {max(values):.0f}"


@dataclass
class Run:
    """One measured pass through the pipeline for one utterance."""

    utterance: str
    heard: str = ""
    reply: str = ""
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    ttft_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    speech_ms: float = 0.0
    error: str = ""

    @property
    def to_first_word_ms(self) -> float:
        """End of speech to the first LLM token: ASR plus TTFT."""
        return self.vad_ms + self.asr_ms + self.ttft_ms

    @property
    def to_first_audio_ms(self) -> float:
        """End of speech to audio being ready.

        On a non-streaming speech backend the whole phrase is rendered before the
        first sample can play; the report states that rather than hiding it.
        """
        return self.vad_ms + self.asr_ms + self.ttft_ms + self.tts_ms


def word_accuracy(heard: str, expected: str) -> float:
    """Fraction of expected words present in the transcript, for context."""
    want = [w for w in re.findall(r"[a-z0-9]+", expected.lower()) if w]
    got = set(re.findall(r"[a-z0-9]+", heard.lower()))
    if not want:
        return 0.0
    hits = sum(1 for word in want if word in got)
    return hits / len(want)


def real_speech_inputs(limit: int) -> list[tuple[Path, str]]:
    """Real human recordings that ship with the ASR model.

    Preferred over synthesized input: a synthesized prompt is only as good as the
    machine's voice, and on the Pi espeak's robotic audio made whisper mis-hear
    the phrase, which would look like an ASR defect in the results.
    """
    folder = MODELS_DIR / "sherpa-onnx-whisper-tiny.en" / "test_wavs"
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        return []
    expected: dict[str, str] = {}
    transcript_file = folder / "trans.txt"
    if transcript_file.exists():
        # Each line is "name.wav TRANSCRIPT" - one line, two fields.
        for line in transcript_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].endswith(".wav"):
                expected[parts[0]] = parts[1].strip()
    pairs: list[tuple[Path, str]] = []
    for wav in wavs[:limit]:
        pairs.append((wav, expected.get(wav.name, "")))
    return pairs


def run_scripted(args: argparse.Namespace) -> tuple[list[Run], dict[str, Any]]:
    """Feed synthesized audio through the real pipeline and time every stage."""
    OUT_DIR.mkdir(exist_ok=True)
    environment: dict[str, Any] = {}

    try:
        asr = create_asr(MODELS_DIR, backend=args.asr)
    except AsrError as exc:
        raise SystemExit(f"ASR unavailable: {exc}") from exc
    environment["asr"] = f"{args.asr} (sherpa-onnx, offline)"

    # Use the same backend chain as the TTS MCP server (remote, piper, espeak,
    # SAPI) so this benchmark runs unchanged on the Pi and on a laptop.
    backend = active_backend()
    if backend == "none":
        raise SystemExit(
            "TTS unavailable: set ME3_TTS_URL, install piper (with ME3_PIPER_MODEL) "
            "or install espeak-ng"
        )
    environment["tts"] = backend

    try:
        vad = SileroVad(VAD_MODEL, min_silence_duration=args.silence)
        environment["vad"] = f"silero, {args.silence:.2f} s silence ends a turn"
    except VadError as exc:
        raise SystemExit(f"VAD unavailable: {exc}") from exc

    try:
        llm = create_llm(
            args.provider,
            args.model,
            find_openrouter_key(),
            openrouter_order=args.order,
        )
    except LlmError as exc:
        raise SystemExit(f"LLM unavailable: {exc}") from exc
    environment["llm"] = llm.label

    # Input clips: bundled human speech when possible (identical on every
    # machine), otherwise synthesized prompts.
    real = real_speech_inputs(args.clips) if args.input == "real" else []
    environment["input"] = (
        f"{len(real)} bundled human recording(s)"
        if real
        else "synthesized prompts (depends on the machine's voice)"
    )

    runs: list[Run] = []
    for _ in range(args.runs):
        for phrase in args.phrase or UTTERANCES:
            run = Run(utterance=phrase)
            runs.append(run)

            # 1. deterministic input audio
            input_wav = OUT_DIR / "bench_input.wav"
            try:
                if real:
                    source, expected = real[len(runs) % len(real)]
                    run.utterance = expected or source.name
                    samples = load_wav_16k(source)
                else:
                    synthesize(phrase, path=str(input_wav))
                    samples = load_wav_16k(input_wav)
            except (TtsError, OSError) as exc:
                run.error = f"input audio failed: {exc}"
                continue
            run.speech_ms = len(samples) / 16000 * 1000

            # 2. VAD: how long until the turn is closed
            started = time.perf_counter()
            vad.reset()
            for offset in range(0, len(samples), vad.window_size):
                vad.accept(samples[offset : offset + vad.window_size])
            segments = vad.flush()
            run.vad_ms = (time.perf_counter() - started) * 1000
            clip = segments[0] if segments else samples

            # 3. speech to text
            started = time.perf_counter()
            run.heard = asr.transcribe(clip)
            run.asr_ms = (time.perf_counter() - started) * 1000
            if not run.heard:
                run.error = "empty transcript"
                continue

            # 4. language model: first token, then the whole reply
            try:
                stream = llm.chat_stream(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": run.heard},
                    ]
                )
            except LlmError as exc:
                run.error = f"LLM failed: {exc}"
                continue
            run.ttft_ms = stream.first_token_s * 1000
            run.llm_ms = stream.total_s * 1000
            run.reply = stream.text

            # 5. text to speech: audio ready
            reply_wav = OUT_DIR / "bench_reply.wav"
            started = time.perf_counter()
            try:
                synthesize(run.reply or "ok", path=str(reply_wav))
            except TtsError as exc:
                run.error = f"reply synthesis failed: {exc}"
                continue
            run.tts_ms = (time.perf_counter() - started) * 1000

            print(
                f"  {run.utterance[:38]!r} -> {run.heard[:38]!r}  "
                f"vad {run.vad_ms:.0f} asr {run.asr_ms:.0f} ttft {run.ttft_ms:.0f} "
                f"tts {run.tts_ms:.0f} ms"
            )
            if args.pause:
                time.sleep(args.pause)

    return runs, environment


def parse_live_log(path: Path) -> dict[str, Any]:
    """Per-turn and barge-in numbers recorded during real conversations."""
    turn_re = re.compile(
        r"turn (\d+) \[(\w+)\] asr=(\d+) llm=(\d+) tts=(\d+) total=(\d+) ms"
    )
    barge_re = re.compile(r"barge-in accepted \(rms=([\d.]+), (\d+) ms speech, (\d+) ms into the reply")
    ignored_re = re.compile(r"ignored while speaking \(rms=([\d.]+), (\d+) ms, (\d+) ms in\)")
    echo_re = re.compile(r"dropped echo of own speech")

    turns: list[dict[str, float]] = []
    barge: list[dict[str, float]] = []
    ignored = 0
    echoes = 0
    if not path.exists():
        return {"turns": turns, "barge": barge, "ignored": ignored, "echoes": echoes}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = turn_re.search(line)
        if match:
            turns.append(
                {
                    "id": float(match.group(1)),
                    "asr": float(match.group(3)),
                    "llm": float(match.group(4)),
                    "tts": float(match.group(5)),
                    "total": float(match.group(6)),
                }
            )
            continue
        match = barge_re.search(line)
        if match:
            barge.append(
                {
                    "rms": float(match.group(1)),
                    "speech_ms": float(match.group(2)),
                    "into_reply_ms": float(match.group(3)),
                }
            )
            continue
        if ignored_re.search(line):
            ignored += 1
        elif echo_re.search(line):
            echoes += 1
    return {"turns": turns, "barge": barge, "ignored": ignored, "echoes": echoes}


def build_report(
    runs: list[Run], environment: dict[str, Any], live: dict[str, Any], args: argparse.Namespace
) -> str:
    """Markdown report: method, environment, results, limitations."""
    ok = [run for run in runs if not run.error]
    lines: list[str] = []
    add = lines.append

    add("# ME3 latency benchmark report")
    add("")
    add(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    add(f"Mode: scripted ({len(runs)} runs of {len(args.phrase or UTTERANCES)} utterance(s))")
    add("")

    add("## Method")
    add("")
    add("Boundaries, fixed so results are comparable between machines:")
    add("")
    add("| Stage | Starts | Ends |")
    add("|---|---|---|")
    add("| VAD | first audio sample of the utterance | turn closed, ready to transcribe |")
    add("| ASR | turn closed | transcript text available |")
    add("| TTFT | transcript ready | first language-model token streamed |")
    add("| TTS | reply text ready | audio rendered and ready to play |")
    add("| **To first word** | end of speech | first LLM token (VAD + ASR + TTFT) |")
    add("| **To first audio** | end of speech | audio ready (VAD + ASR + TTFT + TTS) |")
    add("")
    add("Input audio is synthesized locally from a fixed phrase list, so the same")
    add("audio is used on every machine and every run. No microphone is involved.")
    add("")

    add("## Environment")
    add("")
    for key, value in environment.items():
        add(f"- **{key}**: {value}")
    add(f"- **provider order preference**: {args.order or 'automatic'}")
    add(f"- **turn silence window**: {args.silence:.2f} s")
    add("")

    add("## Results")
    add("")
    if ok:
        add("All values in milliseconds. p50 / p95 / max.")
        add("")
        add("| Metric | p50 / p95 / max |")
        add("|---|---|")
        add(f"| VAD close | {stats([r.vad_ms for r in ok])} |")
        add(f"| ASR (speech to text) | {stats([r.asr_ms for r in ok])} |")
        add(f"| TTFT (first LLM token) | {stats([r.ttft_ms for r in ok])} |")
        add(f"| LLM full reply | {stats([r.llm_ms for r in ok])} |")
        add(f"| TTS render | {stats([r.tts_ms for r in ok])} |")
        add(f"| **To first word** | {stats([r.to_first_word_ms for r in ok])} |")
        add(f"| **To first audio** | {stats([r.to_first_audio_ms for r in ok])} |")
        add("")
        add("### Per run")
        add("")
        add("| utterance | heard | reply | vad | asr | ttft | tts | to first audio |")
        add("|---|---|---|---|---|---|---|---|")
        for run in ok:
            add(
                f"| {run.utterance} | {run.heard} | {run.reply[:40]} | {run.vad_ms:.0f} | "
                f"{run.asr_ms:.0f} | {run.ttft_ms:.0f} | {run.tts_ms:.0f} | "
                f"{run.to_first_audio_ms:.0f} |"
            )
        accuracies = [word_accuracy(r.heard, r.utterance) for r in ok]
        add("")
        add(f"Transcript word coverage against the spoken phrase: {statistics.mean(accuracies) * 100:.0f}%")
    else:
        add("No successful runs - see errors below.")
    add("")

    failures = [run for run in runs if run.error]
    if failures:
        add("## Failures")
        add("")
        for run in failures:
            add(f"- {run.utterance!r}: {run.error}")
        add("")

    add("## Live conversation (from out/app.log)")
    add("")
    turns = live.get("turns") or []
    if turns:
        add(f"{len(turns)} real turn(s) recorded.")
        add("")
        add("| Metric | p50 / p95 / max |")
        add("|---|---|")
        add(f"| ASR | {stats([t['asr'] for t in turns])} |")
        add(f"| LLM | {stats([t['llm'] for t in turns])} |")
        add(f"| TTS | {stats([t['tts'] for t in turns])} |")
        add(f"| Total response | {stats([t['total'] for t in turns])} |")
    else:
        add("No live turns found in the log.")
    barge = live.get("barge") or []
    add("")
    if barge:
        add(f"**Barge-in**: {len(barge)} accepted interruption(s).")
        add("")
        add("| speech needed to trigger | time into the reply | rms |")
        add("|---|---|---|")
        for item in barge:
            add(f"| {item['speech_ms']:.0f} ms | {item['into_reply_ms']:.0f} ms | {item['rms']:.3f} |")
    else:
        add("**Barge-in**: none recorded yet.")
    add("")
    add(
        f"**Turn-taking guards fired**: {live.get('ignored', 0)} segment(s) ignored while speaking "
        f"(candidate false turns), {live.get('echoes', 0)} echo(es) of the agent's own speech dropped."
    )
    add("")

    add("## Limitations (stated, not hidden)")
    add("")
    add("- The speech backend here renders the whole phrase before audio can play, so")
    add("  *to first audio* includes full synthesis. A streaming voice would cut it.")
    add("- Scripted input is synthesized speech, not a live microphone, so acoustic echo")
    add("  and room noise are not represented; the live section covers those.")
    add("- Barge-in figures come from the running app's log and need a real conversation")
    add("  with a microphone to accumulate; the log records every decision it made.")
    add("- Percentiles over few samples are indicative; use `--runs 5` or more.")
    add("")
    add("## Reproduce")
    add("")
    add("```bat")
    add("python me3_laptop/benchmark.py --runs 5 --report me3_laptop/out/benchmark_report.md")
    add("```")
    add("")
    return chr(10).join(lines)


def main() -> int:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="ME3 latency benchmark")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning:free")
    parser.add_argument("--order", default="", help="OpenRouter provider preference")
    parser.add_argument("--asr", default="whisper", choices=["whisper", "moonshine"])
    parser.add_argument("--runs", type=int, default=2, help="passes over the phrase list")
    parser.add_argument("--phrase", action="append", default=[], help="custom utterance")
    parser.add_argument("--silence", type=float, default=0.8, help="VAD end-of-turn silence")
    parser.add_argument("--pause", type=float, default=0.5, help="pause between runs, seconds")
    parser.add_argument("--report", default="out/benchmark_report.md")
    parser.add_argument("--live-log", default="out/app.log")
    parser.add_argument(
        "--input",
        choices=["real", "synth"],
        default="real",
        help="real: bundled human recordings; synth: synthesize prompts locally",
    )
    parser.add_argument("--clips", type=int, default=3, help="how many human clips to use")
    args = parser.parse_args()

    print("ME3 latency benchmark (scripted)")
    runs, environment = run_scripted(args)
    live = parse_live_log(BASE / args.live_log)

    report_path = BASE / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(runs, environment, live, args), encoding="utf-8")

    ok = [run for run in runs if not run.error]
    print()
    print(f"runs: {len(runs)}   successful: {len(ok)}")
    if ok:
        print(f"  to first word : {stats([r.to_first_word_ms for r in ok])} ms  (p50/p95/max)")
        print(f"  to first audio: {stats([r.to_first_audio_ms for r in ok])} ms")
    print(f"report: {report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
