"""Headless self-test for the laptop audio stack (no microphone needed).

Covers what can be verified from files: ASR on the sample WAVs shipped with
sherpa-onnx models, VAD segmentation on the same audio, and a TTS -> WAV ->
ASR round trip through Windows SAPI. Run it before trusting the live app.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from asr import AsrError, create_asr, discover_models
from audio import input_devices, load_wav_16k, output_devices
from tts import SapiTts, TtsError
from vad import SileroVad, VadError

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
VAD_MODEL = MODELS / "silero_vad.onnx"
ROUND_TRIP_TEXT = "Set a timer for five minutes please."


@dataclass
class Report:
    """Collect the outcome of each check for the final summary."""

    results: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        symbol = {"ok": "PASS", "fail": "FAIL", "skip": "SKIP"}[status]
        print(f"  [{symbol}] {name}: {detail}")
        self.results.append((name, status, detail))

    @property
    def failed(self) -> bool:
        return any(status == "fail" for _, status, _ in self.results)


def sample_wavs(model_dir: Path) -> list[Path]:
    """Sample WAVs that ship inside a sherpa-onnx model folder."""
    folder = model_dir / "test_wavs"
    if not folder.is_dir():
        return []
    wavs = sorted(folder.glob("*.wav"))
    preferred = [w for w in wavs if w.stem == "0"]
    return preferred + [w for w in wavs if w.stem != "0"]


def reference_text(model_dir: Path) -> str:
    """First transcript line bundled with the sample WAVs, if any."""
    trans = model_dir / "test_wavs" / "trans.txt"
    if not trans.exists():
        return ""
    lines = [line.strip() for line in trans.read_text(encoding="utf-8").splitlines()]
    spoken = [line for line in lines if line and not line.endswith(".wav")]
    return spoken[0] if spoken else (lines[0] if lines else "")


def check_devices(report: Report) -> None:
    print("devices:")
    ins = input_devices()
    outs = output_devices()
    for dev in ins:
        print(
            f"  in  {dev.index:>3} {dev.name} ({dev.channels}ch @ {dev.default_rate:.0f} Hz)"
        )
    for dev in outs:
        print(
            f"  out {dev.index:>3} {dev.name} ({dev.channels}ch @ {dev.default_rate:.0f} Hz)"
        )
    report.add("microphones", "ok" if ins else "fail", f"{len(ins)} capture device(s)")
    report.add("speakers", "ok" if outs else "fail", f"{len(outs)} playback device(s)")


def check_asr(report: Report) -> dict[str, str]:
    """Transcribe each model's own sample audio; return backend -> transcript."""
    print("\nASR (offline, sherpa-onnx):")
    transcripts: dict[str, str] = {}
    models = discover_models(MODELS)
    if not models:
        report.add("asr", "fail", f"no models under {MODELS}")
        return transcripts

    for backend, folder in models.items():
        wavs = sample_wavs(folder)
        if not wavs:
            report.add(f"asr/{backend}", "skip", "no sample WAV in the model folder")
            continue
        try:
            engine = create_asr(MODELS, backend=backend)
        except AsrError as exc:
            report.add(f"asr/{backend}", "fail", str(exc))
            continue

        audio = load_wav_16k(wavs[0])
        started = time.perf_counter()
        text = engine.transcribe(audio)
        elapsed = time.perf_counter() - started
        transcripts[backend] = text

        audio_s = len(audio) / 16000
        realtime = elapsed / audio_s if audio_s else 0.0
        print(
            f"  {backend:10s} {audio_s:4.1f}s audio in {elapsed:5.2f}s (x{realtime:4.2f} realtime)"
        )
        print(f"             heard: {text!r}")
        expected = reference_text(folder)
        if expected:
            print(f"             expect: {expected!r}")
        report.add(
            f"asr/{backend}", "ok" if text else "fail", text or "empty transcript"
        )
    return transcripts


def check_vad(report: Report, source: Path | None) -> None:
    print("\nVAD (silero via sherpa-onnx):")
    if not VAD_MODEL.exists():
        report.add("vad", "fail", f"missing {VAD_MODEL}")
        return
    if source is None or not source.exists():
        report.add("vad", "skip", "no source audio available")
        return

    try:
        detector = SileroVad(VAD_MODEL)
    except VadError as exc:
        report.add("vad", "fail", str(exc))
        return

    audio = load_wav_16k(source)
    started = time.perf_counter()
    for start in range(0, len(audio), detector.window_size):
        detector.accept(audio[start : start + detector.window_size])
    segments = detector.flush()
    elapsed = time.perf_counter() - started

    speech_s = sum(len(seg) for seg in segments) / 16000
    audio_s = len(audio) / 16000
    print(
        f"  {len(segments)} segment(s), {speech_s:.2f}s speech of {audio_s:.2f}s audio"
    )
    for index, seg in enumerate(segments):
        print(f"    #{index}: {len(seg) / 16000:.2f}s")
    print(f"  processed in {elapsed * 1000:.0f} ms")
    report.add(
        "vad",
        "ok" if segments else "fail",
        f"{len(segments)} segment(s), {speech_s:.2f}s speech",
    )


def check_tts_round_trip(report: Report, asr_model: str | None) -> None:
    print("\nTTS round trip (Windows SAPI -> WAV -> ASR):")
    out = BASE / "out" / "selftest_tts.wav"
    try:
        tts = SapiTts()
    except TtsError as exc:
        report.add("tts", "fail", str(exc))
        return

    print(f"  voice: {tts.voice_name}")
    print(f"  voices available: {', '.join(tts.list_voices())}")
    started = time.perf_counter()
    tts.to_wav(ROUND_TRIP_TEXT, out)
    synth_s = time.perf_counter() - started
    report.add(
        "tts/synth", "ok" if out.exists() else "fail", f"{out.name} in {synth_s:.2f}s"
    )

    if asr_model is None:
        report.add("tts/round-trip", "skip", "no ASR backend available")
        return

    try:
        engine = create_asr(MODELS, backend=asr_model)
    except AsrError as exc:
        report.add("tts/round-trip", "fail", str(exc))
        return

    audio = load_wav_16k(out)
    started = time.perf_counter()
    heard = engine.transcribe(audio)
    elapsed = time.perf_counter() - started
    print(f"  said:  {ROUND_TRIP_TEXT!r}")
    print(f"  heard: {heard!r} ({elapsed:.2f}s, {len(audio) / 16000:.1f}s audio)")
    report.add("tts/round-trip", "ok" if heard else "fail", f"via {asr_model}")


def main() -> int:
    print("=" * 72)
    print("ME3 laptop audio self-test")
    print("=" * 72)
    report = Report()

    check_devices(report)
    transcripts = check_asr(report)

    preferred = (
        "moonshine" if "moonshine" in transcripts else next(iter(transcripts), None)
    )
    models = discover_models(MODELS)
    source = None
    for folder in models.values():
        wavs = sample_wavs(folder)
        if wavs:
            source = wavs[0]
            break

    check_vad(report, source)
    check_tts_round_trip(report, preferred)

    print("\n" + "=" * 72)
    passed = sum(1 for _, status, _ in report.results if status == "ok")
    skipped = sum(1 for _, status, _ in report.results if status == "skip")
    failed = sum(1 for _, status, _ in report.results if status == "fail")
    print(f"summary: {passed} passed, {failed} failed, {skipped} skipped")
    if report.failed:
        for name, status, detail in report.results:
            if status == "fail":
                print(f"  FAILED {name}: {detail}")
    print("=" * 72)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
