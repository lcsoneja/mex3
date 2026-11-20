"""B3: how fast does a barge-in actually fire?

Measures the wall-clock time from speech audio being judged to interrupt() being
called. Two numbers per run:

  * machinery only - stub recogniser, so the time is the RMS gate + decision,
    which is the part the app controls per 32 ms block;
  * with the REAL ASR - the transcription of the barge-in audio is the latency
    floor, so this is the honest end-to-end detection time.

The spec target is 75 ms detection. This check records the measured numbers and
fails only if the cut does not fire at all - the gap to the target is reported,
not hidden.

Run:  python me3_laptop/bargelatencycheck.py
"""
from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio import load_wav_16k
from server import Session, load_settings
from tts_mcp import synthesize

REPLY = "Yes, I hear you. I am here and ready to help. What can I do for you?"
RUNS = 5


class _FakeAsr:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcript


class _CutClock:
    """Replaces session.interrupt and stamps the moment the cut fires."""

    def __init__(self) -> None:
        self.at: float | None = None

    def __call__(self) -> None:
        self.at = time.perf_counter()


def harness(session: Session, transcript: str | None) -> tuple[_CutClock, Any]:
    """bargecheck's setup: speaking, mid-reply, ASR swapped, cut observable."""
    session._speaking = True
    session._speech_started_at = time.perf_counter() - 2.0
    session._last_spoken = REPLY
    session._last_spoken_at = time.perf_counter()
    session._barge_judged_at = 0.0
    if transcript is None:
        session._asr = None  # force the real recogniser to load
    else:
        session._asr = cast(Any, _FakeAsr(transcript))
    session._asr_backend = session.settings.asr_backend
    clock = _CutClock()
    session.interrupt = cast(Any, clock)  # type: ignore[assignment]
    session._pending.put = cast(Any, lambda item: None)
    return clock, session


def measure(session: Session, audio: np.ndarray, fake: str | None) -> list[float]:
    times: list[float] = []
    for _ in range(RUNS):
        clock, h = harness(session, fake)
        start = time.perf_counter()
        h._judge_interruption(audio)
        took = (clock.at - start) * 1000.0 if clock.at is not None else None
        if took is None:
            print("  FAIL: interrupt() never fired")
            sys.exit(1)
        times.append(took)
        session.barge_ins = 0
        session._speech_cut = False
    return times


def main() -> int:
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(exist_ok=True)
    clip_path = OUT / "barge_stop.wav"
    synthesize("Stop. Stop talking.", voice="", rate=0, path=str(clip_path))
    audio = load_wav_16k(clip_path)
    print(f"clip: {clip_path.name} {len(audio) / 16000:.2f}s rms="
          f"{float(np.sqrt(np.mean(np.square(audio)))):.3f}")

    session = Session(load_settings())
    session.settings.tts_enabled = False
    print("loading the real recogniser for the honest measurement...")
    session.asr()  # warm the ASR outside the timed path

    real = measure(session, audio, fake=None)
    machinery = measure(session, audio, fake="stop")

    def row(label: str, times: list[float]) -> None:
        print(f"  {label:28s} min {min(times):7.1f} ms   median "
              f"{statistics.median(times):7.1f} ms   max {max(times):7.1f} ms")

    print(f"measured over {RUNS} runs each:")
    row("decision machinery only", machinery)
    row("with the real ASR", real)

    asr_floor = statistics.median(real) - statistics.median(machinery)
    print(f"  ASR floor (median real - median machinery): ~{asr_floor:.0f} ms")
    target = 75.0
    best = min(real)
    print(f"  spec target 75 ms: {'MET' if best <= target else 'NOT MET'} "
          f"(best real {best:.1f} ms)")

    ok = best < 5000.0  # a cut that takes seconds is a broken path, not a target miss
    print("RESULT: " + ("PASS (cut fires; see numbers above)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
