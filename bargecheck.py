"""Does the interruption decision still cut replies off with the reply itself?

Regression check for the live-session defect of 2026-11-08: while the agent spoke, the
microphone heard the app's own voice back at rms 0.065-0.081 against a 0.06 threshold,
the level gate accepted it as a person, and replies stopped mid-sentence - so no
conversation ever completed. The decision now transcribes the captured run and compares
it with the sentence being spoken, and only then cuts the reply.

Run:  python me3_laptop/bargecheck.py
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")


import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentReply
from server import Session, load_settings

REPLY = "Yes, I hear you. I'm here and ready to help. What can I do for you?"
MEASURED_ECHO_RMS = (
    0.065  # what the speaker's own voice actually measured through the mic
)
QUIET_RMS = 0.02  # below the interruption threshold: not a person, not an echo


class _FakeAsr:
    """Stands in for the recogniser so the decision can be tested without audio."""

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcript


class _FakeAgent:
    """Stands in for the model so a whole turn can be run without a provider.

    It returns the app's own AgentReply rather than a hand-rolled double, so this
    test cannot drift away from the shape the turn logic actually reads.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def respond(self, text: str, on_delta: Any = None, **kwargs: Any) -> AgentReply:
        # The engine now streams (M2.2): the double plays the same role a provider
        # does - it calls on_delta with token pieces, and the pump turns them into
        # spoken sentences. Provider errors can now propagate too (raise_errors).
        if on_delta is not None:
            for piece in self._pieces():
                on_delta(piece)
        return AgentReply(text=self.text)

    def _pieces(self) -> list[str]:
        # A few words per piece, like a real token stream.
        words = self.text.split(" ")
        pieces: list[str] = []
        for index in range(0, len(words), 3):
            pieces.append(" ".join(words[index : index + 3]) + " ")
        return pieces


class _RecordingVoice:
    """Stands in for the voice and records what the turn believed it was saying.

    The independent audit found the fix ineffective live because the reply text was
    recorded only *after* the wait, so the microphone check compared captured audio
    against the previous sentence. This class exists to pin that ordering down: it
    reads the session's field at the moment synthesis starts, exactly as the echo
    check does.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.seen_at_say = "<say was never called>"
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)
        self.seen_at_say = self.session._last_spoken

    def wait(self, timeout_s: float = 30.0) -> None:
        return None

    def stop(self) -> None:
        # The streamed-speech pump cuts the voice through this method (M2.2).
        return None

    def speaking(self) -> bool:
        return False


class _Harness:
    def __init__(self, transcript: str) -> None:
        self.session = Session(load_settings())
        self.session._speaking = True
        self.session._speech_started_at = time.perf_counter() - 2.0
        self.session._last_spoken = REPLY
        self.session._last_spoken_at = time.perf_counter()
        self.session.settings.wake_speaker_lock = False
        self.session.settings.wake_to_interrupt = False
        self.session.settings.echo_cancellation = False
        # Stand-ins for the real dependencies: the decision is under test, not the
        # recogniser, and a cut has to be observable without playing audio.
        self.session._asr = cast(Any, _FakeAsr(transcript))
        self.session._asr_backend = self.session.settings.asr_backend
        self.cuts: list[str] = []
        self.queued: list[object] = []
        self.session.interrupt = cast(Any, lambda: self.cuts.append("interrupt"))
        self.session._pending.put = cast(Any, self.queued.append)

    def judge(self, rms: float = MEASURED_ECHO_RMS) -> None:
        audio = np.full(16000, rms, dtype=np.float32)
        self.session._judge_interruption(audio)

    def rewind_clock(self, seconds: float = 0.8) -> None:
        """Pretend the throttle window has passed, as it does for ongoing speech."""
        self.session._barge_judged_at = time.perf_counter() - seconds


def main() -> int:
    threshold = load_settings().barge_rms
    print(
        f"interruption threshold {threshold}, probe audio {MEASURED_ECHO_RMS} (the measured echo)"
    )
    failures = 0

    def report(title: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {title:58s} {detail}")

    # 1-2. the app's own voice, exactly what the live session produced
    for title, transcript in (
        ("own voice, transcript equals the reply", REPLY),
        ("own voice, fuzzier transcript", "Yes I hear you I am here and ready to help"),
    ):
        h = _Harness(transcript)
        h.judge()
        report(
            title,
            not h.cuts and h.session.echo_drops == 1,
            f"cuts={len(h.cuts)} echo_drops={h.session.echo_drops}",
        )

    # 3. when Auto-Interrupt (instant_barge) is OFF, a stop word is required
    h = _Harness("turn off the kitchen lights")
    h.session.settings.instant_barge = False
    h.judge()
    report("instant_barge=False, no stop word -> kept speaking", not h.cuts, f"cuts={len(h.cuts)}")

    # 3a. when Auto-Interrupt (instant_barge) is ON (default), multi-word non-echo speech cuts!
    h = _Harness("turn off the kitchen lights")
    h.session.settings.instant_barge = True
    h.judge()
    report("instant_barge=True, real multi-word speech -> cuts", len(h.cuts) == 1, f"cuts={len(h.cuts)}")

    # 3b. a person saying a stop word cuts the reply
    h = _Harness("stop")
    h.judge()
    report("person says 'stop' -> reply cut", len(h.cuts) == 1, f"cuts={len(h.cuts)}")

    # 3c. the word 'of' inside our own speech must never cut (the live defect)
    h = _Harness("of")
    h.judge()
    report("transcript 'of' -> kept speaking", not h.cuts, f"cuts={len(h.cuts)}")

    # 4. one continuous run is judged once per window, not once per audio block
    h = _Harness("stop")
    h.judge()
    h.judge()
    report(
        "throttle: no second judgement inside the window",
        len(h.cuts) == 1,
        f"cuts={len(h.cuts)}",
    )

    # 5. echo first, then the same person keeps talking: they win on the next window
    h = _Harness(REPLY)
    h.judge()
    first = len(h.cuts)
    h.session._asr = cast(Any, _FakeAsr("hey, stop for a second"))
    h.rewind_clock()
    h.judge()
    report(
        "echo ignored, then 'stop' wins next window",
        first == 0 and len(h.cuts) == 1,
        f"cuts={len(h.cuts)}",
    )

    # 6. the level gate still runs first: quiet audio is not an interruption
    h = _Harness("turn off the kitchen lights")
    h.judge(QUIET_RMS)
    report("quiet audio never interrupts", not h.cuts, f"cuts={len(h.cuts)}")

    # 7. the live ordering: the sentence being spoken must be known while it sounds.
    h = _Harness("turn off the kitchen lights")
    session = h.session
    session.settings.filler_enabled = False  # keep the acknowledgement out of this one
    session.settings.tts_enabled = True
    session.settings.front_uses_pi = False
    session.settings.provider = "openrouter"
    session.settings.model = "nvidia/nemotron-3.5-super-vl:free"
    voice = _RecordingVoice(session)
    session.agent = cast(Any, lambda: _FakeAgent(REPLY))
    session.tts = cast(Any, lambda: voice)
    session.ask("what can you do")
    # M2.2: the reply is spoken as it streams, so it arrives as its sentences (the
    # pump may join whatever queued up together) - but the whole reply must be
    # spoken, and at every say() the session field must name the text going out.
    report(
        "live turn: the reply is current while it sounds",
        bool(voice.said)
        and " ".join(voice.said) == REPLY
        and voice.seen_at_say == voice.said[-1],
        f"said={len(voice.said)} piece(s), seen={voice.seen_at_say[:34]!r}",
    )

    # 8-10. the finished-segment path, which used to cut on loudness alone
    for title, transcript, rms, expect_cut in (
        ("finished segment: own voice does not cut", REPLY, MEASURED_ECHO_RMS, False),
        (
            "finished segment: 'cancel' cuts",
            "please cancel that",
            MEASURED_ECHO_RMS,
            True,
        ),
        (
            "finished segment: no stop word keeps speaking (instant_barge=False)",
            "what time is it",
            MEASURED_ECHO_RMS,
            False,
        ),
        (
            "finished segment: quiet audio does not cut",
            "turn off the kitchen lights",
            QUIET_RMS,
            False,
        ),
    ):
        h = _Harness(transcript)
        h.session.settings.instant_barge = False
        segment = np.full(16000, rms, dtype=np.float32)
        cut = h.session._should_cut_for(segment, rms, 1000.0, 2000.0)
        report(title, cut == expect_cut, f"cut={cut}")

    # 13. stop-word-only mode cuts on 'stop' even with idle wake lock
    h = _Harness("stop")
    h.session.settings.wake_to_interrupt = True
    h.session.settings.wake_speaker_lock = True
    h.judge()
    report(
        "stop-word mode cuts on 'stop' even when idle",
        len(h.cuts) == 1,
        f"cuts={len(h.cuts)}",
    )

    # 11. no words in the captured audio cannot cut a reply: a check that cannot hear
    # content must not fail open (found by the verification audit).
    h = _Harness("")
    h.judge()
    report("empty transcript cannot cut a reply", not h.cuts, f"cuts={len(h.cuts)}")

    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
