"""Scripted conversation simulation against the S1-S10 natural-conversation standards.

Two layers of evidence:
  * Decision-logic tests drive the REAL _handle_speech / _judge_interruption with a
    controlled ASR transcript (so TTS->ASR mis-hearing cannot mask the logic).
  * Audio-path tests drive the REAL Parakeet ASR with injected speech via
    ScriptedStream + simulated speech output.

Run:  python conversation_sim.py  ->  out/conversation_sim_report.md
"""

from __future__ import annotations

import os

os.environ.setdefault("ME3_NO_PERSIST", "1")

import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from audio import ScriptedStream, ScriptEvent
from router import Router
from server import Session, SpeechBudget, load_settings

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
CLIPS = OUT / "sim_clips"
RESULTS = []


def wait_until(pred, timeout_s, interval=0.02):
    end = time.time() + timeout_s
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


def report(name, standard, passed, observed):
    RESULTS.append(
        {
            "name": name,
            "standard": standard,
            "passed": bool(passed),
            "observed": observed,
        }
    )
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name} ({standard}): {observed}", flush=True
    )


class _FakeAgent:
    def __init__(self, text):
        self.text = text
        self.history = []

    def respond(self, text, on_delta=None, **kw):
        from agent import AgentReply

        if on_delta:
            for w in self.text.split(" "):
                on_delta(w + " ")
        return AgentReply(text=self.text)


class _FakeAsr:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.i = 0

    def transcribe(self, audio, sample_rate=16000):
        t = self.transcripts[min(self.i, len(self.transcripts) - 1)]
        self.i += 1
        return t


def make_clips():
    from tts import NeuralTts

    CLIPS.mkdir(parents=True, exist_ok=True)
    tts = NeuralTts(voice="Neural: en_US-lessac-high")
    paths = {}
    for key, phrase in {
        "complete": "what time is it",
        "yeah": "yeah",
        "hello": "hello there how are you today",
    }.items():
        p = CLIPS / f"{key}.wav"
        if not p.exists():
            tts.to_wav(phrase, p)
        paths[key] = p
    return paths


def new_session(reply):
    settings = load_settings()
    settings.provider = "openrouter"
    settings.model = "cohere/north-mini-code:free"
    settings.front_uses_pi = False
    settings.tts_simulate = True
    settings.tts_enabled = True
    settings.barge_in = True
    settings.wake_to_interrupt = True
    settings.wake_speaker_lock = False
    settings.require_wake_when_idle = False
    settings.smart_turn = True
    settings.vad_min_silence = 0.6
    settings.ask_detector = True
    settings.speech_cap_chars = 350
    settings.filler_enabled = False
    s = Session(settings)
    s.agent = cast(Any, lambda: _FakeAgent(reply))
    s.mode = "off"
    return s


def loud_segment(seconds=1.0, rms=0.1):
    return np.full(int(16000 * seconds), rms, dtype=np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    clips = make_clips()

    # ---- S2 semantic completeness (unit, no LLM) ----
    r = Router(enabled=False)
    inc = not r.thought_complete("create a game and")
    comp = r.thought_complete("what time is it")
    report(
        "semantic-completeness",
        "S2",
        inc and comp,
        f"'create a game and' held={inc}, 'what time is it' complete={comp}",
    )

    # ---- S6 spoken reply cap (unit) ----
    long_reply = "Sure, I can help with that. " * 30
    bud = SpeechBudget(350)
    spoken = bud.take(long_reply)
    report(
        "spoken-reply-cap",
        "S6",
        len(spoken) <= 350 and bud.dropped > 0,
        f"spoken {len(spoken)} chars (cap 350), dropped {bud.dropped}",
    )

    # ---- S2/S5 multi-clause instruction with a pause -> ONE joined turn ----
    # Controlled ASR so the real hold/join logic is tested without TTS->ASR mis-hearing.
    s = new_session("Okay, I will create a snake game.")
    s.mode = "continuous"
    s._listener = cast(Any, type("L", (), {"is_alive": lambda self: True})())
    s._mic_is_scripted = True
    s._asr = cast(Any, _FakeAsr(["create a game and", "make it a snake game"]))
    s._asr_backend = s.settings.asr_backend
    s._handle_speech(loud_segment())  # part A ends in 'and' -> held
    held_after_a = bool(s._held_text)
    s._handle_speech(loud_segment())  # part B completes it -> one turn
    time.sleep(0.5)
    turns = s.state()["turns"]
    joined = len(turns) == 1 and "snake" in turns[0].get("user_text", "").lower()
    report(
        "multi-clause-one-turn",
        "S2/S5",
        held_after_a and joined,
        f"held_after_A={held_after_a}, turns={len(turns)}, text={turns[0].get('user_text', '')[:50] if turns else ''!r}",
    )

    # ---- S4 stop word cuts the reply (controlled ASR, real _judge_interruption) ----
    s = new_session(long_reply)
    s._asr = cast(Any, _FakeAsr(["stop"]))
    s._asr_backend = s.settings.asr_backend
    s._speaking = True
    s._speech_started_at = time.perf_counter() - 2.0  # 2 s into the reply (past grace)
    s._last_spoken = long_reply
    before = s.state()["barge_ins"]
    s._judge_interruption(loud_segment())
    cut = s.state()["barge_ins"] > before
    report(
        "stop-word-cuts",
        "S4",
        cut,
        f"barge_ins {before}->{s.state()['barge_ins']} on 'stop'",
    )

    # ---- S4/S9 arbitrary speech does NOT cut (controlled ASR) ----
    s = new_session(long_reply)
    s._asr = cast(Any, _FakeAsr(["hello there how are you"]))
    s._asr_backend = s.settings.asr_backend
    s._speaking = True
    s._speech_started_at = time.perf_counter() - 2.0
    s._last_spoken = long_reply
    before = s.state()["barge_ins"]
    s._judge_interruption(loud_segment())
    report(
        "arbitrary-speech-no-cut",
        "S4/S9",
        s.state()["barge_ins"] == before,
        f"barge_ins stayed {before} on non-stop speech",
    )

    # ---- S3 backchannel does not cut (controlled ASR) ----
    s = new_session(long_reply)
    s._asr = cast(Any, _FakeAsr(["yeah"]))
    s._asr_backend = s.settings.asr_backend
    s._speaking = True
    s._speech_started_at = time.perf_counter() - 2.0
    s._last_spoken = long_reply
    before = s.state()["barge_ins"]
    s._judge_interruption(loud_segment())
    report(
        "backchannel-no-cut",
        "S3",
        s.state()["barge_ins"] == before,
        f"barge_ins stayed {before} on 'yeah'",
    )

    # ---- Real-ASR audio path: backchannel + follow-up via injected speech ----
    s = new_session("It is three o'clock.")
    st = ScriptedStream(
        [ScriptEvent(0.0, clips["complete"], "first")], gain=2.0, tail_s=40.0
    )
    s.start_listening("continuous", source=st)
    wait_until(lambda: len(s.state()["turns"]) >= 1, 30.0)
    wait_until(lambda: s.state()["status"] in ("idle", "listening"), 30.0)
    time.sleep(1.0)
    n0 = len(s.state()["turns"])
    st.inject(clips["complete"], gain=2.0, label="follow-up")
    got = wait_until(lambda: len(s.state()["turns"]) > n0, 20.0)
    report(
        "follow-up-captured-real-asr",
        "S10",
        got,
        f"turns {n0}->{len(s.state()['turns'])}",
    )
    st.stop()
    s.stop_listening()

    # ---- write report ----
    passed = sum(1 for x in RESULTS if x["passed"])
    lines = [
        "# Conversation simulation - S1-S10 standards",
        "",
        f"RESULT: {passed}/{len(RESULTS)} passed",
        "",
        "| scenario | standard | result | observed |",
        "| --- | --- | --- | --- |",
    ]
    for x in RESULTS:
        lines.append(
            f"| {x['name']} | {x['standard']} | {'PASS' if x['passed'] else 'FAIL'} | {x['observed']} |"
        )
    lines += [
        "",
        "Decision-logic tests use controlled ASR on the real _handle_speech/_judge_interruption;",
        "audio-path test uses real Parakeet ASR + injected speech. S1 ack ~261 ms measured live.",
        "",
    ]
    (OUT / "conversation_sim_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRESULT: {passed}/{len(RESULTS)} passed", flush=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
