"""Microphone capture and aborted-on-demand playback for the ME3 laptop app.

Playback must be interruptible: barge-in is the feature that makes the agent
feel conversational, so the player owns an abortable output stream instead of
using the fire-and-forget helpers in sounddevice.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Poka-yoke: serialize PortAudio stream open/close/stop across mic & TTS threads
# to prevent 0xc0000005 in libportaudio64bit.dll_unloaded during races/teardown.
PA_LOCK = threading.RLock()

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

TARGET_RATE = 16000
BLOCK_SIZE = 512


class AudioError(RuntimeError):
    """Raised when an audio device cannot be opened or used."""


@dataclass(frozen=True)
class DeviceInfo:
    """One input or output device as reported by PortAudio."""

    index: int
    name: str
    channels: int
    default_rate: float


def _devices(attr: str) -> list[DeviceInfo]:
    found: list[DeviceInfo] = []
    for index, raw in enumerate(sd.query_devices()):
        try:
            channels = int(raw[attr])
            rate = float(raw["default_samplerate"])
        except (TypeError, ValueError) as exc:
            print(f"[audio] skipping device {index}: unusable descriptor ({exc})")
            continue
        if channels <= 0:
            continue
        found.append(
            DeviceInfo(
                index=index, name=str(raw["name"]), channels=channels, default_rate=rate
            )
        )
    return found


def can_open_input(index: int) -> bool:
    """Public probe: can this capture device actually be opened right now?

    R7 asks this before a turn does. A microphone held by another app answers
    "no" here, which is the difference between a diagnosis and a failed demo.
    """
    return _can_open_input(index)


def probe_input_blocks(index: int, timeout_s: float = 0.35) -> bool:
    """True when the input device both opens AND delivers at least one audio callback."""
    got = threading.Event()

    def _cb(indata, frames, time_info, status) -> None:
        del indata, frames, time_info, status
        got.set()

    try:
        with PA_LOCK, sd.InputStream(
            device=index,
            channels=1,
            samplerate=TARGET_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=_cb,
        ):
            return got.wait(timeout=timeout_s)
    except (sd.PortAudioError, ValueError, OSError):
        return False


def can_open_output(index: int) -> bool:
    """Public probe for the playback side, used by the same repair."""
    return _can_open_output(index)


def input_devices() -> list[DeviceInfo]:
    """Every device with at least one capture channel."""
    return _devices("max_input_channels")


def output_devices() -> list[DeviceInfo]:
    """Every device with at least one playback channel."""
    return _devices("max_output_channels")


def default_devices() -> tuple[int | None, int | None]:
    """PortAudio default (input, output) indices.

    PortAudio happily reports a driver-level default that cannot actually be
    opened (a common Windows situation), so this probes each candidate and
    falls back to the first device that really opens.
    """
    pair = sd.default.device
    in_index = pair[0] if pair[0] != -1 else None
    out_index = pair[1] if pair[1] != -1 else None

    if in_index is None or not _can_open_input(in_index):
        in_index = next(
            (d.index for d in input_devices() if _can_open_input(d.index)), None
        )
    if out_index is None or not _can_open_output(out_index):
        out_index = next(
            (d.index for d in output_devices() if _can_open_output(d.index)), None
        )
    return in_index, out_index


def _can_open_input(index: int) -> bool:
    try:
        with sd.InputStream(
            device=index, channels=1, samplerate=TARGET_RATE, blocksize=BLOCK_SIZE
        ):
            return True
    except (sd.PortAudioError, ValueError):
        return False


def _can_open_output(index: int) -> bool:
    try:
        with sd.OutputStream(
            device=index, channels=1, samplerate=TARGET_RATE, blocksize=BLOCK_SIZE
        ):
            return True
    except (sd.PortAudioError, ValueError):
        return False


def load_wav_16k(path: Path | str) -> np.ndarray:
    """Read any WAV as mono float32 at 16 kHz."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != TARGET_RATE:
        divisor = int(np.gcd(int(rate), TARGET_RATE))
        mono = resample_poly(mono, TARGET_RATE // divisor, int(rate) // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32)


@dataclass
class ScriptEvent:
    """One injected sound: a WAV file that starts at a given time."""

    offset_s: float
    path: Path
    label: str = ""


class ScriptedStream:
    """Replays WAV files on a timeline, in real time, like a microphone.

    This is what lets the full-duplex behaviour be tested and benchmarked on a
    machine with no microphone: a scenario is a list of (offset, file) events,
    mixed into one stream that is fed to the pipeline at real-time pace. An
    interruption is simply a second event that starts while the agent is talking.
    """

    def __init__(
        self,
        events: list[ScriptEvent],
        sample_rate: int = TARGET_RATE,
        block_size: int = BLOCK_SIZE,
        gain: float = 1.0,
        tail_s: float = 4.0,
        loop: bool = False,
    ) -> None:
        if not events:
            raise AudioError("ScriptedStream needs at least one event")
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.gain = float(gain)
        self.loop = loop
        self.events = list(events)

        ends: list[float] = []
        clips: list[tuple[float, np.ndarray]] = []
        for event in self.events:
            samples = load_wav_16k(event.path)
            clips.append((float(event.offset_s), samples))
            ends.append(float(event.offset_s) + len(samples) / sample_rate)
        self.duration_s = max(ends) + float(tail_s)

        self._timeline = np.zeros(int(self.duration_s * sample_rate), dtype=np.float32)
        for offset, samples in clips:
            start = int(offset * sample_rate)
            stop = min(len(self._timeline), start + len(samples))
            if start < len(self._timeline):
                self._timeline[start:stop] += samples[: stop - start]
        if self.gain != 1.0:
            self._timeline = np.clip(self._timeline * self.gain, -1.0, 1.0)

        self._position = 0
        self._started_at = 0.0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._position = 0
        self._started_at = time.perf_counter()
        self._active = True

    def stop(self) -> None:
        self._active = False

    def inject(self, path: Path | str, gain: float = 1.0, label: str = "") -> int:
        """Add a clip that starts now, as if the user spoke at this moment.

        Returns the sample position it was inserted at, so a caller can time its
        reaction - this is how an interruption is injected mid-reply.
        """
        samples = np.clip(load_wav_16k(path) * float(gain), -1.0, 1.0)
        start = max(0, self._position)
        needed = start + len(samples)
        if needed > len(self._timeline):
            grow = needed - len(self._timeline)
            self._timeline = np.concatenate(
                [self._timeline, np.zeros(grow, dtype=np.float32)]
            )
        self._timeline[start:needed] += samples
        self._timeline = np.clip(self._timeline, -1.0, 1.0)
        self.duration_s = len(self._timeline) / self.sample_rate
        self.events.append(ScriptEvent(self._position / self.sample_rate, Path(path), label))
        return start

    def read(self, timeout: float = 0.3) -> np.ndarray | None:
        """Next block, delivered at the pace the audio would have arrived."""
        if not self._active:
            return None
        due = self._started_at + self._position / self.sample_rate
        wait = due - time.perf_counter()
        if wait > 0:
            time.sleep(min(wait, timeout))
        block = self._timeline[self._position : self._position + self.block_size]
        if block.size == 0:
            if self.loop:
                self._position = 0
                self._started_at = time.perf_counter()
                return self.read(timeout)
            self._active = False
            time.sleep(0.05)
            return None
        self._position += self.block_size
        return np.array(block, dtype=np.float32)


class MicStream:
    """Continuous mono capture at 16 kHz, delivered as fixed-size blocks."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = TARGET_RATE,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._last_block_at = 0.0

    @property
    def active(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def start(self) -> None:
        with PA_LOCK:
            if self.active:
                return
            try:
                self._stream = sd.InputStream(
                    device=self.device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    dtype="float32",
                    callback=self._on_block,
                )
                self._stream.start()
                self._last_block_at = time.monotonic()
            except (sd.PortAudioError, ValueError, OSError) as exc:
                self._stream = None
                raise AudioError(f"cannot open microphone {self.device}: {exc}") from exc

    def stop(self) -> None:
        with PA_LOCK:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except (sd.PortAudioError, OSError, RuntimeError):
                    pass

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Next block, or None when nothing arrived within the timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _on_block(self, indata, frames, time_info, status) -> None:
        del frames, time_info  # unused PortAudio callback arguments
        if status:
            print(f"[mic] {status}")
        self._last_block_at = time.monotonic()
        self._queue.put(np.array(indata[:, 0], dtype=np.float32, copy=True))


class Player:
    """Mono playback that can be aborted mid-utterance."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._data: np.ndarray | None = None
        self._position = 0

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._stream is not None

    def play(
        self, samples: np.ndarray, sample_rate: int = TARGET_RATE, blocking: bool = True
    ) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        self.stop()
        with self._lock:
            self._data = data
            self._position = 0
            # W7: serialize PortAudio open/start with the mic stream (PA_LOCK) to
            # stop the libportaudio64bit.dll_unloaded race that crashed the engine.
            with PA_LOCK:
                self._stream = sd.OutputStream(
                    device=self.device,
                    channels=1,
                    samplerate=sample_rate,
                    dtype="float32",
                    callback=self._on_output,
                )
                self._stream.start()
        if blocking:
            while self.playing:
                time.sleep(0.02)

    def stop(self) -> None:
        """Abort playback right now; safe to call when idle."""
        with self._lock:
            stream, self._stream = self._stream, None
            self._data = None
            self._position = 0
        if stream is None:
            return
        with PA_LOCK:
            try:
                stream.abort()
            except sd.PortAudioError as exc:
                print(f"[player] abort failed: {exc}")
            finally:
                stream.close()

    def _on_output(self, outdata, frames, time_info, status) -> None:
        del time_info
        if status:
            print(f"[player] {status}")
        with self._lock:
            data = self._data
            if data is None:
                raise sd.CallbackStop
            remaining = len(data) - self._position
            take = min(frames, remaining)
            if take > 0:
                outdata[:take, 0] = data[self._position : self._position + take, 0]
            if take < frames:
                outdata[take:, 0] = 0.0
            self._position += take
            if self._position >= len(data):
                raise sd.CallbackStop
