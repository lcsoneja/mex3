"""Reference-based echo suppression (pure numpy, Smart-App-Control safe).

The app's own TTS is fed in as a reference ring buffer. Each microphone block
is matched against the last few seconds of playback at plausible acoustic
delays (0-200 ms). A strong normalized correlation means "this is our own
voice": the block is attenuated before VAD/ASR ever see it, so the assistant
never answers itself. Real speech over the playback (double-talk) correlates
poorly and passes through to the barge-in judge.

WebRTC AEC3 (pywebrtc-audio) was tried first; Smart App Control blocks its
native DLL on this laptop, hence the numpy correlator.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.signal import resample_poly

RATE = 16000
# The ring must outlive a whole spoken reply: playback is fed sentence-by-
# sentence, and the mic hears audio fed seconds earlier. 12 s covers long replies.
REF_SECONDS = 12.0
# 4 ms grid: must divide the 512-sample mic block and the context length,
# otherwise the true echo offset falls between grid points and scores collapse.
STRIDE = 64
MIN_RMS = 0.004  # silence is not echo material; skip unstable low-energy matches

_ref_lock = threading.Lock()
_ref_buf = np.zeros(int(RATE * REF_SECONDS), dtype=np.float32)
_ref_pos = 0
_ref_filled = 0
_ref_active_until = 0.0


def feed_reference(samples: np.ndarray, rate: int) -> None:
    """Register audio that is about to play, so the mic path can recognize it."""
    global _ref_pos, _ref_filled, _ref_active_until
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return
    if int(rate) != RATE:
        divisor = int(np.gcd(int(rate), RATE))
        data = resample_poly(data, RATE // divisor, int(rate) // divisor).astype(
            np.float32
        )
    with _ref_lock:
        n = len(_ref_buf)
        first = min(data.size, n - _ref_pos)
        _ref_buf[_ref_pos : _ref_pos + first] = data[:first]
        rest = data.size - first
        if rest > 0:
            _ref_buf[:rest] = data[first : first + rest]
        _ref_pos = (_ref_pos + data.size) % n
        _ref_filled = min(n, _ref_filled + data.size)
        # Playback starts now and runs for the clip's duration (+ small tail).
        _ref_active_until = time.perf_counter() + data.size / RATE + 0.3


def _reference_window(seconds: float) -> np.ndarray | None:
    with _ref_lock:
        if _ref_filled == 0 or time.perf_counter() > _ref_active_until:
            return None
        n = len(_ref_buf)
        want = min(_ref_filled, int(seconds * RATE))
        start = (_ref_pos - want) % n
        if start + want <= n:
            return _ref_buf[start : start + want].copy()
        return np.concatenate([_ref_buf[start:], _ref_buf[: want - (n - start)]])


class EchoSuppressor:
    """Detect-and-duck echo: correlate mic context against recent playback.

    Decisions use a rolling ~0.4 s context, not single 32 ms blocks: short
    same-voice snippets correlate by chance, longer windows separate "our
    sentence" from "the owner's sentence" reliably. Judgments are throttled
    (every 4th block) and the verdict applies to the blocks between them.
    """

    name = "numpy-correlator"

    def __init__(
        self,
        threshold: float = 0.35,
        attenuation: float = 0.08,
        context_s: float = 0.4,
        judge_every: int = 4,
    ) -> None:
        self.threshold = float(threshold)
        self.attenuation = float(attenuation)
        self._ctx = int(context_s * RATE)
        self._judge_every = max(1, int(judge_every))
        self._mic_hist = np.zeros(0, dtype=np.float32)
        self._since = 0
        self._last_echo = False
        self.last_score = 0.0

    @staticmethod
    def _voiced(block: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(block * block))) >= MIN_RMS

    def _match(self, data: np.ndarray) -> tuple[float, np.ndarray | None]:
        ref = _reference_window(REF_SECONDS)
        if ref is None or ref.size < data.size:
            return 0.0, None
        win = np.lib.stride_tricks.sliding_window_view(ref, data.size)[::STRIDE]
        if win.shape[0] == 0:
            return 0.0, None
        dots = win @ data
        norms = np.sqrt(np.sum(win * win, axis=1)) * (np.linalg.norm(data) + 1e-9)
        cos = dots / (norms + 1e-9)
        idx = int(np.argmax(cos))
        return float(cos[idx]), win[idx]

    def is_echo(self, block: np.ndarray) -> bool:
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        if not self._voiced(data):
            self.last_score = 0.0
            return False
        if data.size < self._ctx:
            data = np.concatenate([self._mic_hist, data])[-self._ctx :]
        score, _ = self._match(data)
        self.last_score = score
        return score >= self.threshold

    def process(self, block: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (clean_block, was_echo). Echo is subtracted then ducked."""
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        self._mic_hist = np.concatenate([self._mic_hist, data])[-self._ctx :]
        if not self._voiced(data):
            return data, False
        self._since += 1
        if self._since < self._judge_every:
            if self._last_echo:
                return (data * self.attenuation).astype(np.float32), True
            return data, False
        self._since = 0
        score, seg = self._match(self._mic_hist)
        self.last_score = score
        self._last_echo = score >= self.threshold and seg is not None
        if not self._last_echo or seg is None:
            return data, False
        tail = seg[-data.size :]
        alpha = float(np.dot(tail, data) / (np.dot(tail, tail) + 1e-9))
        residual = data - alpha * tail
        return (residual * self.attenuation).astype(np.float32), True
