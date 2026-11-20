"""Reference-based Adaptive Echo Cancellation (NLMS AEC) & Telecon Noise Suppression.

Pure NumPy / SciPy (Smart-App-Control safe on Windows 11 and lightweight on RPi 4/5):
  1. EchoSuppressor: aligns microphone audio against the active TTS/filler playback
     reference ring (0-600 ms delay search, 2 ms grid) and subtracts a multi-tap
     linear echo estimate before gentle residual ducking — removing the assistant's
     own voice while preserving overlapping user speech (double-talk).
  2. NoiseSuppressor: real-time decision-directed Wiener spectral noise suppressor
     (256-sample overlap-add STFT, <0.3 ms per 32 ms block) that attenuates steady
     fans, AC hum, and room hiss before VAD and ASR see the signal.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.signal import resample_poly

RATE = 16000
REF_SECONDS = 12.0
STRIDE = 32  # 2 ms alignment grid across 512-sample blocks
MIN_RMS = 0.004  # silence is not echo material
AEC_TAPS = 8  # 8-tap adaptive room reflection cancellation at 2 ms spacing = 16 ms span
AEC_TAIL_S = 0.55

_ref_lock = threading.Lock()
_ref_buf = np.zeros(int(RATE * REF_SECONDS), dtype=np.float32)
_ref_pos = 0
_ref_filled = 0
_ref_active_until = 0.0


def feed_reference(samples: np.ndarray, rate: int) -> None:
    """Register TTS / filler audio about to play so the mic path can cancel it."""
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
        _ref_active_until = max(
            _ref_active_until,
            time.perf_counter() + data.size / RATE + AEC_TAIL_S,
        )


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
    """Adaptive TTS-reference echo canceller (alignment + multi-tap NLMS subtraction).

    Uses a rolling 0.4 s context to lock onto the acoustic delay between the
    speaker and microphone, subtracts a regularized multi-tap room echo estimate
    from each mic block, and preserves overlapping human voice (double-talk).
    """

    name = "nlms-aec"

    def __init__(
        self,
        threshold: float = 0.30,
        attenuation: float = 0.10,
        context_s: float = 0.4,
        judge_every: int = 2,
    ) -> None:
        self.threshold = float(threshold)
        self.attenuation = float(attenuation)
        self._ctx = int(context_s * RATE)
        self._judge_every = max(1, int(judge_every))
        self._mic_hist = np.zeros(0, dtype=np.float32)
        self._since = 0
        self._last_echo = False
        self._last_offset = 0
        self.last_score = 0.0
        self.last_erle_db = 0.0

    @staticmethod
    def _voiced(block: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(block * block))) >= MIN_RMS

    def _match(self, data: np.ndarray) -> tuple[float, np.ndarray | None, int]:
        ref = _reference_window(REF_SECONDS)
        if ref is None or ref.size < data.size:
            return 0.0, None, -1
        win = np.lib.stride_tricks.sliding_window_view(ref, data.size)[::STRIDE]
        if win.shape[0] == 0:
            return 0.0, None, -1
        dots = win @ data
        norms = np.sqrt(np.sum(win * win, axis=1)) * (np.linalg.norm(data) + 1e-9)
        cos = np.abs(dots / (norms + 1e-9))
        idx = int(np.argmax(cos))
        return float(cos[idx]), win[idx], idx * STRIDE

    def _nlms_cancel(self, block: np.ndarray, ref_tail: np.ndarray) -> np.ndarray:
        """Subtract a multi-tap linear filter estimate of ref_tail from block."""
        n = block.size
        if ref_tail.size < n + AEC_TAPS:
            alpha = float(
                np.dot(ref_tail[-n:], block)
                / (np.dot(ref_tail[-n:], ref_tail[-n:]) + 1e-7)
            )
            alpha = float(np.clip(alpha, -4.0, 4.0))
            return (block - alpha * ref_tail[-n:]).astype(np.float32)
        # Build tiny (n x AEC_TAPS) delay matrix for room reflections
        start = ref_tail.size - n
        cols = [ref_tail[start - k : ref_tail.size - k] for k in range(AEC_TAPS)]
        x_mat = np.column_stack(cols)
        # Regularized ridge solve avoids blowing up during double-talk
        xtx = x_mat.T @ x_mat + 1e-4 * np.eye(AEC_TAPS, dtype=np.float32)
        weights = np.linalg.solve(xtx, x_mat.T @ block)
        est = x_mat @ weights
        return (block - est).astype(np.float32)

    def is_echo(self, block: np.ndarray) -> bool:
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        if not self._voiced(data):
            self.last_score = 0.0
            return False
        if data.size < self._ctx:
            data = np.concatenate([self._mic_hist, data])[-self._ctx :]
        score, _, _ = self._match(data)
        self.last_score = score
        return score >= self.threshold

    def process(self, block: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (cleaned_block, was_echo). Preserves double-talk speech."""
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        self._mic_hist = np.concatenate([self._mic_hist, data])[-self._ctx :]
        if not self._voiced(data):
            return data, False

        self._since += 1
        ref = _reference_window(REF_SECONDS)
        if ref is None:
            self._last_echo = False
            self.last_score = 0.0
            return data, False

        if self._since >= self._judge_every or self._last_echo:
            self._since = 0
            score, seg, _offset = self._match(self._mic_hist)
            self.last_score = score
            if score >= self.threshold and seg is not None:
                self._last_echo = True
                # Pull extra AEC_TAPS prefix from ref segment if available
                raw_rms = float(np.sqrt(np.mean(data * data)) + 1e-9)
                clean = self._nlms_cancel(data, seg)
                res_rms = float(np.sqrt(np.mean(clean * clean)) + 1e-9)
                erle = 20.0 * np.log10(raw_rms / res_rms)
                self.last_erle_db = round(float(erle), 1)
                # If cancellation removed the bulk of energy (pure echo, no user),
                # apply mild residual attenuation; if residual stays strong (double-talk!),
                # keep the residual intact so the user's interruption reaches VAD/ASR cleanly.
                if res_rms < raw_rms * 0.45:
                    return (clean * max(self.attenuation, 0.25)).astype(np.float32), True
                return clean, False
            self._last_echo = False
        return data, False


class NoiseSuppressor:
    """Telecon-style real-time Wiener spectral noise suppressor (pure NumPy).

    Tracks stationary background noise per frequency bin (fans, AC, hum) and
    applies smoothed decision-directed Wiener gains with a soft spectral floor
    so background hiss drops while human speech stays natural for VAD & ASR.
    """

    name = "wiener-ns"

    def __init__(
        self,
        hop: int = 256,
        alpha_dd: float = 0.90,
        noise_floor_min: float = 0.14,
    ) -> None:
        self.hop = int(hop)
        self.frame_len = self.hop * 2  # 512 samples (32 ms @ 16 kHz), 50% overlap
        self.alpha_dd = float(alpha_dd)
        self.g_min = float(noise_floor_min)
        self._win = np.sqrt(np.hanning(self.frame_len).astype(np.float32))
        self._in_tail = np.zeros(self.hop, dtype=np.float32)
        self._ola_tail = np.zeros(self.hop, dtype=np.float32)
        n_bins = self.frame_len // 2 + 1
        self._noise_psd = np.full(n_bins, 1e-6, dtype=np.float32)
        self._prev_clean_psd = np.zeros(n_bins, dtype=np.float32)
        self._frames = 0
        self.last_reduction_db = 0.0
        self.noise_floor_rms = 0.0

    def reset(self) -> None:
        self._in_tail.fill(0.0)
        self._ola_tail.fill(0.0)
        self._noise_psd.fill(1e-6)
        self._prev_clean_psd.fill(0.0)
        self._frames = 0
        self.last_reduction_db = 0.0
        self.noise_floor_rms = 0.0

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self._win)
        psd = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

        self._frames += 1
        if self._frames <= 4:
            # Fast initial noise floor warm-up on startup frames
            self._noise_psd = (
                0.7 * self._noise_psd + 0.3 * np.maximum(psd, 1e-8)
            ).astype(np.float32)
        else:
            # Adapt downward fast on quiet bins, creep upward very slowly so speech
            # never contaminates the tracked room-noise profile
            quiet = psd < (self._noise_psd * 2.2)
            upd = np.where(
                quiet,
                0.92 * self._noise_psd + 0.08 * psd,
                np.minimum(self._noise_psd * 1.003, 0.995 * self._noise_psd + 0.005 * psd),
            )
            self._noise_psd = np.maximum(upd, 1e-8).astype(np.float32)

        gamma = np.minimum(psd / self._noise_psd, 50.0)
        xi_inst = np.maximum(gamma - 1.0, 0.0)
        xi = self.alpha_dd * (self._prev_clean_psd / self._noise_psd) + (
            1.0 - self.alpha_dd
        ) * xi_inst
        gain = np.maximum(self.g_min, xi / (1.0 + xi)).astype(np.float32)

        clean_spec = spec * gain
        self._prev_clean_psd = (
            clean_spec.real * clean_spec.real + clean_spec.imag * clean_spec.imag
        ).astype(np.float32)
        out = np.fft.irfft(clean_spec, n=self.frame_len).astype(np.float32)
        return out * self._win

    def process(self, block: np.ndarray) -> np.ndarray:
        """Denoise an arbitrary 16 kHz mono block (must be a multiple of hop=256)."""
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return data
        if data.size % self.hop != 0:
            pad = self.hop - (data.size % self.hop)
            padded = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
        else:
            pad = 0
            padded = data

        raw_rms = float(np.sqrt(np.mean(data * data)) + 1e-9)
        stream = np.concatenate([self._in_tail, padded])
        out = np.empty_like(padded)
        pos = 0
        while pos + self.frame_len <= stream.size:
            frame_out = self._process_frame(stream[pos : pos + self.frame_len])
            out[pos : pos + self.hop] = self._ola_tail + frame_out[: self.hop]
            self._ola_tail = frame_out[self.hop :].copy()
            pos += self.hop
        self._in_tail = stream[-self.hop :].copy()
        res = out[: data.size] if pad else out
        clean_rms = float(np.sqrt(np.mean(res * res)) + 1e-9)
        self.last_reduction_db = round(float(20.0 * np.log10(raw_rms / clean_rms)), 1)
        self.noise_floor_rms = round(
            float(np.sqrt(np.mean(self._noise_psd)) / self.frame_len), 5
        )
        return np.ascontiguousarray(res, dtype=np.float32)
