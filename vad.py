"""Silero voice activity detection, wrapping the sherpa-onnx implementation.

Feed 512-sample blocks; complete utterances come back as speech segments.
Accessors are probed at call time because sherpa-onnx has shipped both
properties and methods for the same names across releases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SIZE = 512


class VadError(RuntimeError):
    """Raised when the VAD model cannot be created or used."""


class SileroVad:
    """Streaming VAD with one method per pipeline stage."""

    def __init__(
        self,
        model_path: Path | str,
        threshold: float = 0.5,
        min_silence_duration: float = 0.35,
        min_speech_duration: float = 0.2,
        max_speech_duration: float = 20.0,
        buffer_seconds: int = 30,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise VadError(
                "sherpa-onnx is not installed (pip install sherpa-onnx)"
            ) from exc

        resolved = Path(model_path)
        if not resolved.exists():
            raise VadError(f"VAD model not found: {resolved}")

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(resolved)
        config.silero_vad.threshold = float(threshold)
        config.silero_vad.min_silence_duration = float(min_silence_duration)
        config.silero_vad.min_speech_duration = float(min_speech_duration)
        config.silero_vad.max_speech_duration = float(max_speech_duration)
        config.silero_vad.window_size = int(window_size)
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = "cpu"

        self.window_size = int(window_size)
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=buffer_seconds
        )

    def accept(self, block: np.ndarray) -> None:
        """Feed one block of float32 audio."""
        if block.size:
            self._vad.accept_waveform(np.asarray(block, dtype=np.float32).reshape(-1))

    def segments(self) -> list[np.ndarray]:
        """Pop every complete speech segment produced so far."""
        out: list[np.ndarray] = []
        while not self._is_empty():
            out.append(np.array(self._vad.front.samples, dtype=np.float32))
            self._vad.pop()
        return out

    def is_speech(self) -> bool:
        """True once speech has been detected in the current buffer."""
        detected = self._vad.is_speech_detected
        return bool(detected()) if callable(detected) else bool(detected)

    def partial(self) -> np.ndarray | None:
        """Audio of the utterance being spoken right now, if any.

        Complete segments only appear after the speaker pauses, which is far too
        late to react to an interruption - this exposes the in-progress audio so
        barge-in can be judged within a few hundred milliseconds.
        """
        segment = getattr(self._vad, "current_segment", None)
        if segment is None:
            return None
        samples = getattr(segment, "samples", None)
        if samples is None or len(samples) == 0:
            return None
        return np.array(samples, dtype=np.float32)

    def flush(self) -> list[np.ndarray]:
        """Close the current utterance (used when a push-to-talk ends)."""
        self._vad.flush()
        return self.segments()

    def reset(self) -> None:
        """Drop buffered audio."""
        self._vad.reset()

    def _is_empty(self) -> bool:
        """The empty accessor is a property in some builds and a method in others."""
        empty = self._vad.empty
        return bool(empty()) if callable(empty) else bool(empty)
