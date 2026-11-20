"""Offline speech-to-text backends for the ME3 laptop voice app.

Every backend accepts mono float32 audio at 16 kHz and returns a transcript.
The interface is deliberately narrow: on the Raspberry Pi the same contract is
exposed through an MCP server instead of an in-process class.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class AsrError(RuntimeError):
    """Raised when a speech-to-text backend cannot be created or used."""


def _import_sherpa():
    """Import sherpa_onnx lazily so the UI still starts without it."""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise AsrError(
            "sherpa-onnx is not installed (pip install sherpa-onnx)"
        ) from exc
    return sherpa_onnx


def _pick(model_dir: Path, candidates: list[str]) -> Path:
    """Return the first existing candidate file inside model_dir."""
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    raise AsrError(f"none of {candidates} found in {model_dir}")


def _token_file(model_dir: Path, stem: str) -> Path:
    """Whisper exports put the tokens next to the encoder, under either name."""
    candidates = [f"{stem}-tokens.txt"]
    candidates += sorted(path.name for path in model_dir.glob("*-tokens.txt"))
    return _pick(model_dir, candidates)


class WhisperAsr:
    """Whisper tiny.en / small.en exported for sherpa-onnx."""

    backend = "whisper"

    def __init__(self, model_dir: Path | str, num_threads: int = 2) -> None:
        self.model_dir = Path(model_dir)
        self.name = self.model_dir.name
        if not self.model_dir.is_dir():
            raise AsrError(f"model folder not found: {self.model_dir}")

        tokens = _token_file(self.model_dir, self.name)
        stem = tokens.name.removesuffix("-tokens.txt")
        encoder = _pick(
            self.model_dir, [f"{stem}-encoder.int8.onnx", f"{stem}-encoder.onnx"]
        )
        decoder = _pick(
            self.model_dir, [f"{stem}-decoder.int8.onnx", f"{stem}-decoder.onnx"]
        )

        sherpa = _import_sherpa()
        self._lock = threading.Lock()
        self._recognizer = sherpa.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=num_threads,
            language="en",
            task="transcribe",
            decoding_method="greedy_search",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            self._recognizer.decode_stream(stream)
            return str(stream.result.text).strip()


class MoonshineAsr:
    """Moonshine base/tiny exported for sherpa-onnx (fast, English only)."""

    backend = "moonshine"

    def __init__(self, model_dir: Path | str, num_threads: int = 2) -> None:
        self.model_dir = Path(model_dir)
        self.name = self.model_dir.name
        if not self.model_dir.is_dir():
            raise AsrError(f"model folder not found: {self.model_dir}")

        preprocessor = _pick(self.model_dir, ["preprocess.onnx"])
        encoder = _pick(self.model_dir, ["encode.int8.onnx", "encode.onnx"])
        uncached = _pick(
            self.model_dir, ["uncached_decode.int8.onnx", "uncached_decode.onnx"]
        )
        cached = _pick(
            self.model_dir, ["cached_decode.int8.onnx", "cached_decode.onnx"]
        )
        tokens = _pick(self.model_dir, ["tokens.txt"])

        sherpa = _import_sherpa()
        if not hasattr(sherpa.OfflineRecognizer, "from_moonshine"):
            raise AsrError("this sherpa-onnx build has no Moonshine support")

        self._lock = threading.Lock()
        self._recognizer = sherpa.OfflineRecognizer.from_moonshine(
            preprocessor=str(preprocessor),
            encoder=str(encoder),
            uncached_decoder=str(uncached),
            cached_decoder=str(cached),
            tokens=str(tokens),
            num_threads=num_threads,
            decoding_method="greedy_search",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            self._recognizer.decode_stream(stream)
            return str(stream.result.text).strip()


class ParakeetAsr:
    """NVIDIA Parakeet TDT 0.6B (NeMo transducer) exported for sherpa-onnx.

    Best-in-class offline English accuracy at a CPU-friendly int8 size. The
    export is a standard encoder/decoder/joiner transducer; sherpa-onnx needs
    model_type="nemo_transducer" to pick the right feature pipeline.
    """

    backend = "parakeet"

    def __init__(self, model_dir: Path | str, num_threads: int = 4) -> None:
        self.model_dir = Path(model_dir)
        self.name = self.model_dir.name
        if not self.model_dir.is_dir():
            raise AsrError(f"model folder not found: {self.model_dir}")

        encoder = _pick(self.model_dir, ["encoder.int8.onnx", "encoder.onnx"])
        decoder = _pick(self.model_dir, ["decoder.int8.onnx", "decoder.onnx"])
        joiner = _pick(self.model_dir, ["joiner.int8.onnx", "joiner.onnx"])
        tokens = _pick(self.model_dir, ["tokens.txt"])

        sherpa = _import_sherpa()
        self._lock = threading.Lock()
        self._recognizer = sherpa.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            self._recognizer.decode_stream(stream)
            return str(stream.result.text).strip()


class CloudAsr:
    """Optional cloud ASR (Deepgram nova-3 / OpenAI whisper-1), key-gated.

    Local stays the default (offline rule); cloud is an owner-selected option
    for lower latency / higher accuracy on the laptop. Keys are session-only.
    """

    backend = "cloud"

    def __init__(self, provider: str, api_key: str) -> None:
        self.provider = str(provider)
        self.api_key = str(api_key)
        self.name = f"cloud:{provider}"

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        import io
        import wave

        import requests

        if samples.size == 0:
            return ""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(
                (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            )
        data = buf.getvalue()
        try:
            if self.provider == "deepgram":
                resp = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    params={
                        "model": "nova-3",
                        "smart_format": "true",
                        "language": "en",
                    },
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav",
                    },
                    data=data,
                    timeout=20,
                )
                resp.raise_for_status()
                payload = resp.json()
                return str(
                    payload["results"]["channels"][0]["alternatives"][0]["transcript"]
                ).strip()
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", data, "audio/wav")},
                data={"model": "whisper-1", "language": "en"},
                timeout=30,
            )
            resp.raise_for_status()
            return str(resp.json().get("text", "")).strip()
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            raise AsrError(f"cloud ASR ({self.provider}) failed: {exc}") from exc


def discover_models(models_dir: Path | str) -> dict[str, Path]:
    """Map backend name -> folder for every ASR model present on disk."""
    root = Path(models_dir)
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("sherpa-onnx-whisper"):
            found["whisper"] = folder
        elif folder.name.startswith("sherpa-onnx-moonshine"):
            found["moonshine"] = folder
        elif folder.name.startswith("sherpa-onnx-nemo"):
            found["parakeet"] = folder
    return found


def create_asr(models_dir: Path | str, backend: str = "whisper", num_threads: int = 2):
    """Build the requested backend, raising AsrError with a clear message."""
    models = discover_models(models_dir)
    if backend not in models:
        available = ", ".join(models) or "none"
        raise AsrError(f"backend '{backend}' unavailable (found: {available})")
    if backend == "whisper":
        return WhisperAsr(models[backend], num_threads=num_threads)
    if backend == "parakeet":
        return ParakeetAsr(models[backend], num_threads=max(4, num_threads))
    return MoonshineAsr(models[backend], num_threads=num_threads)
