from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WorldFeatures:
    """WORLD parameters for one mono waveform."""

    f0: np.ndarray
    spectral_envelope: np.ndarray
    aperiodicity: np.ndarray
    time_axis: np.ndarray
    sample_rate: int
    frame_period_ms: float

    def updated(self, **changes: object) -> "WorldFeatures":
        return replace(self, **changes)


def _audio_dependencies():
    try:
        import librosa
        import pyworld
        import soundfile
    except ImportError as exc:
        raise RuntimeError(
            "audio preprocessing requires librosa, pyworld, and soundfile"
        ) from exc
    return librosa, pyworld, soundfile


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    librosa, _, _ = _audio_dependencies()
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def save_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    _, _, soundfile = _audio_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    soundfile.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_16")


def analyse_world(
    waveform: np.ndarray,
    sample_rate: int,
    frame_period_ms: float = 5.0,
    f0_floor_hz: float = 50.0,
    f0_ceiling_hz: float = 600.0,
) -> WorldFeatures:
    _, pyworld, _ = _audio_dependencies()
    audio = np.ascontiguousarray(waveform, dtype=np.float64)
    initial_f0, time_axis = pyworld.dio(
        audio,
        sample_rate,
        f0_floor=f0_floor_hz,
        f0_ceil=f0_ceiling_hz,
        frame_period=frame_period_ms,
    )
    f0 = pyworld.stonemask(audio, initial_f0, time_axis, sample_rate)
    spectral_envelope = pyworld.cheaptrick(audio, f0, time_axis, sample_rate)
    aperiodicity = pyworld.d4c(audio, f0, time_axis, sample_rate)
    return WorldFeatures(
        f0=np.asarray(f0, dtype=np.float64),
        spectral_envelope=np.asarray(spectral_envelope, dtype=np.float64),
        aperiodicity=np.asarray(aperiodicity, dtype=np.float64),
        time_axis=np.asarray(time_axis, dtype=np.float64),
        sample_rate=int(sample_rate),
        frame_period_ms=float(frame_period_ms),
    )


def synthesise_world(features: WorldFeatures) -> np.ndarray:
    _, pyworld, _ = _audio_dependencies()
    frame_count = len(features.f0)
    if features.spectral_envelope.shape[0] != frame_count:
        raise ValueError("F0 and spectral-envelope frame counts differ")
    if features.aperiodicity.shape != features.spectral_envelope.shape:
        raise ValueError("spectral-envelope and aperiodicity shapes differ")
    return np.asarray(
        pyworld.synthesize(
            np.ascontiguousarray(features.f0, dtype=np.float64),
            np.ascontiguousarray(features.spectral_envelope, dtype=np.float64),
            np.ascontiguousarray(features.aperiodicity, dtype=np.float64),
            features.sample_rate,
            frame_period=features.frame_period_ms,
        ),
        dtype=np.float64,
    )


def frame_power(spectral_envelope: np.ndarray) -> np.ndarray:
    spectrum = np.maximum(np.asarray(spectral_envelope, dtype=np.float64), 1e-12)
    return np.mean(spectrum, axis=1)


def scale_to_frame_power(
    spectral_envelope: np.ndarray,
    target_power: np.ndarray,
) -> np.ndarray:
    spectrum = np.maximum(np.asarray(spectral_envelope, dtype=np.float64), 1e-12)
    target = np.maximum(np.asarray(target_power, dtype=np.float64), 1e-12)
    if target.ndim != 1 or len(target) != spectrum.shape[0]:
        raise ValueError("target_power must have one value per WORLD frame")
    current = frame_power(spectrum)
    return spectrum * (target / current)[:, None]


def estimate_carrier(features: list[WorldFeatures]) -> np.ndarray:
    """Estimate a label-free, time-invariant spectral carrier."""
    if not features:
        raise ValueError("at least one utterance is required to estimate a carrier")
    bins = {item.spectral_envelope.shape[1] for item in features}
    if len(bins) != 1:
        raise ValueError("all carrier utterances must use the same FFT size")
    utterance_medians = [
        np.median(np.maximum(item.spectral_envelope, 1e-12), axis=0)
        for item in features
    ]
    carrier = np.median(np.stack(utterance_medians), axis=0)
    return carrier / max(float(np.mean(carrier)), 1e-12)

