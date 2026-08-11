from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram


@dataclass(frozen=True)
class BandStatistics:
    band_index: int
    low_hz: float
    high_hz: float
    mean_power: float
    mean_power_db: float
    variance_db: float
    mean_abs_delta_db: float


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert PCM samples to float without normalizing each recording."""
    if np.issubdtype(audio.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        return audio.astype(np.float32) / scale
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        return (audio.astype(np.float32) - midpoint) / midpoint
    return audio.astype(np.float32)


def load_wav_preserve_level(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as [samples, channels] while preserving relative level."""
    sample_rate, audio = wavfile.read(Path(path))
    audio = pcm_to_float(audio)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError(f"Expected mono or multichannel WAV, got shape {audio.shape}.")
    return audio, int(sample_rate)


def average_channel_power_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1_024,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency, time, and channel-averaged power spectral density."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError("audio must have shape [samples] or [samples, channels].")
    if n_fft <= 1:
        raise ValueError("n_fft must be greater than 1.")
    if hop_length <= 0 or hop_length > n_fft:
        raise ValueError("hop_length must be between 1 and n_fft.")
    if audio.shape[0] < n_fft:
        raise ValueError("audio must contain at least n_fft samples.")

    noverlap = n_fft - hop_length
    channel_psd = []
    frequencies: np.ndarray | None = None
    times: np.ndarray | None = None
    for channel in range(audio.shape[1]):
        channel_frequencies, channel_times, psd = spectrogram(
            audio[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        frequencies = channel_frequencies
        times = channel_times
        channel_psd.append(psd.astype(np.float64, copy=False))

    if frequencies is None or times is None:
        raise RuntimeError("No channels were available for spectral analysis.")
    mean_power = np.mean(np.stack(channel_psd, axis=0), axis=0)
    return frequencies, times, mean_power


def equal_band_edges(
    sample_rate: int,
    num_bands: int = 3,
    max_frequency_hz: float | None = None,
) -> np.ndarray:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive.")
    nyquist = sample_rate / 2.0
    upper = nyquist if max_frequency_hz is None else float(max_frequency_hz)
    if upper <= 0 or upper > nyquist:
        raise ValueError(f"max_frequency_hz must be in (0, {nyquist}].")
    return np.linspace(0.0, upper, num_bands + 1)


def summarize_frequency_bands(
    frequencies: np.ndarray,
    power: np.ndarray,
    band_edges_hz: np.ndarray,
    epsilon: float = 1e-12,
) -> list[BandStatistics]:
    """Summarize integrated power and temporal variation for each band."""
    if power.ndim != 2 or power.shape[0] != frequencies.shape[0]:
        raise ValueError("power must have shape [frequency, time].")
    if band_edges_hz.ndim != 1 or band_edges_hz.size < 2:
        raise ValueError("band_edges_hz must contain at least two edges.")
    if not np.all(np.diff(band_edges_hz) > 0):
        raise ValueError("band_edges_hz must be strictly increasing.")

    frequency_resolution = (
        float(np.median(np.diff(frequencies))) if frequencies.size > 1 else 1.0
    )
    summaries: list[BandStatistics] = []
    for index, (low_hz, high_hz) in enumerate(
        zip(band_edges_hz[:-1], band_edges_hz[1:], strict=True)
    ):
        is_last = index == band_edges_hz.size - 2
        mask = (frequencies >= low_hz) & (
            frequencies <= high_hz if is_last else frequencies < high_hz
        )
        if not np.any(mask):
            raise ValueError(f"Band {low_hz}-{high_hz} Hz contains no frequency bins.")

        frame_power = power[mask].sum(axis=0) * frequency_resolution
        frame_power_db = 10.0 * np.log10(np.maximum(frame_power, epsilon))
        mean_abs_delta = (
            float(np.mean(np.abs(np.diff(frame_power_db))))
            if frame_power_db.size > 1
            else 0.0
        )
        mean_power = float(np.mean(frame_power))
        summaries.append(
            BandStatistics(
                band_index=index,
                low_hz=float(low_hz),
                high_hz=float(high_hz),
                mean_power=mean_power,
                mean_power_db=float(10.0 * np.log10(max(mean_power, epsilon))),
                variance_db=float(np.var(frame_power_db)),
                mean_abs_delta_db=mean_abs_delta,
            )
        )
    return summaries


def analyze_wav_bands(
    path: str | Path,
    n_fft: int = 1_024,
    hop_length: int = 512,
    num_bands: int = 3,
    max_frequency_hz: float | None = None,
) -> tuple[int, int, float, list[BandStatistics]]:
    audio, sample_rate = load_wav_preserve_level(path)
    frequencies, _, power = average_channel_power_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    edges = equal_band_edges(sample_rate, num_bands, max_frequency_hz)
    summaries = summarize_frequency_bands(frequencies, power, edges)
    duration_seconds = audio.shape[0] / sample_rate
    return sample_rate, audio.shape[1], duration_seconds, summaries
