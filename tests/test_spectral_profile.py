from __future__ import annotations

import numpy as np

from src.utils.spectral_profile import (
    average_channel_power_spectrogram,
    equal_band_edges,
    summarize_frequency_bands,
)


def sine_wave(frequency: float, amplitude: float = 1.0) -> tuple[np.ndarray, int]:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * time), sample_rate


def band_statistics(audio: np.ndarray, sample_rate: int):
    frequencies, _, power = average_channel_power_spectrogram(
        audio,
        sample_rate,
        n_fft=512,
        hop_length=256,
    )
    return summarize_frequency_bands(
        frequencies,
        power,
        equal_band_edges(sample_rate, num_bands=3),
    )


def test_tone_energy_is_assigned_to_expected_equal_band() -> None:
    low_tone, sample_rate = sine_wave(500.0)
    high_tone, _ = sine_wave(3_500.0)

    low_statistics = band_statistics(low_tone, sample_rate)
    high_statistics = band_statistics(high_tone, sample_rate)

    assert np.argmax([band.mean_power for band in low_statistics]) == 0
    assert np.argmax([band.mean_power for band in high_statistics]) == 2


def test_relative_amplitude_is_preserved_in_band_power() -> None:
    loud, sample_rate = sine_wave(500.0, amplitude=1.0)
    quiet, _ = sine_wave(500.0, amplitude=0.25)

    loud_db = band_statistics(loud, sample_rate)[0].mean_power_db
    quiet_db = band_statistics(quiet, sample_rate)[0].mean_power_db

    assert 11.5 < loud_db - quiet_db < 12.5


def test_multichannel_power_does_not_cancel_opposite_phase_channels() -> None:
    tone, sample_rate = sine_wave(500.0)
    opposite_phase = np.stack([tone, -tone], axis=1)

    statistics = band_statistics(opposite_phase, sample_rate)

    assert statistics[0].mean_power > 0.1
    assert statistics[0].mean_power > statistics[1].mean_power * 1_000
