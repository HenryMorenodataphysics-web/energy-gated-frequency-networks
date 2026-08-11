from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def add_noise_at_snr(
    signal: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise at a target signal-to-noise ratio."""

    signal_power = signal.square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn(signal.shape, generator=generator, device=signal.device)
    return signal + noise * torch.sqrt(noise_power)


def _sine_wave(
    frequency_hz: float,
    time: torch.Tensor,
    amplitude: float,
    phase: float,
) -> torch.Tensor:
    return amplitude * torch.sin(2 * math.pi * frequency_hz * time + phase)


def generate_synthetic_frequency_batch(
    batch_size: int,
    sample_rate: int = 16_000,
    duration: float = 1.0,
    num_classes: int = 4,
    snr_db: float | None = None,
    difficulty: str = "easy",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate simple frequency-labeled waveforms.

    Easy classes are intentionally controlled:
        0: low-frequency dominant
        1: mid-frequency dominant
        2: high-frequency dominant
        3: low + high mixture

    Medium and hard classes are closer and include distractor tones, random time shifts,
    amplitude variation, and partial dropout of components.
    """

    if num_classes != 4:
        raise ValueError("The initial synthetic generator expects num_classes=4.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    num_samples = int(sample_rate * duration)
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate

    labels = torch.randint(0, num_classes, (batch_size,), generator=generator)
    waves = []

    if difficulty == "easy":
        class_frequencies = {
            0: (180.0, 260.0),
            1: (900.0, 1_300.0),
            2: (3_000.0, 4_200.0),
            3: (220.0, 3_600.0),
        }
        jitter_range = 0.08
        distractor_count = 0
        dropout_probability = 0.0
        chirp_strength = 0.0
    elif difficulty == "medium":
        class_frequencies = {
            0: (360.0, 620.0),
            1: (620.0, 980.0),
            2: (980.0, 1_520.0),
            3: (420.0, 1_260.0),
        }
        jitter_range = 0.10
        distractor_count = 1
        dropout_probability = 0.05
        chirp_strength = 0.05
    elif difficulty == "hard":
        class_frequencies = {
            0: (420.0, 760.0),
            1: (520.0, 980.0),
            2: (700.0, 1_260.0),
            3: (440.0, 1_180.0),
        }
        jitter_range = 0.18
        distractor_count = 2
        dropout_probability = 0.20
        chirp_strength = 0.12
    else:
        raise ValueError("difficulty must be 'easy', 'medium', or 'hard'.")

    for label in labels.tolist():
        freqs = class_frequencies[label]
        wave = torch.zeros_like(time)
        for freq in freqs:
            if torch.rand(1, generator=generator).item() < dropout_probability:
                continue
            jitter = torch.empty(1).uniform_(
                -jitter_range,
                jitter_range,
                generator=generator,
            ).item()
            amplitude = torch.empty(1).uniform_(0.6, 1.0, generator=generator).item()
            phase = torch.empty(1).uniform_(0.0, 2 * math.pi, generator=generator).item()
            chirp = 1 + chirp_strength * (time - time.mean())
            wave = wave + amplitude * torch.sin(
                2 * math.pi * freq * (1 + jitter) * chirp * time + phase
            )

        for _ in range(distractor_count):
            distractor_freq = torch.empty(1).uniform_(180.0, 3_800.0, generator=generator).item()
            distractor_amp = torch.empty(1).uniform_(0.05, 0.30, generator=generator).item()
            distractor_phase = torch.empty(1).uniform_(0.0, 2 * math.pi, generator=generator).item()
            wave = wave + _sine_wave(distractor_freq, time, distractor_amp, distractor_phase)

        envelope = torch.hann_window(num_samples, periodic=False)
        wave = wave * envelope
        if difficulty in {"medium", "hard"}:
            shift = int(torch.randint(-num_samples // 12, num_samples // 12 + 1, (1,), generator=generator))
            wave = torch.roll(wave, shifts=shift)
        wave = wave / wave.abs().amax().clamp_min(1e-6)
        waves.append(wave)

    x = torch.stack(waves, dim=0).unsqueeze(1)
    if snr_db is not None:
        x = add_noise_at_snr(x, snr_db=snr_db, generator=generator)

    return x, labels


@dataclass(frozen=True)
class SyntheticConfig:
    num_samples: int = 512
    sample_rate: int = 16_000
    duration: float = 1.0
    num_classes: int = 4
    snr_db: float | None = 20.0
    difficulty: str = "easy"
    seed: int = 42


class SyntheticFrequencyDataset(Dataset):
    """Small in-memory dataset for fast architecture validation."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.x, self.y = generate_synthetic_frequency_batch(
            batch_size=self.config.num_samples,
            sample_rate=self.config.sample_rate,
            duration=self.config.duration,
            num_classes=self.config.num_classes,
            snr_db=self.config.snr_db,
            difficulty=self.config.difficulty,
            seed=self.config.seed,
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def numpy_fft_magnitude(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return positive FFT frequencies and magnitudes for plotting notebooks."""

    freqs = np.fft.rfftfreq(signal.shape[-1], d=1 / sample_rate)
    magnitude = np.abs(np.fft.rfft(signal))
    return freqs, magnitude
