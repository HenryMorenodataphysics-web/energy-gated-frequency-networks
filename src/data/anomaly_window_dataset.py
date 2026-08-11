from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.utils.spectral_profile import load_wav_preserve_level

from .anomaly_protocol import AnomalyAudioRecord


class AnomalyWindowDataset(Dataset):
    """Load level-preserving multichannel windows from generic anomaly records."""

    def __init__(
        self,
        records: Sequence[AnomalyAudioRecord],
        target_sample_rate: int,
        duration_seconds: float,
        crop_mode: str = "random",
        evaluation_windows: int = 1,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty.")
        if target_sample_rate <= 0 or duration_seconds <= 0:
            raise ValueError("sample rate and duration must be positive.")
        if crop_mode not in {"random", "center", "grid"}:
            raise ValueError("crop_mode must be random, center, or grid.")
        if evaluation_windows <= 0:
            raise ValueError("evaluation_windows must be positive.")
        if crop_mode != "grid" and evaluation_windows != 1:
            raise ValueError("multiple evaluation windows require crop_mode='grid'.")

        self.records = tuple(records)
        self.target_sample_rate = int(target_sample_rate)
        self.target_samples = int(round(target_sample_rate * duration_seconds))
        self.crop_mode = crop_mode
        self.evaluation_windows = int(evaluation_windows)
        repetitions = self.evaluation_windows if crop_mode == "grid" else 1
        self._index = tuple(
            (record_index, window_index)
            for record_index in range(len(self.records))
            for window_index in range(repetitions)
        )

    def __len__(self) -> int:
        return len(self._index)

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.target_sample_rate:
            return audio
        divisor = int(np.gcd(source_rate, self.target_sample_rate))
        return resample_poly(
            audio,
            self.target_sample_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32)

    def _window_start(self, sample_count: int, window_index: int) -> int:
        maximum_start = max(sample_count - self.target_samples, 0)
        if self.crop_mode == "random":
            return int(torch.randint(maximum_start + 1, (1,)).item())
        if self.crop_mode == "center":
            return maximum_start // 2
        if self.evaluation_windows == 1:
            return maximum_start // 2
        return int(round(maximum_start * window_index / (self.evaluation_windows - 1)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record_index, window_index = self._index[index]
        record = self.records[record_index]
        audio, source_rate = load_wav_preserve_level(record.path)
        audio = self._resample(audio, source_rate)
        start = self._window_start(audio.shape[0], window_index)
        window = audio[start : start + self.target_samples]
        if window.shape[0] < self.target_samples:
            window = np.pad(
                window,
                ((0, self.target_samples - window.shape[0]), (0, 0)),
            )

        return {
            "waveform": torch.from_numpy(window.T.copy()).float(),
            "label": torch.tensor(0 if record.is_normal else 1, dtype=torch.long),
            "condition_id": f"{record.dataset_name}/{record.condition_id}",
            "recording_id": f"{record.dataset_name}/{record.group_id}",
        }
