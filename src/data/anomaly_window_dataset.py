from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, Sampler

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

    def condition_id_at(self, index: int) -> str:
        record_index, _ = self._index[index]
        record = self.records[record_index]
        return f"{record.dataset_name}/{record.condition_id}"

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

    def _window_start(
        self,
        sample_count: int,
        window_samples: int,
        window_index: int,
    ) -> int:
        maximum_start = max(sample_count - window_samples, 0)
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
        information = sf.info(str(record.path))
        source_rate = int(information.samplerate)
        source_window_samples = int(
            np.ceil(self.target_samples * source_rate / self.target_sample_rate)
        )
        start = self._window_start(
            int(information.frames),
            source_window_samples,
            window_index,
        )
        audio, _ = sf.read(
            str(record.path),
            start=start,
            frames=source_window_samples,
            dtype="float32",
            always_2d=True,
        )
        audio = self._resample(audio, source_rate)
        window = audio[: self.target_samples]
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


class ConditionBatchSampler(Sampler[list[int]]):
    """Build batches from one operating condition without mixing condition IDs."""

    def __init__(
        self,
        dataset: AnomalyWindowDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[str, list[int]] = {}
        for index in range(len(dataset)):
            grouped.setdefault(dataset.condition_id_at(index), []).append(index)
        self._grouped_indices = {
            condition_id: tuple(indices)
            for condition_id, indices in sorted(grouped.items())
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices_tuple in self._grouped_indices.values():
            indices = list(indices_tuple)
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self._grouped_indices.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self._grouped_indices.values()
        )


class HybridConditionBatchSampler(Sampler[list[int]]):
    """Yield class-balanced batches without mixing operating conditions."""

    def __init__(
        self,
        dataset: AnomalyWindowDataset,
        batch_size: int,
        shuffle: bool,
        seed: int = 42,
    ) -> None:
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("hybrid batch size must be a positive even integer.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[str, dict[int, list[int]]] = {}
        for index, (record_index, _) in enumerate(dataset._index):
            record = dataset.records[record_index]
            condition = dataset.condition_id_at(index)
            grouped.setdefault(condition, {0: [], 1: []})[
                0 if record.is_normal else 1
            ].append(index)
        missing = [
            condition
            for condition, labels in grouped.items()
            if not labels[0] or not labels[1]
        ]
        if missing:
            raise ValueError(
                "each hybrid condition needs both labels; missing: "
                + ", ".join(sorted(missing))
            )
        self._grouped_indices = {
            condition: {label: tuple(indices) for label, indices in labels.items()}
            for condition, labels in sorted(grouped.items())
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        half = self.batch_size // 2
        batches: list[list[int]] = []
        for labels in self._grouped_indices.values():
            normal = list(labels[0])
            anomalous = list(labels[1])
            if self.shuffle:
                generator.shuffle(normal)
                generator.shuffle(anomalous)
            batch_count = math.ceil(max(len(normal), len(anomalous)) / half)
            for batch_index in range(batch_count):
                start = batch_index * half
                batch = [
                    normal[index % len(normal)]
                    for index in range(start, start + half)
                ]
                batch.extend(
                    anomalous[index % len(anomalous)]
                    for index in range(start, start + half)
                )
                if self.shuffle:
                    generator.shuffle(batch)
                batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        half = self.batch_size // 2
        return sum(
            math.ceil(max(len(labels[0]), len(labels[1])) / half)
            for labels in self._grouped_indices.values()
        )
