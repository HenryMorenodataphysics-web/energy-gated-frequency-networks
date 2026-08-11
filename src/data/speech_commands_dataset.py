from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


DEFAULT_SPEECH_COMMANDS_LABELS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


class SpeechCommandsSubset(Dataset):
    """Google Speech Commands wrapper with fixed-length waveform output."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        labels: tuple[str, ...] = DEFAULT_SPEECH_COMMANDS_LABELS,
        download: bool = False,
        target_sample_rate: int = 16_000,
        duration: float = 1.0,
        augment: bool = False,
        noise_std: float = 0.003,
        gain_range: tuple[float, float] = (0.8, 1.2),
        max_shift_fraction: float = 0.08,
        max_mask_fraction: float = 0.08,
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.labels = tuple(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.target_sample_rate = target_sample_rate
        self.target_samples = int(target_sample_rate * duration)
        self.augment = augment
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.max_shift_fraction = max_shift_fraction
        self.max_mask_fraction = max_mask_fraction

        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root),
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=download,
            subset=subset,
        )
        self.records: list[tuple[int, str]] = []
        for index in range(len(self.dataset)):
            label = self.dataset.get_metadata(index)[2]
            if label in self.label_to_index:
                self.records.append((index, label))

        if not self.records:
            raise RuntimeError(
                f"No selected labels found in Speech Commands subset={subset}. "
                f"Requested labels={self.labels}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_index, label = self.records[index]
        waveform, sample_rate, *_ = self.dataset[dataset_index]
        waveform = waveform.float()

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        waveform = self._crop_or_pad(waveform)
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True).clamp_min(1e-6)
        waveform = waveform / std

        if self.augment:
            waveform = self._augment_waveform(waveform)

        target = torch.tensor(self.label_to_index[label], dtype=torch.long)
        return waveform, target

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        current_samples = waveform.shape[-1]
        if current_samples > self.target_samples:
            return waveform[..., : self.target_samples]
        if current_samples < self.target_samples:
            return F.pad(waveform, (0, self.target_samples - current_samples))
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(1).uniform_(*self.gain_range).item()
        waveform = waveform * gain

        max_shift = int(self.target_samples * self.max_shift_fraction)
        if max_shift > 0:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        if self.noise_std > 0:
            waveform = waveform + torch.randn_like(waveform) * self.noise_std

        max_mask = int(self.target_samples * self.max_mask_fraction)
        if max_mask > 0 and torch.rand(1).item() < 0.5:
            mask_width = int(torch.randint(1, max_mask + 1, (1,)).item())
            start = int(torch.randint(0, self.target_samples - mask_width + 1, (1,)).item())
            waveform[:, start : start + mask_width] = 0.0

        return waveform
