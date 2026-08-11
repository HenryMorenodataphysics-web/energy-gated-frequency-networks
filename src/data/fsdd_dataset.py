from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FSDDRecord:
    path: Path
    digit: int
    speaker: str
    index: int


def parse_fsdd_filename(path: Path) -> FSDDRecord:
    """Parse filenames like 7_jackson_32.wav."""

    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected FSDD filename: {path.name}")

    digit = int(parts[0])
    index = int(parts[-1])
    speaker = "_".join(parts[1:-1])
    return FSDDRecord(path=path, digit=digit, speaker=speaker, index=index)


def find_fsdd_recordings(root: str | Path) -> list[FSDDRecord]:
    """Find FSDD wav files under common folder layouts."""

    root = Path(root)
    candidates = [
        root / "recordings",
        root / "free-spoken-digit-dataset" / "recordings",
        root,
    ]

    wav_paths: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            wav_paths.extend(sorted(candidate.glob("*.wav")))

    if not wav_paths:
        wav_paths = sorted(root.rglob("*.wav"))

    records = []
    for path in wav_paths:
        try:
            records.append(parse_fsdd_filename(path))
        except ValueError:
            continue

    if not records:
        raise FileNotFoundError(
            f"No FSDD wav recordings found under {root}. Expected files like 0_george_0.wav."
        )

    return records


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(path)
    audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)

    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def crop_or_pad(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.shape[0] > target_samples:
        return audio[:target_samples]
    if audio.shape[0] < target_samples:
        pad_width = target_samples - audio.shape[0]
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio


class FSDDDataset(Dataset):
    """Free Spoken Digit Dataset loader.

    Returns:
        x: [1, target_samples]
        y: digit label from 0 to 9
    """

    def __init__(
        self,
        records: list[FSDDRecord],
        target_sample_rate: int = 16_000,
        duration: float = 1.0,
        normalize: bool = True,
        augment: bool = False,
        noise_std: float = 0.005,
        gain_range: tuple[float, float] = (0.75, 1.25),
        max_shift_fraction: float = 0.08,
        max_mask_fraction: float = 0.08,
    ) -> None:
        self.records = records
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.target_samples = int(target_sample_rate * duration)
        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.max_shift_fraction = max_shift_fraction
        self.max_mask_fraction = max_mask_fraction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        audio, sample_rate = load_wav_mono(record.path)
        audio = resample_audio(audio, sample_rate, self.target_sample_rate)
        audio = crop_or_pad(audio, self.target_samples)

        if self.normalize:
            audio = audio - audio.mean()
            std = audio.std()
            if std > 1e-6:
                audio = audio / std

        x = torch.from_numpy(audio).float().unsqueeze(0)
        if self.augment:
            x = self._augment_waveform(x)
        y = torch.tensor(record.digit, dtype=torch.long)
        return x, y

    def _augment_waveform(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(1).uniform_(*self.gain_range).item()
        x = x * gain

        max_shift = int(self.target_samples * self.max_shift_fraction)
        if max_shift > 0:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        max_mask = int(self.target_samples * self.max_mask_fraction)
        if max_mask > 0 and torch.rand(1).item() < 0.5:
            mask_width = int(torch.randint(1, max_mask + 1, (1,)).item())
            start = int(torch.randint(0, self.target_samples - mask_width + 1, (1,)).item())
            x[:, start : start + mask_width] = 0.0

        return x


def split_records_by_speaker(
    records: list[FSDDRecord],
    test_speakers: set[str],
    val_speakers: set[str],
) -> tuple[list[FSDDRecord], list[FSDDRecord], list[FSDDRecord]]:
    train_records = []
    val_records = []
    test_records = []

    for record in records:
        if record.speaker in test_speakers:
            test_records.append(record)
        elif record.speaker in val_speakers:
            val_records.append(record)
        else:
            train_records.append(record)

    return train_records, val_records, test_records


def split_records_random(
    records: list[FSDDRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[FSDDRecord], list[FSDDRecord], list[FSDDRecord]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator).tolist()

    test_size = int(len(records) * test_ratio)
    val_size = int(len(records) * val_ratio)
    test_indices = set(indices[:test_size])
    val_indices = set(indices[test_size : test_size + val_size])

    train_records = []
    val_records = []
    test_records = []
    for idx, record in enumerate(records):
        if idx in test_indices:
            test_records.append(record)
        elif idx in val_indices:
            val_records.append(record)
        else:
            train_records.append(record)

    return train_records, val_records, test_records
