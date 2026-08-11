from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .anomaly_protocol import AnomalyAudioRecord


MIMII_MACHINE_TYPES = ("fan", "pump", "slider", "slide_rail", "valve")
MIMII_LABEL_TO_INDEX = {"normal": 0, "abnormal": 1}


@dataclass(frozen=True)
class MIMIIRecord:
    path: Path
    label: str
    machine_type: str
    machine_id: str
    snr: str

    @property
    def target(self) -> int:
        return MIMII_LABEL_TO_INDEX[self.label]


def to_anomaly_audio_record(record: MIMIIRecord) -> AnomalyAudioRecord:
    """Adapt a MIMII-specific record to the shared anomaly protocol."""
    condition_id = "/".join((record.machine_type, record.machine_id, record.snr))
    return AnomalyAudioRecord(
        path=record.path,
        dataset_name="mimii",
        machine_type=record.machine_type,
        machine_id=record.machine_id,
        condition_id=condition_id,
        group_id=record.path.as_posix(),
        label="normal" if record.label == "normal" else "anomalous",
        metadata=(("snr", record.snr), ("source_label", record.label)),
    )


def _normalize_machine_type(value: str) -> str:
    if value == "slide":
        return "slide_rail"
    return value


def parse_mimii_path(path: Path) -> MIMIIRecord | None:
    parts = [part.lower() for part in path.parts]

    label = None
    for candidate in ("normal", "abnormal"):
        if candidate in parts:
            label = candidate
            break
    if label is None:
        return None

    machine_type = "unknown"
    for part in parts:
        normalized = _normalize_machine_type(part)
        if normalized in MIMII_MACHINE_TYPES:
            machine_type = normalized
            break

    machine_id = "unknown"
    for part in parts:
        if part.startswith("id_"):
            machine_id = part
            break

    snr = "unknown"
    for part in parts:
        if part.endswith("db"):
            snr = part
            break

    return MIMIIRecord(
        path=path,
        label=label,
        machine_type=machine_type,
        machine_id=machine_id,
        snr=snr,
    )


def find_mimii_recordings(
    root: str | Path,
    machine_type: str = "all",
    machine_id: str = "all",
    snr: str = "all",
) -> list[MIMIIRecord]:
    root = Path(root)
    records: list[MIMIIRecord] = []

    for path in sorted(root.rglob("*.wav")):
        record = parse_mimii_path(path)
        if record is None:
            continue
        if machine_type != "all" and record.machine_type != machine_type:
            continue
        if machine_id != "all" and record.machine_id != machine_id:
            continue
        if snr != "all" and record.snr != snr:
            continue
        records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No MIMII wav files found under {root}. Expected paths containing "
            "normal/abnormal folders, for example fan/id_00/normal/*.wav."
        )

    return records


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(path)
    audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / np.iinfo(audio.dtype).max
    else:
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


def crop_or_pad(
    audio: np.ndarray,
    target_samples: int,
    crop_mode: str = "center",
) -> np.ndarray:
    if audio.shape[0] > target_samples:
        if crop_mode == "random":
            start = int(torch.randint(0, audio.shape[0] - target_samples + 1, (1,)).item())
        else:
            start = max((audio.shape[0] - target_samples) // 2, 0)
        return audio[start : start + target_samples]
    if audio.shape[0] < target_samples:
        return np.pad(audio, (0, target_samples - audio.shape[0]), mode="constant")
    return audio


class MIMIIDataset(Dataset):
    """MIMII normal-vs-abnormal audio dataset.

    Returns:
        x: waveform tensor shaped [1, target_samples]
        y: 0 for normal, 1 for abnormal
    """

    def __init__(
        self,
        records: list[MIMIIRecord],
        target_sample_rate: int = 16_000,
        duration: float = 2.0,
        normalize: bool = True,
        crop_mode: str = "center",
        augment: bool = False,
        noise_std: float = 0.003,
        gain_range: tuple[float, float] = (0.8, 1.2),
        max_shift_fraction: float = 0.08,
        max_mask_fraction: float = 0.08,
    ) -> None:
        self.records = records
        self.target_sample_rate = target_sample_rate
        self.target_samples = int(target_sample_rate * duration)
        self.normalize = normalize
        self.crop_mode = crop_mode
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
        audio = crop_or_pad(audio, self.target_samples, self.crop_mode)

        if self.normalize:
            audio = audio - audio.mean()
            std = audio.std()
            if std > 1e-6:
                audio = audio / std

        x = torch.from_numpy(audio).float().unsqueeze(0)
        if self.augment:
            x = self._augment_waveform(x)
        y = torch.tensor(record.target, dtype=torch.long)
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


def split_records_stratified(
    records: list[MIMIIRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[MIMIIRecord], list[MIMIIRecord], list[MIMIIRecord]]:
    generator = torch.Generator().manual_seed(seed)
    by_label: dict[str, list[MIMIIRecord]] = {"normal": [], "abnormal": []}
    for record in records:
        by_label[record.label].append(record)

    train_records: list[MIMIIRecord] = []
    val_records: list[MIMIIRecord] = []
    test_records: list[MIMIIRecord] = []

    for label_records in by_label.values():
        indices = torch.randperm(len(label_records), generator=generator).tolist()
        test_size = int(len(label_records) * test_ratio)
        val_size = int(len(label_records) * val_ratio)
        test_indices = indices[:test_size]
        val_indices = indices[test_size : test_size + val_size]
        train_indices = indices[test_size + val_size :]

        test_records.extend(label_records[index] for index in test_indices)
        val_records.extend(label_records[index] for index in val_indices)
        train_records.extend(label_records[index] for index in train_indices)

    return train_records, val_records, test_records


def summarize_records(records: list[MIMIIRecord]) -> dict[tuple[str, str, str, str], int]:
    summary: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        key = (record.machine_type, record.machine_id, record.snr, record.label)
        summary[key] = summary.get(key, 0) + 1
    return summary
