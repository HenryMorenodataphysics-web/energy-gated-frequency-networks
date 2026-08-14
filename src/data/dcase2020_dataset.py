from __future__ import annotations

import re
from pathlib import Path

from .anomaly_protocol import (
    AnomalyAudioRecord,
    AnomalyDataSplit,
    split_anomaly_records,
    validate_anomaly_split,
)


_FILENAME_PATTERN = re.compile(
    r"^(?P<label>normal|anomaly)_id_(?P<machine_id>\d+)_.*\.wav$",
    re.IGNORECASE,
)


def _record_from_path(path: Path, root: Path, machine_type: str) -> AnomalyAudioRecord:
    match = _FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected DCASE 2020 filename: {path.name}")
    machine_id = f"id_{match.group('machine_id')}"
    source_label = match.group("label").lower()
    return AnomalyAudioRecord(
        path=path,
        dataset_name="dcase2020",
        machine_type=machine_type,
        machine_id=machine_id,
        condition_id=f"{machine_type}/{machine_id}",
        group_id=path.relative_to(root).as_posix(),
        label="normal" if source_label == "normal" else "anomalous",
        metadata=(("official_partition", path.parent.name),),
    )


def find_dcase2020_development_split(
    root: str | Path,
    validation_normal_ratio: float = 0.15,
    seed: int = 42,
) -> AnomalyDataSplit:
    """Load one DCASE 2020 machine while preserving its official test set."""
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"expected DCASE train/ and test/ directories under {root}."
        )
    machine_type = root.name.lower()
    training_records = [
        _record_from_path(path, root, machine_type)
        for path in sorted(train_dir.glob("*.wav"))
    ]
    official_test = [
        _record_from_path(path, root, machine_type)
        for path in sorted(test_dir.glob("*.wav"))
    ]
    if not training_records or not official_test:
        raise FileNotFoundError(f"no DCASE wav files found under {root}.")
    if any(not record.is_normal for record in training_records):
        raise ValueError("the official DCASE training partition must be normal-only.")

    fitted_split = split_anomaly_records(
        training_records,
        validation_normal_ratio=validation_normal_ratio,
        test_normal_ratio=0.0,
        seed=seed,
    )
    split = AnomalyDataSplit(
        train=fitted_split.train,
        validation=fitted_split.validation,
        test=tuple(official_test),
    )
    validate_anomaly_split(split)
    return split
