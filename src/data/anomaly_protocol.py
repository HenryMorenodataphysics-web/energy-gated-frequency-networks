from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AnomalyLabel = Literal["normal", "anomalous"]


@dataclass(frozen=True)
class AnomalyAudioRecord:
    """Dataset-independent description of one industrial audio example."""

    path: Path
    dataset_name: str
    machine_type: str
    machine_id: str
    condition_id: str
    group_id: str
    label: AnomalyLabel
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def is_normal(self) -> bool:
        return self.label == "normal"

    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


@dataclass(frozen=True)
class AnomalyDataSplit:
    train: tuple[AnomalyAudioRecord, ...]
    validation: tuple[AnomalyAudioRecord, ...]
    test: tuple[AnomalyAudioRecord, ...]


def _record_sort_key(record: AnomalyAudioRecord) -> tuple[str, ...]:
    return (
        record.dataset_name,
        record.condition_id,
        record.group_id,
        str(record.path),
    )


def _holdout_sizes(
    group_count: int,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    validation_size = int(group_count * validation_ratio)
    test_size = int(group_count * test_ratio)
    if group_count >= 3 and validation_ratio > 0 and validation_size == 0:
        validation_size = 1
    if group_count >= 3 and test_ratio > 0 and test_size == 0:
        test_size = 1

    while validation_size + test_size >= group_count:
        if test_size >= validation_size and test_size > 0:
            test_size -= 1
        elif validation_size > 0:
            validation_size -= 1
    return validation_size, test_size


def split_anomaly_records(
    records: list[AnomalyAudioRecord],
    validation_normal_ratio: float = 0.15,
    test_normal_ratio: float = 0.15,
    seed: int = 42,
) -> AnomalyDataSplit:
    """Create a one-class split while keeping source groups and conditions intact."""
    if not records:
        raise ValueError("records must not be empty.")
    if validation_normal_ratio < 0 or test_normal_ratio < 0:
        raise ValueError("split ratios must be non-negative.")
    if validation_normal_ratio + test_normal_ratio >= 1:
        raise ValueError("validation and test ratios must leave normal training data.")

    grouped: dict[tuple[str, str], list[AnomalyAudioRecord]] = {}
    for record in records:
        group_key = (record.dataset_name, record.group_id)
        grouped.setdefault(group_key, []).append(record)

    normal_by_condition: dict[tuple[str, str], list[list[AnomalyAudioRecord]]] = {}
    anomalous_groups: list[list[AnomalyAudioRecord]] = []
    for group in grouped.values():
        labels = {record.label for record in group}
        conditions = {(record.dataset_name, record.condition_id) for record in group}
        if len(labels) != 1:
            raise ValueError(f"group_id {group[0].group_id!r} contains mixed labels.")
        if len(conditions) != 1:
            raise ValueError(f"group_id {group[0].group_id!r} spans multiple conditions.")

        if group[0].is_normal:
            condition = next(iter(conditions))
            normal_by_condition.setdefault(condition, []).append(group)
        else:
            anomalous_groups.append(group)

    generator = random.Random(seed)
    train: list[AnomalyAudioRecord] = []
    validation: list[AnomalyAudioRecord] = []
    test: list[AnomalyAudioRecord] = []

    for condition in sorted(normal_by_condition):
        condition_groups = sorted(
            normal_by_condition[condition],
            key=lambda group: _record_sort_key(group[0]),
        )
        generator.shuffle(condition_groups)
        validation_size, test_size = _holdout_sizes(
            len(condition_groups),
            validation_normal_ratio,
            test_normal_ratio,
        )
        validation_groups = condition_groups[:validation_size]
        test_groups = condition_groups[validation_size : validation_size + test_size]
        train_groups = condition_groups[validation_size + test_size :]
        for group in train_groups:
            train.extend(group)
        for group in validation_groups:
            validation.extend(group)
        for group in test_groups:
            test.extend(group)

    for group in anomalous_groups:
        test.extend(group)

    return AnomalyDataSplit(
        train=tuple(sorted(train, key=_record_sort_key)),
        validation=tuple(sorted(validation, key=_record_sort_key)),
        test=tuple(sorted(test, key=_record_sort_key)),
    )


def validate_anomaly_split(split: AnomalyDataSplit) -> None:
    """Raise when a split leaks groups or uses anomalies for fitting."""
    if any(not record.is_normal for record in split.train):
        raise ValueError("training split must contain only normal records.")
    if any(not record.is_normal for record in split.validation):
        raise ValueError("validation split must contain only normal records.")

    group_sets = []
    for records in (split.train, split.validation, split.test):
        group_sets.append({(record.dataset_name, record.group_id) for record in records})
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError("group leakage detected between splits.")
