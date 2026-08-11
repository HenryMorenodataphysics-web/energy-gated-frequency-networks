from __future__ import annotations

from pathlib import Path

import pytest

from src.data import (
    AnomalyAudioRecord,
    MIMIIRecord,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
)


def make_record(
    dataset: str,
    condition: str,
    index: int,
    label: str = "normal",
    group_id: str | None = None,
) -> AnomalyAudioRecord:
    return AnomalyAudioRecord(
        path=Path(dataset) / condition / f"{index}.wav",
        dataset_name=dataset,
        machine_type="machine",
        machine_id=condition,
        condition_id=condition,
        group_id=group_id or f"{dataset}-{condition}-{index}",
        label=label,  # type: ignore[arg-type]
    )


def test_split_is_one_class_group_safe_and_reproducible() -> None:
    records = []
    for dataset in ("dataset_a", "dataset_b"):
        for condition in ("machine_1", "machine_2"):
            records.extend(make_record(dataset, condition, index) for index in range(10))
            records.extend(
                make_record(dataset, condition, 100 + index, label="anomalous")
                for index in range(2)
            )

    first = split_anomaly_records(records, seed=123)
    second = split_anomaly_records(records, seed=123)
    validate_anomaly_split(first)

    assert first == second
    assert all(record.is_normal for record in first.train)
    assert all(record.is_normal for record in first.validation)
    assert sum(not record.is_normal for record in first.test) == 8
    assert {(r.dataset_name, r.condition_id) for r in first.train} == {
        (dataset, condition)
        for dataset in ("dataset_a", "dataset_b")
        for condition in ("machine_1", "machine_2")
    }


def test_segments_from_one_source_group_never_leak() -> None:
    records = [make_record("dataset", "machine", index) for index in range(8)]
    records.extend(
        [
            make_record("dataset", "machine", 20, group_id="shared-source"),
            make_record("dataset", "machine", 21, group_id="shared-source"),
        ]
    )
    split = split_anomaly_records(records, seed=9)
    memberships = [
        sum(record.group_id == "shared-source" for record in split_records)
        for split_records in (split.train, split.validation, split.test)
    ]

    assert sorted(memberships) == [0, 0, 2]


def test_split_rejects_mixed_labels_within_source_group() -> None:
    records = [
        make_record("dataset", "machine", 0, label="normal", group_id="source"),
        make_record("dataset", "machine", 1, label="anomalous", group_id="source"),
    ]

    with pytest.raises(ValueError, match="mixed labels"):
        split_anomaly_records(records)


def test_mimii_adapter_maps_dataset_specific_fields() -> None:
    source = MIMIIRecord(
        path=Path("valve/id_02/abnormal/example.wav"),
        label="abnormal",
        machine_type="valve",
        machine_id="id_02",
        snr="6db",
    )

    adapted = to_anomaly_audio_record(source)

    assert adapted.dataset_name == "mimii"
    assert adapted.label == "anomalous"
    assert adapted.condition_id == "valve/id_02/6db"
    assert adapted.metadata_dict()["source_label"] == "abnormal"
