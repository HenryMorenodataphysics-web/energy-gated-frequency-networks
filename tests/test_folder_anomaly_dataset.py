from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.data import find_folder_anomaly_recordings, split_anomaly_records
from scripts.train_mimii_one_class import bootstrap_normal_profile, parse_args


def write_audio(path: Path, sample_rate: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)


def test_discovers_normal_only_audio_with_global_condition(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    write_audio(normal_dir / "machine_a" / "one.wav")
    write_audio(normal_dir / "machine_b" / "two.flac")
    (normal_dir / "ignored.txt").write_text("not audio", encoding="utf-8")

    records = find_folder_anomaly_recordings(normal_dir, dataset_name="factory")

    assert len(records) == 2
    assert {record.label for record in records} == {"normal"}
    assert {record.condition_id for record in records} == {"global"}
    assert {record.dataset_name for record in records} == {"factory"}


def test_parent_conditions_and_anomalies_do_not_leak_into_fitting(
    tmp_path: Path,
) -> None:
    normal_dir = tmp_path / "normal"
    anomalous_dir = tmp_path / "anomalous"
    for index in range(4):
        write_audio(normal_dir / "fan" / f"normal_{index}.wav")
    write_audio(anomalous_dir / "fan" / "impact.wav")

    records = find_folder_anomaly_recordings(
        normal_dir,
        anomalous_dir,
        dataset_name="factory",
        condition_mode="parent",
    )
    split = split_anomaly_records(records, seed=42)

    assert {record.condition_id for record in records} == {"fan"}
    assert all(record.label == "normal" for record in split.train)
    assert all(record.label == "normal" for record in split.validation)
    assert sum(record.label == "anomalous" for record in split.test) == 1
    assert not (
        {record.group_id for record in split.train}
        & {record.group_id for record in split.test}
    )


def test_generic_cli_and_bootstrap_profile_share_frontend_configuration(
    tmp_path: Path,
) -> None:
    normal_dir = tmp_path / "normal"
    for index in range(3):
        write_audio(normal_dir / f"normal_{index}.wav")
    records = find_folder_anomaly_recordings(normal_dir, dataset_name="factory")
    args = parse_args(
        [
            "--dataset-format",
            "folders",
            "--model",
            "egfn",
            "--normal-dir",
            str(normal_dir),
            "--sample-rate",
            "8000",
            "--subbands-per-macro",
            "2,3,2",
        ]
    )

    profile = bootstrap_normal_profile(
        tuple(records),
        args.sample_rate,
        args.n_fft,
        args.hop_length,
        args.macro_edges_hz,
        args.subbands_per_macro,
    )

    assert profile.condition_ids == ("factory/global",)
    assert profile.num_subbands == 7
    assert profile.record_counts.equal(torch.tensor([3]))
    assert profile.metadata["frontend"]["sample_rate"] == 8_000
