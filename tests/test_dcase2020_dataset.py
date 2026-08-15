from pathlib import Path

import numpy as np
import soundfile as sf

from src.data import find_dcase2020_development_split


def write_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(800, dtype=np.float32), 8_000)


def test_preserves_official_test_and_holds_validation_from_train(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fan"
    for index in range(4):
        write_audio(root / "train" / f"normal_id_00_{index:08d}.wav")
    write_audio(root / "test" / "normal_id_00_00000000.wav")
    write_audio(root / "test" / "anomaly_id_00_00000000.wav")

    split = find_dcase2020_development_split(root, seed=42)

    assert len(split.train) == 3
    assert len(split.validation) == 1
    assert len(split.test) == 2
    assert all(record.is_normal for record in split.train + split.validation)
    assert {record.label for record in split.test} == {"normal", "anomalous"}
    assert {record.condition_id for record in split.test} == {"fan/id_00"}
    assert all(
        record.metadata_dict()["official_partition"] == "test"
        for record in split.test
    )
