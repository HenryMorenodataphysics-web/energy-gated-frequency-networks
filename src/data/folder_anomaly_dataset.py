from __future__ import annotations

from pathlib import Path
from typing import Literal

from .anomaly_protocol import AnomalyAudioRecord


ConditionMode = Literal["global", "parent"]
SUPPORTED_AUDIO_SUFFIXES = (".aif", ".aiff", ".flac", ".ogg", ".wav")


def _condition_from_relative_path(relative_path: Path, mode: ConditionMode) -> str:
    if mode == "global" or relative_path.parent == Path("."):
        return "global"
    return relative_path.parent.as_posix()


def _records_from_directory(
    directory: Path,
    label: Literal["normal", "anomalous"],
    dataset_name: str,
    condition_mode: ConditionMode,
) -> list[AnomalyAudioRecord]:
    records: list[AnomalyAudioRecord] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        relative_path = path.relative_to(directory)
        condition_id = _condition_from_relative_path(relative_path, condition_mode)
        condition_parts = Path(condition_id).parts
        records.append(
            AnomalyAudioRecord(
                path=path,
                dataset_name=dataset_name,
                machine_type=(
                    condition_parts[0] if condition_id != "global" else "generic"
                ),
                machine_id=condition_id,
                condition_id=condition_id,
                group_id=f"{label}/{relative_path.as_posix()}",
                label=label,
                metadata=(("source_relative_path", relative_path.as_posix()),),
            )
        )
    return records


def find_folder_anomaly_recordings(
    normal_dir: str | Path,
    anomalous_dir: str | Path | None = None,
    dataset_name: str = "folder_audio",
    condition_mode: ConditionMode = "global",
) -> list[AnomalyAudioRecord]:
    """Discover a normal-only or normal/anomalous folder dataset."""
    normal_path = Path(normal_dir)
    if not normal_path.is_dir():
        raise FileNotFoundError(f"normal audio directory not found: {normal_path}")
    if not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if condition_mode not in {"global", "parent"}:
        raise ValueError("condition_mode must be global or parent.")

    records = _records_from_directory(
        normal_path, "normal", dataset_name.strip(), condition_mode
    )
    if not records:
        suffixes = ", ".join(SUPPORTED_AUDIO_SUFFIXES)
        raise FileNotFoundError(
            f"no supported audio files found under {normal_path}; expected {suffixes}."
        )

    if anomalous_dir is not None:
        anomalous_path = Path(anomalous_dir)
        if not anomalous_path.is_dir():
            raise FileNotFoundError(
                f"anomalous audio directory not found: {anomalous_path}"
            )
        anomalous_records = _records_from_directory(
            anomalous_path, "anomalous", dataset_name.strip(), condition_mode
        )
        if not anomalous_records:
            raise FileNotFoundError(
                f"no supported audio files found under {anomalous_path}."
            )
        records.extend(anomalous_records)

    return sorted(records, key=lambda record: (record.label, record.group_id))
