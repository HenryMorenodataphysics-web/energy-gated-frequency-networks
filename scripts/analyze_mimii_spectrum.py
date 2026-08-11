from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mimii_dataset import MIMIIRecord, find_mimii_recordings
from src.utils.spectral_profile import analyze_wav_bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure frequency-band energy statistics in MIMII WAV files."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument("--n-fft", type=int, default=1_024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--num-bands", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "mimii_spectrum",
    )
    return parser.parse_args()


def select_balanced(records: list[MIMIIRecord], max_files: int | None) -> list[MIMIIRecord]:
    if max_files is None or max_files >= len(records):
        return records
    if max_files <= 0:
        raise ValueError("--max-files must be positive.")

    by_label: dict[str, list[MIMIIRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    selected: list[MIMIIRecord] = []
    labels = sorted(by_label)
    base, remainder = divmod(max_files, len(labels))
    for index, label in enumerate(labels):
        limit = base + (1 if index < remainder else 0)
        selected.extend(by_label[label][:limit])
    return sorted(selected, key=lambda record: str(record.path))


def aggregate_rows(
    rows: list[dict[str, object]],
    group_by_machine_id: bool = False,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key_parts: list[object] = []
        if group_by_machine_id:
            key_parts.append(str(row["machine_id"]))
        key_parts.extend(
            (
                str(row["label"]),
                int(row["band_index"]),
                float(row["low_hz"]),
                float(row["high_hz"]),
            )
        )
        key = tuple(key_parts)
        groups[key].append(row)

    summary: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        if group_by_machine_id:
            machine_id, label, band_index, low_hz, high_hz = key
        else:
            label, band_index, low_hz, high_hz = key
        energy_db = np.asarray([float(row["mean_power_db"]) for row in group])
        variance_db = np.asarray([float(row["variance_db"]) for row in group])
        delta_db = np.asarray([float(row["mean_abs_delta_db"]) for row in group])
        result: dict[str, object] = {}
        if group_by_machine_id:
            result["machine_id"] = machine_id
        result.update(
            {
                "label": label,
                "band_index": band_index,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "recordings": len(group),
                "mean_power_db": float(energy_db.mean()),
                "std_power_db": float(energy_db.std()),
                "mean_variance_db": float(variance_db.mean()),
                "mean_abs_delta_db": float(delta_db.mean()),
            }
        )
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = find_mimii_recordings(
        args.data_dir,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        snr=args.snr,
    )
    selected = select_balanced(records, args.max_files)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(selected, start=1):
        sample_rate, channels, duration, bands = analyze_wav_bands(
            record.path,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            num_bands=args.num_bands,
        )
        for band in bands:
            rows.append(
                {
                    "path": str(record.path),
                    "label": record.label,
                    "machine_type": record.machine_type,
                    "machine_id": record.machine_id,
                    "snr": record.snr,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_seconds": duration,
                    "band_index": band.band_index,
                    "low_hz": band.low_hz,
                    "high_hz": band.high_hz,
                    "mean_power": band.mean_power,
                    "mean_power_db": band.mean_power_db,
                    "variance_db": band.variance_db,
                    "mean_abs_delta_db": band.mean_abs_delta_db,
                }
            )
        if index % 100 == 0 or index == len(selected):
            print(f"processed={index}/{len(selected)}")

    summary = aggregate_rows(rows)
    summary_by_machine_id = aggregate_rows(rows, group_by_machine_id=True)
    write_csv(args.output_dir / "recording_band_statistics.csv", rows)
    write_csv(args.output_dir / "band_summary.csv", summary)
    write_csv(args.output_dir / "band_summary_by_machine_id.csv", summary_by_machine_id)
    metadata = {
        "data_dir": str(args.data_dir),
        "machine_type": args.machine_type,
        "machine_id": args.machine_id,
        "snr": args.snr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "num_bands": args.num_bands,
        "recordings": len(selected),
        "summary": summary,
        "summary_by_machine_id": summary_by_machine_id,
    }
    with (args.output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
