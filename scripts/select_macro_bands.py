from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.macro_band_selection import (
    select_macro_band_candidates,
    validate_training_profile_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select reproducible macro-band candidates from normal training audio."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory produced by analyze_mimii_spectrum.py.",
    )
    parser.add_argument("--num-macro-bands", type=int, default=3)
    parser.add_argument("--min-fine-bands", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_analysis(
    analysis_dir: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    metadata_path = analysis_dir / "analysis.json"
    statistics_path = analysis_dir / "recording_band_statistics.csv"
    if not metadata_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError(
            "analysis directory must contain analysis.json and "
            "recording_band_statistics.csv."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with statistics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("recording_band_statistics.csv is empty.")
    return metadata, rows


def build_power_matrix(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    band_definitions = sorted(
        {
            (int(row["band_index"]), float(row["low_hz"]), float(row["high_hz"]))
            for row in rows
        }
    )
    expected_indices = list(range(len(band_definitions)))
    if [definition[0] for definition in band_definitions] != expected_indices:
        raise ValueError("fine-band indices must be contiguous and start at zero.")

    fine_edges = np.asarray(
        [band_definitions[0][1], *(definition[2] for definition in band_definitions)],
        dtype=np.float64,
    )
    by_path: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        by_path.setdefault(row["path"], {})[int(row["band_index"])] = row

    powers = []
    condition_ids = []
    labels = []
    for path in sorted(by_path):
        band_rows = by_path[path]
        if sorted(band_rows) != expected_indices:
            raise ValueError(f"recording {path!r} does not contain every fine band.")
        first = band_rows[0]
        powers.append([float(band_rows[index]["mean_power"]) for index in expected_indices])
        condition_ids.append(
            "/".join(
                (
                    first.get("machine_type", "unknown"),
                    first.get("machine_id", "unknown"),
                    first.get("snr", "unknown"),
                )
            )
        )
        labels.append(first["label"])
    return fine_edges, np.asarray(powers), condition_ids, labels


def main() -> None:
    args = parse_args()
    metadata, rows = load_analysis(args.analysis_dir)
    fine_edges, powers, condition_ids, labels = build_power_matrix(rows)
    validate_training_profile_source(str(metadata.get("split", "unknown")), labels)
    candidates = select_macro_band_candidates(
        fine_edges,
        powers,
        condition_ids,
        num_macro_bands=args.num_macro_bands,
        min_fine_bands=args.min_fine_bands,
    )

    output_path = args.output or args.analysis_dir / "macro_band_spec.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_analysis": str(args.analysis_dir),
        "source_split": metadata["split"],
        "source_recordings": len(powers),
        "source_conditions": sorted(set(condition_ids)),
        "fine_band_count": powers.shape[1],
        "num_macro_bands": args.num_macro_bands,
        "min_fine_bands": args.min_fine_bands,
        "candidates": [
            {
                "policy": candidate.policy,
                "edges_hz": candidate.edges_hz,
                "edges_fraction": candidate.edges_fraction,
                "energy_fraction": candidate.energy_fraction,
            }
            for candidate in candidates
        ],
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for candidate in candidates:
        print(
            f"policy={candidate.policy} edges_hz={candidate.edges_hz} "
            f"energy_fraction={candidate.energy_fraction}"
        )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
