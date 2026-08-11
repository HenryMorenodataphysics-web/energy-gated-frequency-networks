from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import find_mimii_recordings, summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local MIMII dataset folder.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="all")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = find_mimii_recordings(
        args.data_dir,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        snr=args.snr,
    )
    summary = summarize_records(records)

    print(f"records={len(records)}")
    print("machine_type,machine_id,snr,label,count")
    for (machine_type, machine_id, snr, label), count in sorted(summary.items()):
        print(f"{machine_type},{machine_id},{snr},{label},{count}")


if __name__ == "__main__":
    main()
