from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


FSDD_ZIP_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Free Spoken Digit Dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/fsdd"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / "free-spoken-digit-dataset-master.zip"

    if not zip_path.exists():
        print(f"Downloading {FSDD_ZIP_URL}")
        urlretrieve(FSDD_ZIP_URL, zip_path)
    else:
        print(f"Found existing archive: {zip_path}")

    extract_dir = args.output_dir / "free-spoken-digit-dataset-master"
    if not extract_dir.exists():
        print(f"Extracting to {args.output_dir}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(args.output_dir)
    else:
        print(f"Found existing extracted dataset: {extract_dir}")

    recordings = list(extract_dir.joinpath("recordings").glob("*.wav"))
    print(f"recordings={len(recordings)}")
    print(f"dataset_dir={extract_dir}")


if __name__ == "__main__":
    main()
