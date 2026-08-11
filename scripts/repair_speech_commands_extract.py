from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair a partial Speech Commands extraction without redownloading."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "speech_commands",
    )
    parser.add_argument("--archive-name", default="speech_commands_v0.02.tar.gz")
    parser.add_argument("--folder-name", default="SpeechCommands")
    parser.add_argument("--dataset-name", default="speech_commands_v0.02")
    return parser.parse_args()


def remove_readonly_and_retry(function, path, _exc_info) -> None:
    target = Path(path)
    try:
        target.chmod(0o777)
    except OSError:
        pass
    function(path)


def safe_member_path(destination: Path, member_name: str) -> Path:
    target = (destination / member_name).resolve()
    destination_resolved = destination.resolve()
    if destination_resolved != target and destination_resolved not in target.parents:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target


def extract_without_chmod(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        total = len(members)
        for index, member in enumerate(members, start=1):
            target = safe_member_path(destination, member.name)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

            if index % 5000 == 0 or index == total:
                print(f"extracted={index}/{total}")


def normalize_flat_extract(extract_root: Path, dataset_dir: Path) -> None:
    flat_validation_list = extract_root / "validation_list.txt"
    flat_testing_list = extract_root / "testing_list.txt"

    if not flat_validation_list.exists() or not flat_testing_list.exists():
        return

    print(f"normalizing_flat_extract={extract_root}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for child in list(extract_root.iterdir()):
        if child == dataset_dir:
            continue
        shutil.move(str(child), str(dataset_dir / child.name))


def main() -> None:
    args = parse_args()
    archive_path = args.data_dir / args.archive_name
    extract_root = args.data_dir / args.folder_name
    dataset_dir = extract_root / args.dataset_name

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. Run train_speech_commands.py --download first."
        )

    if dataset_dir.exists():
        print(f"removing_partial_extract={dataset_dir}")
        shutil.rmtree(dataset_dir, onerror=remove_readonly_and_retry)

    if not (extract_root / "validation_list.txt").exists():
        print(f"extracting_archive={archive_path}")
        extract_without_chmod(archive_path, dataset_dir)

    normalize_flat_extract(extract_root, dataset_dir)

    validation_list = dataset_dir / "validation_list.txt"
    testing_list = dataset_dir / "testing_list.txt"
    if not validation_list.exists() or not testing_list.exists():
        raise RuntimeError("Extraction finished, but split list files are missing.")

    wav_count = sum(1 for _ in dataset_dir.glob("*/*.wav"))
    print(f"dataset_dir={dataset_dir}")
    print(f"wav_count={wav_count}")
    print("repair_ok=True")


if __name__ == "__main__":
    main()
