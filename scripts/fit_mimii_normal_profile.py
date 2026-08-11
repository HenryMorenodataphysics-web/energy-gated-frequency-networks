from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import NormalProfileEstimator
from src.blocks import HierarchicalSpectralFrontend
from src.data import (
    find_mimii_recordings,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
)
from src.utils.spectral_profile import load_wav_preserve_level


def parse_subband_counts(value: str) -> tuple[int, int, int]:
    counts = tuple(int(item.strip()) for item in value.split(","))
    if len(counts) != 3 or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("expected three positive counts, for example 4,8,4")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a condition-aware normal profile from MIMII training audio."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument(
        "--macro-band-spec",
        type=Path,
        default=ROOT / "outputs" / "mimii_valve_train_24bands" / "macro_band_spec.json",
    )
    parser.add_argument("--policy", default="condition_balanced_energy")
    parser.add_argument("--subbands-per-macro", type=parse_subband_counts, default=(4, 8, 4))
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-normal-ratio", type=float, default=0.15)
    parser.add_argument("--test-normal-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-records-per-condition", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "mimii_normal_profile" / "normal_profile.json",
    )
    return parser.parse_args()


def load_macro_edges(path: Path, policy: str) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for candidate in payload.get("candidates", []):
        if candidate.get("policy") == policy:
            edges = [float(value) for value in candidate["edges_hz"]]
            if len(edges) != 4:
                raise ValueError("selected macro-band policy must define three bands.")
            return edges
    raise ValueError(f"policy {policy!r} was not found in {path}.")


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive.")

    records = [
        to_anomaly_audio_record(record)
        for record in find_mimii_recordings(
            args.data_dir,
            machine_type=args.machine_type,
            machine_id=args.machine_id,
            snr=args.snr,
        )
    ]
    split = split_anomaly_records(
        records,
        validation_normal_ratio=args.validation_normal_ratio,
        test_normal_ratio=args.test_normal_ratio,
        seed=args.seed,
    )
    validate_anomaly_split(split)
    train_records = list(split.train)
    if args.max_files is not None:
        train_records = train_records[: args.max_files]
    if not train_records:
        raise RuntimeError("the selected training split is empty.")

    first_audio, sample_rate = load_wav_preserve_level(train_records[0].path)
    device = select_device(args.device)
    macro_edges = load_macro_edges(args.macro_band_spec, args.policy)
    frontend = HierarchicalSpectralFrontend(
        sample_rate=sample_rate,
        macro_edges_hz=macro_edges,
        subbands_per_macro=args.subbands_per_macro,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    ).to(device)
    frontend.eval()
    estimator = NormalProfileEstimator(
        num_subbands=frontend.num_subbands,
        minimum_records_per_condition=args.minimum_records_per_condition,
    )

    expected_shape = first_audio.shape
    with torch.no_grad():
        for start in range(0, len(train_records), args.batch_size):
            batch_records = train_records[start : start + args.batch_size]
            waveforms = []
            for record in batch_records:
                audio, record_sample_rate = load_wav_preserve_level(record.path)
                if record_sample_rate != sample_rate or audio.shape != expected_shape:
                    raise ValueError(
                        "all recordings in one fitted profile must share sample rate and shape."
                    )
                waveforms.append(torch.from_numpy(audio.T.copy()))
            waveform = torch.stack(waveforms).to(device=device, dtype=torch.float32)
            subband_energy = frontend(waveform)["subband_energy"]
            condition_ids = [
                f"{record.dataset_name}/{record.condition_id}" for record in batch_records
            ]
            estimator.update(
                subband_energy,
                condition_ids,
                [record.label for record in batch_records],
            )
            processed = min(start + args.batch_size, len(train_records))
            if processed % 100 == 0 or processed == len(train_records):
                print(f"processed={processed}/{len(train_records)} device={device.type}")

    frontend_signature = {
        "sample_rate": sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "macro_edges_hz": macro_edges,
        "subbands_per_macro": list(args.subbands_per_macro),
        "subband_edges_hz": frontend.subband_edges_hz.detach().cpu().tolist(),
    }
    profile = estimator.finalize(
        metadata={
            "dataset": "mimii",
            "source_split": "train",
            "source_recordings": len(train_records),
            "machine_type": args.machine_type,
            "machine_id": args.machine_id,
            "snr": args.snr,
            "split_seed": args.seed,
            "validation_normal_ratio": args.validation_normal_ratio,
            "test_normal_ratio": args.test_normal_ratio,
            "macro_band_policy": args.policy,
            "macro_band_spec": str(args.macro_band_spec),
            "frontend": frontend_signature,
        }
    )
    profile.save_json(args.output)
    print(
        f"saved={args.output} conditions={len(profile.condition_ids)} "
        f"subbands={profile.num_subbands}"
    )


if __name__ == "__main__":
    main()
