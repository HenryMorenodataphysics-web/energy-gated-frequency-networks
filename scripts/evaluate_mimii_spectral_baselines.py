from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mimii_one_class import (
    aggregate_score_outputs,
    apply_condition_thresholds,
    cap_by_label,
    fit_condition_thresholds,
    write_recording_scores,
)
from src.anomaly import ConditionedSpectralBaselineEstimator, LogMelFrontend
from src.data import (
    AnomalyWindowDataset,
    find_mimii_recordings,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
)
from src.utils.binary_evaluation import metrics_at_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate normal-only log-mel sanity baselines through the MIMII adapter."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--pca-rank", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=2_048)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--recording-quantile", type=float, default=0.95)
    parser.add_argument("--normal-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--validation-normal-ratio", type=float, default=0.15)
    parser.add_argument("--test-normal-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records-per-label", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "mimii_spectral_baselines_v5",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def collect_scores(
    frontend: LogMelFrontend,
    baselines,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    recording_ids: list[str] = []
    condition_ids: list[str] = []
    labels: list[int] = []
    scores: dict[str, list[float]] = defaultdict(list)
    frontend.eval()
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            conditions = list(batch["condition_id"])
            output = baselines.score(frontend(waveform), conditions)
            recording_ids.extend(batch["recording_id"])
            condition_ids.extend(conditions)
            labels.extend(batch["label"].tolist())
            for name, values in output.items():
                if name.endswith("_score"):
                    scores[name].extend(values.cpu().tolist())
    return {
        "recording_ids": recording_ids,
        "condition_ids": condition_ids,
        "labels": np.asarray(labels),
        "scores": {name: np.asarray(values) for name, values in scores.items()},
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.sample_rate <= 0 or args.duration <= 0:
        raise ValueError("batch size, sample rate, and duration must be positive.")
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

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
    train_records = split.train[: args.max_train_records]
    validation_records = cap_by_label(split.validation, args.max_eval_records_per_label)
    test_records = cap_by_label(split.test, args.max_eval_records_per_label)
    print(
        f"device={device} train_normal={len(train_records)} "
        f"validation_normal={len(validation_records)} test_records={len(test_records)}"
    )

    def make_dataset(selected):
        return AnomalyWindowDataset(
            selected,
            target_sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            crop_mode="grid",
            evaluation_windows=args.evaluation_windows,
        )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(make_dataset(train_records), **loader_options)
    validation_loader = DataLoader(make_dataset(validation_records), **loader_options)
    test_loader = DataLoader(make_dataset(test_records), **loader_options)

    frontend = LogMelFrontend(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    ).to(device)
    estimator = ConditionedSpectralBaselineEstimator(
        n_mels=args.n_mels,
        memory_size=args.memory_size,
        pca_rank=args.pca_rank,
        top_fraction=args.top_fraction,
        seed=args.seed,
    )
    with torch.no_grad():
        for batch in train_loader:
            if torch.any(batch["label"] != 0):
                raise ValueError("spectral baselines may only fit normal training audio.")
            waveform = batch["waveform"].to(device, non_blocking=True)
            estimator.update(frontend(waveform), list(batch["condition_id"]))
    baselines = estimator.finalize().to(device)

    validation = aggregate_score_outputs(
        collect_scores(frontend, baselines, validation_loader, device),
        args.recording_quantile,
    )
    test = aggregate_score_outputs(
        collect_scores(frontend, baselines, test_loader, device),
        args.recording_quantile,
    )
    raw_score_names = tuple(sorted(test["scores"]))
    metrics = {}
    thresholds = {}
    for score_name in raw_score_names:
        condition_thresholds, fallback = fit_condition_thresholds(
            validation,
            score_name,
            args.normal_threshold_quantile,
        )
        calibrated_name = f"{score_name}_calibrated"
        test["scores"][calibrated_name] = apply_condition_thresholds(
            test,
            score_name,
            condition_thresholds,
            fallback,
        )
        metrics[score_name] = metrics_at_threshold(
            test["labels"], test["scores"][calibrated_name], 1.0
        )
        thresholds[score_name] = {
            "by_condition": condition_thresholds,
            "fallback": fallback,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_recording_scores(test, args.output_dir / "test_recording_scores.csv")
    result = {
        "protocol": "normal_only_spectral_sanity_v5",
        "fit_labels": ["normal"],
        "metrics": metrics,
        "thresholds": thresholds,
        "split_counts": {
            "train_normal": len(train_records),
            "validation_normal": len(validation_records),
            "test_normal": int((test["labels"] == 0).sum()),
            "test_anomalous": int((test["labels"] == 1).sum()),
        },
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    for score_name, values in metrics.items():
        print(
            f"{score_name} auc={values['auc']:.4f} "
            f"f1={values['f1']:.4f} recall={values['recall']:.4f}"
        )
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
