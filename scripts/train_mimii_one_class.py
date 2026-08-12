from __future__ import annotations

import argparse
import csv
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

from src.anomaly import (
    ConditionedNormalProfile,
    anti_collapse_loss,
    standardized_embedding_scores,
)
from src.blocks import HierarchicalSpectralFrontend
from src.data import (
    AnomalyAudioRecord,
    AnomalyWindowDataset,
    find_mimii_recordings,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
)
from src.models import Conv1DAnomalyEncoder, HierarchicalAnomalyDetector
from src.utils.binary_evaluation import metrics_at_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EGFN or Conv1D with the same one-class MIMII protocol."
    )
    parser.add_argument("--model", choices=("egfn", "conv1d"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument(
        "--normal-profile",
        type=Path,
        default=ROOT / "outputs" / "mimii_normal_profile" / "normal_profile.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.1)
    parser.add_argument("--variance-target", type=float, default=0.05)
    parser.add_argument("--invariance-weight", type=float, default=25.0)
    parser.add_argument("--variance-weight", type=float, default=25.0)
    parser.add_argument("--covariance-weight", type=float, default=1.0)
    parser.add_argument("--view-noise-fraction", type=float, default=0.005)
    parser.add_argument("--view-max-shift-fraction", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--recording-quantile", type=float, default=0.95)
    parser.add_argument("--normal-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--validation-normal-ratio", type=float, default=0.15)
    parser.add_argument("--test-normal-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--early-stopping-min-relative-improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records-per-label", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def cap_by_label(
    records: tuple[AnomalyAudioRecord, ...],
    limit: int | None,
) -> tuple[AnomalyAudioRecord, ...]:
    if limit is None:
        return records
    if limit <= 0:
        raise ValueError("record limits must be positive.")
    selected: list[AnomalyAudioRecord] = []
    by_label: dict[str, list[AnomalyAudioRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    for label in sorted(by_label):
        selected.extend(by_label[label][:limit])
    return tuple(selected)


def build_model(
    model_name: str,
    profile: ConditionedNormalProfile,
) -> torch.nn.Module:
    if model_name == "conv1d":
        return Conv1DAnomalyEncoder(embedding_channels=8)
    signature = profile.metadata.get("frontend")
    if not isinstance(signature, dict):
        raise ValueError("normal profile does not contain a frontend signature.")
    frontend = HierarchicalSpectralFrontend(
        sample_rate=int(signature["sample_rate"]),
        macro_edges_hz=signature["macro_edges_hz"],
        subbands_per_macro=signature["subbands_per_macro"],
        n_fft=int(signature["n_fft"]),
        hop_length=int(signature["hop_length"]),
    )
    return HierarchicalAnomalyDetector(frontend, profile, embedding_channels=8)


def forward_model(
    model: torch.nn.Module,
    model_name: str,
    waveform: torch.Tensor,
    condition_ids: list[str],
    progress: float,
) -> dict[str, torch.Tensor]:
    if model_name == "egfn":
        return model(waveform, condition_ids, regularization_progress=progress)
    return model(waveform)


def make_audio_view(
    waveform: torch.Tensor,
    noise_fraction: float,
    max_shift_fraction: float,
) -> torch.Tensor:
    if noise_fraction < 0 or not 0 <= max_shift_fraction < 1:
        raise ValueError("view augmentation values are outside their valid ranges.")
    maximum_shift = int(waveform.shape[-1] * max_shift_fraction)
    shifts = torch.randint(
        -maximum_shift,
        maximum_shift + 1,
        (waveform.shape[0],),
        device=waveform.device,
    )
    shifted = torch.stack(
        [torch.roll(item, int(shift.item()), dims=-1) for item, shift in zip(waveform, shifts)]
    )
    if noise_fraction == 0:
        return shifted
    rms = shifted.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return shifted + torch.randn_like(shifted) * rms * noise_fraction


def estimate_embedding_statistics(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            output = forward_model(
                model,
                model_name,
                waveform,
                list(batch["condition_id"]),
                progress=0.0,
            )
            embeddings.append(output["embedding"].cpu())
    if not embeddings:
        raise RuntimeError("cannot estimate statistics from an empty loader.")
    stacked = torch.cat(embeddings)
    return stacked.mean(dim=0).to(device), stacked.std(dim=0).clamp_min(1e-3).to(device)


def run_training_epoch(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress: float,
    gate_weight: float,
    variance_target: float,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
    noise_fraction: float,
    max_shift_fraction: float,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    example_count = 0
    for batch in loader:
        waveform = batch["waveform"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        first_view = make_audio_view(waveform, noise_fraction, max_shift_fraction)
        second_view = make_audio_view(waveform, noise_fraction, max_shift_fraction)
        first_output = forward_model(
            model,
            model_name,
            first_view,
            list(batch["condition_id"]),
            progress,
        )
        second_output = forward_model(
            model,
            model_name,
            second_view,
            list(batch["condition_id"]),
            progress,
        )
        objective = anti_collapse_loss(
            first_output["embedding"],
            second_output["embedding"],
            variance_target=variance_target,
            invariance_weight=invariance_weight,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )
        first_gate_loss = first_output.get(
            "gate_regularization_loss", objective["representation_loss"].new_zeros(())
        )
        second_gate_loss = second_output.get(
            "gate_regularization_loss", first_gate_loss
        )
        gate_loss = 0.5 * (first_gate_loss + second_gate_loss)
        loss = objective["representation_loss"] + gate_weight * gate_loss
        loss.backward()
        optimizer.step()

        batch_size = waveform.shape[0]
        totals["loss"] += loss.item() * batch_size
        totals["representation_loss"] += objective["representation_loss"].item() * batch_size
        totals["invariance_loss"] += objective["invariance_loss"].item() * batch_size
        totals["variance_loss"] += objective["variance_loss"].item() * batch_size
        totals["covariance_loss"] += objective["covariance_loss"].item() * batch_size
        totals["gate_loss"] += gate_loss.item() * batch_size
        totals["embedding_std"] += objective["embedding_std"].item() * batch_size
        example_count += batch_size
    return {name: value / example_count for name, value in totals.items()}


def collect_window_scores(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    normal_mean: torch.Tensor,
    normal_std: torch.Tensor,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    recording_ids: list[str] = []
    labels: list[int] = []
    learned_scores: list[float] = []
    profile_scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            output = forward_model(
                model,
                model_name,
                waveform,
                list(batch["condition_id"]),
                progress=1.0,
            )
            recording_ids.extend(batch["recording_id"])
            labels.extend(batch["label"].tolist())
            learned_scores.extend(
                standardized_embedding_scores(
                    output["embedding"], normal_mean, normal_std
                ).cpu().tolist()
            )
            if "recording_score" in output:
                profile_scores.extend(output["recording_score"].cpu().tolist())
    optional_profile = np.asarray(profile_scores) if profile_scores else None
    return recording_ids, np.asarray(labels), np.asarray(learned_scores), optional_profile


def aggregate_recordings(
    recording_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for recording_id, label, score in zip(recording_ids, labels, scores, strict=True):
        label = int(label)
        if recording_id in grouped_labels and grouped_labels[recording_id] != label:
            raise ValueError("one recording contains conflicting labels.")
        grouped_labels[recording_id] = label
        grouped_scores[recording_id].append(float(score))
    ordered_ids = sorted(grouped_scores)
    return (
        np.asarray([grouped_labels[recording_id] for recording_id in ordered_ids]),
        np.asarray(
            [np.quantile(grouped_scores[recording_id], quantile) for recording_id in ordered_ids]
        ),
    )


def evaluate_normal_mean(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    normal_mean: torch.Tensor,
    normal_std: torch.Tensor,
    device: torch.device,
    recording_quantile: float,
) -> float:
    ids, labels, scores, _ = collect_window_scores(
        model, model_name, loader, normal_mean, normal_std, device
    )
    record_labels, record_scores = aggregate_recordings(ids, labels, scores, recording_quantile)
    if np.any(record_labels != 0):
        raise ValueError("validation must contain only normal recordings.")
    return float(record_scores.mean())


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def is_meaningful_improvement(
    current: float,
    reference: float,
    minimum_relative_improvement: float,
) -> bool:
    if not 0 <= minimum_relative_improvement < 1:
        raise ValueError("minimum_relative_improvement must be in [0, 1).")
    return not np.isfinite(reference) or current < reference * (
        1.0 - minimum_relative_improvement
    )


def save_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary_path.replace(path)


def save_recovery_checkpoint(
    path: Path,
    model_state: dict[str, torch.Tensor],
    normal_mean: torch.Tensor,
    normal_std: torch.Tensor,
    epoch: int,
    validation_score: float,
    args: argparse.Namespace,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state": model_state,
            "normal_mean": normal_mean.detach().cpu(),
            "normal_std": normal_std.detach().cpu(),
            "epoch": int(epoch),
            "validation_normal_score": float(validation_score),
            "args": serializable_args(args),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size < 2:
        raise ValueError("epochs must be positive and batch size must be at least two.")
    for name in ("recording_quantile", "normal_threshold_quantile"):
        if not 0 < getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1).")
    if not 0 <= args.early_stopping_min_relative_improvement < 1:
        raise ValueError(
            "--early-stopping-min-relative-improvement must be in [0, 1)."
        )
    if args.variance_target <= 0 or min(
        args.invariance_weight,
        args.variance_weight,
        args.covariance_weight,
    ) < 0:
        raise ValueError("representation objective values are invalid.")
    if args.view_noise_fraction < 0 or not 0 <= args.view_max_shift_fraction < 1:
        raise ValueError("view augmentation values are invalid.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    output_dir = args.output_dir or (
        ROOT / "outputs" / "mimii_one_class" / f"{args.model}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_checkpoint_path = args.checkpoint or output_dir / "best_checkpoint.pt"
    recovery_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
        f"train_normal={len(train_records)} validation_normal={len(validation_records)} "
        f"test_records={len(test_records)}"
    )

    profile = ConditionedNormalProfile.load_json(args.normal_profile)
    sample_rate = int(profile.metadata["frontend"]["sample_rate"])
    train_set = AnomalyWindowDataset(
        train_records,
        target_sample_rate=sample_rate,
        duration_seconds=args.duration,
        crop_mode="random",
    )
    validation_set = AnomalyWindowDataset(
        validation_records,
        target_sample_rate=sample_rate,
        duration_seconds=args.duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    test_set = AnomalyWindowDataset(
        test_records,
        target_sample_rate=sample_rate,
        duration_seconds=args.duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    center_loader = DataLoader(train_set, batch_size=args.batch_size, **loader_options)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        **loader_options,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_options)

    model = build_model(args.model, profile).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"trainable_parameters={trainable_parameters}")
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    if args.evaluate_only:
        if not recovery_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"recovery checkpoint not found: {recovery_checkpoint_path}"
            )
        checkpoint = torch.load(
            recovery_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("model") != args.model:
            raise ValueError("checkpoint model does not match --model.")
        if int(checkpoint_args.get("seed", -1)) != args.seed:
            raise ValueError("checkpoint seed does not match --seed.")
        if "variance_target" not in checkpoint_args:
            raise ValueError("checkpoint was not trained with the anti-collapse objective.")
        best_state = checkpoint["model_state"]
        normal_mean = checkpoint["normal_mean"].to(device)
        normal_std = checkpoint["normal_std"].to(device)
        print(
            f"loaded_checkpoint={recovery_checkpoint_path} "
            f"epoch={checkpoint.get('epoch', 'unknown')}"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_validation_score = float("inf")
        early_stopping_reference = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            progress = (epoch - 1) / max(args.epochs - 1, 1)
            train_metrics = run_training_epoch(
                model,
                args.model,
                train_loader,
                optimizer,
                device,
                progress,
                args.gate_regularization_weight,
                args.variance_target,
                args.invariance_weight,
                args.variance_weight,
                args.covariance_weight,
                args.view_noise_fraction,
                args.view_max_shift_fraction,
            )
            normal_mean, normal_std = estimate_embedding_statistics(
                model, args.model, center_loader, device
            )
            validation_score = evaluate_normal_mean(
                model,
                args.model,
                validation_loader,
                normal_mean,
                normal_std,
                device,
                args.recording_quantile,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{name}": value for name, value in train_metrics.items()},
                "validation_normal_score": validation_score,
            }
            history.append(row)
            save_history(history, output_dir / "history.csv")
            print(
                f"epoch={epoch:02d} loss={train_metrics['loss']:.6f} "
                f"embedding_std={train_metrics['embedding_std']:.6f} "
                f"val_normal_score={validation_score:.6f}"
            )
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                save_recovery_checkpoint(
                    recovery_checkpoint_path,
                    best_state,
                    normal_mean,
                    normal_std,
                    epoch,
                    validation_score,
                    args,
                )
            if is_meaningful_improvement(
                validation_score,
                early_stopping_reference,
                args.early_stopping_min_relative_improvement,
            ):
                early_stopping_reference = validation_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"early_stopping_epoch={epoch} "
                    f"min_relative_improvement="
                    f"{args.early_stopping_min_relative_improvement:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint.")
    if not args.evaluate_only:
        best_checkpoint = torch.load(
            recovery_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        normal_mean = best_checkpoint["normal_mean"].to(device)
        normal_std = best_checkpoint["normal_std"].to(device)
    model.load_state_dict(best_state)
    model.to(device)

    (
        validation_ids,
        validation_labels,
        validation_scores,
        validation_profile_window_scores,
    ) = collect_window_scores(
        model, args.model, validation_loader, normal_mean, normal_std, device
    )
    validation_labels, validation_scores = aggregate_recordings(
        validation_ids,
        validation_labels,
        validation_scores,
        args.recording_quantile,
    )
    threshold = float(np.quantile(validation_scores, args.normal_threshold_quantile))
    test_ids, test_labels, test_scores, profile_window_scores = collect_window_scores(
        model, args.model, test_loader, normal_mean, normal_std, device
    )
    test_labels, test_scores = aggregate_recordings(
        test_ids,
        test_labels,
        test_scores,
        args.recording_quantile,
    )
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    result: dict[str, object] = {
        "model": args.model,
        "protocol": "normal_only_anti_collapse_representation",
        "trainable_parameters": trainable_parameters,
        "normal_mean": normal_mean.detach().cpu().tolist(),
        "normal_std": normal_std.detach().cpu().tolist(),
        "threshold_source": f"validation_normal_quantile_{args.normal_threshold_quantile}",
        "metrics": metrics,
        "split_counts": {
            "train_normal": len(train_records),
            "validation_normal": len(validation_records),
            "test_normal": int((test_labels == 0).sum()),
            "test_anomalous": int((test_labels == 1).sum()),
        },
        "args": serializable_args(args),
    }
    if profile_window_scores is not None:
        if validation_profile_window_scores is None:
            raise RuntimeError("validation profile scores are missing.")
        _, validation_profile_scores = aggregate_recordings(
            validation_ids,
            np.zeros(len(validation_profile_window_scores), dtype=np.int64),
            validation_profile_window_scores,
            args.recording_quantile,
        )
        _, profile_scores = aggregate_recordings(
            test_ids,
            np.asarray([0] * len(profile_window_scores)),
            profile_window_scores,
            args.recording_quantile,
        )
        result["profile_score_auc"] = metrics_at_threshold(
            test_labels,
            profile_scores,
            float(
                np.quantile(
                    validation_profile_scores,
                    args.normal_threshold_quantile,
                )
            ),
        )["auc"]

    save_history(history, output_dir / "history.csv")
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state": best_state,
            "normal_mean": normal_mean.detach().cpu(),
            "normal_std": normal_std.detach().cpu(),
            "threshold": threshold,
            "args": serializable_args(args),
        },
        output_dir / "checkpoint.pt",
    )
    print(
        f"test_auc={metrics['auc']:.4f} test_f1={metrics['f1']:.4f} "
        f"threshold={threshold:.6f}"
    )
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
