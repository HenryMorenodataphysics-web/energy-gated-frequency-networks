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
    ConditionedEmbeddingEstimator,
    ConditionedEmbeddingProfile,
    ConditionedFeatureMemory,
    ConditionedFeatureMemoryEstimator,
    ConditionedNormalProfile,
    NormalProfileEstimator,
    anti_collapse_loss,
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
    parser.add_argument("--learnable-subband-weights", action="store_true")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
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
    parser.add_argument("--memory-size", type=int, default=512)
    parser.add_argument("--memory-temporal-pool", type=int, default=4)
    parser.add_argument("--memory-top-fraction", type=float, default=0.05)
    parser.add_argument("--memory-query-chunk-size", type=int, default=2048)
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
    learnable_subband_weights: bool = False,
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
        learnable_subband_weights=learnable_subband_weights,
    )
    return HierarchicalAnomalyDetector(frontend, profile, embedding_channels=8)


def forward_model(
    model: torch.nn.Module,
    model_name: str,
    waveform: torch.Tensor,
    condition_ids: list[str],
    progress: float,
    masked_subbands: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if model_name == "egfn":
        return model(
            waveform,
            condition_ids,
            regularization_progress=progress,
            masked_subbands=masked_subbands,
        )
    return model(waveform)


def random_subband_mask(
    batch_size: int,
    num_subbands: int,
    mask_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0 or num_subbands <= 1 or not 0 < mask_fraction < 1:
        raise ValueError("masked reconstruction configuration is invalid.")
    mask_count = min(num_subbands - 1, max(1, round(num_subbands * mask_fraction)))
    priorities = torch.rand(batch_size, num_subbands, device=device)
    indices = priorities.topk(mask_count, dim=1).indices
    return torch.zeros(
        batch_size, num_subbands, dtype=torch.bool, device=device
    ).scatter_(1, indices, True)


def deterministic_subband_mask(
    batch_size: int,
    num_subbands: int,
    mask_fraction: float,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    mask_count = min(num_subbands - 1, max(1, round(num_subbands * mask_fraction)))
    positions = torch.arange(mask_count, device=device)
    starts = torch.arange(batch_size, device=device) + offset
    indices = (starts[:, None] + positions[None, :]) % num_subbands
    return torch.zeros(
        batch_size, num_subbands, dtype=torch.bool, device=device
    ).scatter_(1, indices, True)


def masked_reconstruction_loss(
    output: dict[str, torch.Tensor],
    masked_subbands: torch.Tensor | None,
) -> torch.Tensor:
    if masked_subbands is None or "reconstructed_log_energy_z" not in output:
        return output["embedding"].new_zeros(())
    prediction = output["reconstructed_log_energy_z"]
    target = output["z_scores"][:, :, 0, :].detach().clamp(-10.0, 10.0)
    mask = masked_subbands.unsqueeze(-1).expand_as(target)
    return torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask])


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


def estimate_embedding_profiles(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, ConditionedEmbeddingProfile]:
    model.eval()
    estimators = {
        name: ConditionedEmbeddingEstimator()
        for name in ("sustained_embedding", "event_embedding", "subband_embedding")
    }
    observed = set()
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
            for name, estimator in estimators.items():
                if name in output:
                    estimator.update(output[name], list(batch["condition_id"]))
                    observed.add(name)
    if not observed:
        raise RuntimeError("model did not expose event-aware embedding components.")
    return {
        name: estimators[name].finalize().to(device)
        for name in sorted(observed)
    }


def estimate_spectral_profile(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metadata: dict[str, object],
) -> ConditionedNormalProfile:
    if not hasattr(model, "frontend"):
        raise ValueError("spectral profile fitting requires an EGFN frontend.")
    estimator = NormalProfileEstimator(num_subbands=model.frontend.num_subbands)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            frontend_output = model.frontend(waveform)
            conditions = list(batch["condition_id"])
            estimator.update(
                frontend_output["subband_energy"],
                conditions,
                ["normal"] * waveform.shape[0],
            )
    return estimator.finalize(metadata=metadata).to(device)


def local_feature_map(output: dict[str, torch.Tensor]) -> torch.Tensor:
    if "embedding_map" in output:
        return output["embedding_map"]
    if "feature_map" in output:
        feature_map = output["feature_map"]
        if feature_map.ndim == 3:
            return feature_map.unsqueeze(2)
    raise ValueError("model did not expose a local feature map.")


def estimate_feature_memory(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    max_vectors_per_condition: int,
    temporal_pool: int,
    top_fraction: float,
    query_chunk_size: int,
    seed: int,
) -> ConditionedFeatureMemory:
    estimator = ConditionedFeatureMemoryEstimator(
        max_vectors_per_condition=max_vectors_per_condition,
        temporal_pool=temporal_pool,
        top_fraction=top_fraction,
        query_chunk_size=query_chunk_size,
        seed=seed,
    )
    model.eval()
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
            estimator.update(local_feature_map(output), list(batch["condition_id"]))
    return estimator.finalize().to(device)


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
    reconstruction_weight: float,
    mask_fraction: float,
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
        if model_name == "egfn" and reconstruction_weight > 0:
            num_subbands = model.frontend.num_subbands
            first_mask = random_subband_mask(
                waveform.shape[0], num_subbands, mask_fraction, device
            )
            second_mask = random_subband_mask(
                waveform.shape[0], num_subbands, mask_fraction, device
            )
        else:
            first_mask = None
            second_mask = None
        first_output = forward_model(
            model,
            model_name,
            first_view,
            list(batch["condition_id"]),
            progress,
            masked_subbands=first_mask,
        )
        second_output = forward_model(
            model,
            model_name,
            second_view,
            list(batch["condition_id"]),
            progress,
            masked_subbands=second_mask,
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
        reconstruction_loss = 0.5 * (
            masked_reconstruction_loss(first_output, first_mask)
            + masked_reconstruction_loss(second_output, second_mask)
        )
        loss = (
            objective["representation_loss"]
            + gate_weight * gate_loss
            + reconstruction_weight * reconstruction_loss
        )
        loss.backward()
        optimizer.step()

        batch_size = waveform.shape[0]
        totals["loss"] += loss.item() * batch_size
        totals["representation_loss"] += objective["representation_loss"].item() * batch_size
        totals["invariance_loss"] += objective["invariance_loss"].item() * batch_size
        totals["variance_loss"] += objective["variance_loss"].item() * batch_size
        totals["covariance_loss"] += objective["covariance_loss"].item() * batch_size
        totals["gate_loss"] += gate_loss.item() * batch_size
        totals["reconstruction_loss"] += reconstruction_loss.item() * batch_size
        totals["embedding_std"] += objective["embedding_std"].item() * batch_size
        example_count += batch_size
    return {name: value / example_count for name, value in totals.items()}


def collect_window_scores(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    embedding_profiles: dict[str, ConditionedEmbeddingProfile],
    feature_memory: ConditionedFeatureMemory,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    recording_ids: list[str] = []
    condition_ids: list[str] = []
    labels: list[int] = []
    scores: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            batch_conditions = list(batch["condition_id"])
            output = forward_model(
                model,
                model_name,
                waveform,
                batch_conditions,
                progress=1.0,
            )
            recording_ids.extend(batch["recording_id"])
            condition_ids.extend(batch_conditions)
            labels.extend(batch["label"].tolist())
            component_scores = []
            for name in ("sustained_embedding", "event_embedding"):
                component_score, _, _ = embedding_profiles[name].scores(
                    output[name], batch_conditions
                )
                score_name = name.replace("_embedding", "_score")
                scores[score_name].extend(component_score.cpu().tolist())
                component_scores.append(component_score)
            if "subband_embedding" in output:
                _, subband_z, _ = embedding_profiles["subband_embedding"].scores(
                    output["subband_embedding"], batch_conditions
                )
                if subband_z.shape[1] % 2 != 0:
                    raise ValueError("subband embedding must contain mean and event halves.")
                num_subbands = subband_z.shape[1] // 2
                local_subband_score = subband_z.reshape(
                    subband_z.shape[0], 2, num_subbands
                ).square().mean(dim=1)
                top_count = max(1, int(np.ceil(num_subbands * 0.25)))
                subband_score = local_subband_score.topk(top_count, dim=1).values.mean(dim=1)
                scores["subband_score"].extend(subband_score.cpu().tolist())
                component_scores.append(subband_score)
            embedding_score = torch.stack(component_scores).amax(dim=0)
            scores["embedding_score"].extend(embedding_score.cpu().tolist())
            memory_output = feature_memory.score(
                local_feature_map(output), batch_conditions
            )
            memory_score = memory_output["recording_memory_score"]
            scores["memory_score"].extend(memory_score.cpu().tolist())
            scores["learned_score"].extend(memory_score.cpu().tolist())
            if "recording_score" in output:
                scores["profile_score"].extend(output["recording_score"].cpu().tolist())
    return {
        "recording_ids": recording_ids,
        "condition_ids": condition_ids,
        "labels": np.asarray(labels),
        "scores": {name: np.asarray(values) for name, values in scores.items()},
    }


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


def aggregate_score_outputs(
    outputs: dict[str, object],
    quantile: float,
) -> dict[str, object]:
    recording_ids = outputs["recording_ids"]
    condition_ids = outputs["condition_ids"]
    labels = outputs["labels"]
    score_arrays = outputs["scores"]
    grouped: dict[str, dict[str, object]] = {}
    for index, recording_id in enumerate(recording_ids):
        condition_id = condition_ids[index]
        label = int(labels[index])
        row = grouped.setdefault(
            recording_id,
            {"condition_id": condition_id, "label": label, "scores": defaultdict(list)},
        )
        if row["condition_id"] != condition_id or row["label"] != label:
            raise ValueError("one recording contains conflicting metadata.")
        for name, values in score_arrays.items():
            row["scores"][name].append(float(values[index]))
    ordered_ids = sorted(grouped)
    return {
        "recording_ids": ordered_ids,
        "condition_ids": [grouped[item]["condition_id"] for item in ordered_ids],
        "labels": np.asarray([grouped[item]["label"] for item in ordered_ids]),
        "scores": {
            name: np.asarray(
                [
                    np.quantile(grouped[item]["scores"][name], quantile)
                    for item in ordered_ids
                ]
            )
            for name in score_arrays
        },
    }


def evaluate_representation_epoch(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    gate_weight: float,
    variance_target: float,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
    reconstruction_weight: float,
    mask_fraction: float,
    noise_fraction: float,
    max_shift_fraction: float,
) -> float:
    model.eval()
    total = 0.0
    example_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            waveform = batch["waveform"].to(device, non_blocking=True)
            if waveform.shape[0] < 2:
                continue
            conditions = list(batch["condition_id"])
            if model_name == "egfn" and reconstruction_weight > 0:
                num_subbands = model.frontend.num_subbands
                first_mask = deterministic_subband_mask(
                    waveform.shape[0],
                    num_subbands,
                    mask_fraction,
                    batch_index * waveform.shape[0],
                    device,
                )
                second_mask = deterministic_subband_mask(
                    waveform.shape[0],
                    num_subbands,
                    mask_fraction,
                    batch_index * waveform.shape[0] + num_subbands // 2,
                    device,
                )
            else:
                first_mask = None
                second_mask = None
            first_output = forward_model(
                model,
                model_name,
                make_audio_view(waveform, noise_fraction, max_shift_fraction),
                conditions,
                progress=1.0,
                masked_subbands=first_mask,
            )
            second_output = forward_model(
                model,
                model_name,
                make_audio_view(waveform, noise_fraction, max_shift_fraction),
                conditions,
                progress=1.0,
                masked_subbands=second_mask,
            )
            objective = anti_collapse_loss(
                first_output["embedding"],
                second_output["embedding"],
                variance_target=variance_target,
                invariance_weight=invariance_weight,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
            )
            gate_loss = 0.5 * (
                first_output.get(
                    "gate_regularization_loss",
                    objective["representation_loss"].new_zeros(()),
                )
                + second_output.get(
                    "gate_regularization_loss",
                    objective["representation_loss"].new_zeros(()),
                )
            )
            reconstruction_loss = 0.5 * (
                masked_reconstruction_loss(first_output, first_mask)
                + masked_reconstruction_loss(second_output, second_mask)
            )
            loss = (
                objective["representation_loss"]
                + gate_weight * gate_loss
                + reconstruction_weight * reconstruction_loss
            )
            total += loss.item() * waveform.shape[0]
            example_count += waveform.shape[0]
    if example_count == 0:
        raise RuntimeError("validation loader did not contain a complete batch.")
    return total / example_count


def metrics_and_distributions_by_condition(
    outputs: dict[str, object],
    threshold: float,
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, dict[str, float | int]]]]:
    conditions = np.asarray(outputs["condition_ids"])
    labels = outputs["labels"]
    scores = outputs["scores"]["learned_score"]
    condition_metrics = {}
    distributions = {}
    for condition_id in sorted(set(conditions.tolist())):
        mask = conditions == condition_id
        condition_metrics[condition_id] = metrics_at_threshold(
            labels[mask], scores[mask], threshold
        )
        distributions[condition_id] = {}
        for label_name, label_value in (("normal", 0), ("anomalous", 1)):
            values = scores[mask & (labels == label_value)]
            if values.size == 0:
                continue
            distributions[condition_id][label_name] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
            }
    return condition_metrics, distributions


def write_recording_scores(outputs: dict[str, object], path: Path) -> None:
    score_names = sorted(outputs["scores"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording_id", "condition_id", "label", *score_names],
        )
        writer.writeheader()
        for index, recording_id in enumerate(outputs["recording_ids"]):
            writer.writerow(
                {
                    "recording_id": recording_id,
                    "condition_id": outputs["condition_ids"][index],
                    "label": int(outputs["labels"][index]),
                    **{
                        name: float(values[index])
                        for name, values in outputs["scores"].items()
                    },
                }
            )


def frontend_filter_summary(model: torch.nn.Module) -> dict[str, object] | None:
    frontend = getattr(model, "frontend", None)
    if frontend is None:
        return None
    weights = frontend.effective_subband_weights().detach().cpu()
    frequencies = frontend.frequency_bins_hz.detach().cpu()
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "learnable": bool(frontend.learnable_subband_weights),
        "centroid_hz": (weights * frequencies).sum(dim=1).tolist(),
        "peak_hz": frequencies[weights.argmax(dim=1)].tolist(),
        "entropy": entropy.tolist(),
        "weights": weights.tolist(),
    }


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
    epoch: int,
    validation_loss: float,
    args: argparse.Namespace,
    embedding_profiles: dict[str, ConditionedEmbeddingProfile] | None = None,
    feature_memory: ConditionedFeatureMemory | None = None,
    spectral_profile: ConditionedNormalProfile | None = None,
    threshold: float | None = None,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "protocol_version": 4,
            "model_state": model_state,
            "epoch": int(epoch),
            "validation_representation_loss": float(validation_loss),
            "args": serializable_args(args),
            "embedding_profiles": (
                None
                if embedding_profiles is None
                else {
                    name: profile.to_dict()
                    for name, profile in embedding_profiles.items()
                }
            ),
            "feature_memory": (
                None if feature_memory is None else feature_memory.to_dict()
            ),
            "spectral_profile": (
                None if spectral_profile is None else spectral_profile.to_dict()
            ),
            "threshold": threshold,
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
    if (
        args.memory_size <= 0
        or args.memory_temporal_pool <= 0
        or args.memory_query_chunk_size <= 0
        or not 0 < args.memory_top_fraction <= 1
    ):
        raise ValueError("feature memory configuration values are invalid.")
    if args.reconstruction_weight < 0 or not 0 < args.mask_fraction < 1:
        raise ValueError("masked reconstruction configuration values are invalid.")
    if args.learnable_subband_weights and args.model != "egfn":
        raise ValueError("learnable subband weights are only available for EGFN.")
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

    model = build_model(
        args.model,
        profile,
        learnable_subband_weights=args.learnable_subband_weights,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"trainable_parameters={trainable_parameters}")
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    embedding_profiles: dict[str, ConditionedEmbeddingProfile] | None = None
    feature_memory: ConditionedFeatureMemory | None = None
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
        if checkpoint.get("protocol_version") != 4:
            raise ValueError("checkpoint predates the adaptive reconstruction protocol.")
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("model") != args.model:
            raise ValueError("checkpoint model does not match --model.")
        if int(checkpoint_args.get("seed", -1)) != args.seed:
            raise ValueError("checkpoint seed does not match --seed.")
        if bool(checkpoint_args.get("learnable_subband_weights", False)) != (
            args.learnable_subband_weights
        ):
            raise ValueError("checkpoint learnable-subband configuration does not match.")
        if "variance_target" not in checkpoint_args:
            raise ValueError("checkpoint was not trained with the anti-collapse objective.")
        best_state = checkpoint["model_state"]
        stored_spectral_profile = checkpoint.get("spectral_profile")
        if args.model == "egfn":
            if not stored_spectral_profile:
                raise ValueError(
                    "checkpoint does not contain its calibrated spectral profile."
                )
            model.normal_profile = ConditionedNormalProfile(
                condition_ids=stored_spectral_profile["condition_ids"],
                mean=torch.tensor(stored_spectral_profile["mean"]),
                std=torch.tensor(stored_spectral_profile["std"]),
                record_counts=torch.tensor(stored_spectral_profile["record_counts"]),
                fallback_mean=torch.tensor(stored_spectral_profile["fallback_mean"]),
                fallback_std=torch.tensor(stored_spectral_profile["fallback_std"]),
                epsilon=float(stored_spectral_profile["epsilon"]),
                metadata=stored_spectral_profile.get("metadata", {}),
            ).to(device)
        stored_profiles = checkpoint.get("embedding_profiles")
        if not stored_profiles:
            raise ValueError("checkpoint does not contain conditioned embedding profiles.")
        embedding_profiles = {
            name: ConditionedEmbeddingProfile.from_dict(payload).to(device)
            for name, payload in stored_profiles.items()
        }
        stored_memory = checkpoint.get("feature_memory")
        if not stored_memory:
            raise ValueError("checkpoint does not contain a local feature memory.")
        feature_memory = ConditionedFeatureMemory.from_dict(stored_memory).to(device)
        best_epoch = int(checkpoint.get("epoch", 0))
        best_validation_loss = float(
            checkpoint.get("validation_representation_loss", float("nan"))
        )
        print(
            f"loaded_checkpoint={recovery_checkpoint_path} "
            f"epoch={best_epoch or 'unknown'}"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
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
                args.reconstruction_weight,
                args.mask_fraction,
                args.view_noise_fraction,
                args.view_max_shift_fraction,
            )
            validation_loss = evaluate_representation_epoch(
                model,
                args.model,
                validation_loader,
                device,
                args.gate_regularization_weight,
                args.variance_target,
                args.invariance_weight,
                args.variance_weight,
                args.covariance_weight,
                args.reconstruction_weight,
                args.mask_fraction,
                args.view_noise_fraction,
                args.view_max_shift_fraction,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{name}": value for name, value in train_metrics.items()},
                "validation_representation_loss": validation_loss,
            }
            history.append(row)
            save_history(history, output_dir / "history.csv")
            print(
                f"epoch={epoch:02d} loss={train_metrics['loss']:.6f} "
                f"embedding_std={train_metrics['embedding_std']:.6f} "
                f"val_representation_loss={validation_loss:.6f}"
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                save_recovery_checkpoint(
                    recovery_checkpoint_path,
                    best_state,
                    epoch,
                    validation_loss,
                    args,
                )
            if is_meaningful_improvement(
                validation_loss,
                early_stopping_reference,
                args.early_stopping_min_relative_improvement,
            ):
                early_stopping_reference = validation_loss
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
        best_state = best_checkpoint["model_state"]
    model.load_state_dict(best_state)
    model.to(device)
    if (
        not args.evaluate_only
        and args.model == "egfn"
        and args.learnable_subband_weights
    ):
        model.normal_profile = estimate_spectral_profile(
            model,
            center_loader,
            device,
            dict(profile.metadata),
        )
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    if embedding_profiles is None:
        embedding_profiles = estimate_embedding_profiles(
            model, args.model, center_loader, device
        )
    if feature_memory is None:
        feature_memory = estimate_feature_memory(
            model,
            args.model,
            center_loader,
            device,
            args.memory_size,
            args.memory_temporal_pool,
            args.memory_top_fraction,
            args.memory_query_chunk_size,
            args.seed,
        )

    validation_windows = collect_window_scores(
        model,
        args.model,
        validation_loader,
        embedding_profiles,
        feature_memory,
        device,
    )
    validation_outputs = aggregate_score_outputs(
        validation_windows, args.recording_quantile
    )
    if np.any(validation_outputs["labels"] != 0):
        raise ValueError("validation must contain only normal recordings.")
    validation_scores = validation_outputs["scores"]["learned_score"]
    threshold = float(np.quantile(validation_scores, args.normal_threshold_quantile))
    test_windows = collect_window_scores(
        model,
        args.model,
        test_loader,
        embedding_profiles,
        feature_memory,
        device,
    )
    test_outputs = aggregate_score_outputs(test_windows, args.recording_quantile)
    test_labels = test_outputs["labels"]
    test_scores = test_outputs["scores"]["learned_score"]
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    condition_metrics, score_distributions = metrics_and_distributions_by_condition(
        test_outputs, threshold
    )
    result: dict[str, object] = {
        "model": args.model,
        "protocol": "normal_only_adaptive_masked_reconstruction_v4",
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_validation_representation_loss": best_validation_loss,
        "embedding_profiles": {
            name: profile.to_dict() for name, profile in embedding_profiles.items()
        },
        "feature_memory": feature_memory.summary(),
        "frontend_filters": frontend_filter_summary(model),
        "threshold_source": f"validation_normal_quantile_{args.normal_threshold_quantile}",
        "metrics": metrics,
        "component_auc": {
            name: metrics_at_threshold(test_labels, values, threshold)["auc"]
            for name, values in test_outputs["scores"].items()
        },
        "metrics_by_condition": condition_metrics,
        "score_distributions_by_condition": score_distributions,
        "split_counts": {
            "train_normal": len(train_records),
            "validation_normal": len(validation_records),
            "test_normal": int((test_labels == 0).sum()),
            "test_anomalous": int((test_labels == 1).sum()),
        },
        "args": serializable_args(args),
    }

    save_history(history, output_dir / "history.csv")
    write_recording_scores(test_outputs, output_dir / "test_recording_scores.csv")
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_recovery_checkpoint(
        output_dir / "checkpoint.pt",
        best_state,
        best_epoch,
        best_validation_loss,
        args,
        embedding_profiles,
        feature_memory,
        model.normal_profile if args.model == "egfn" else None,
        threshold,
    )
    if not args.evaluate_only:
        save_recovery_checkpoint(
            recovery_checkpoint_path,
            best_state,
            best_epoch,
            best_validation_loss,
            args,
            embedding_profiles,
            feature_memory,
            model.normal_profile if args.model == "egfn" else None,
            threshold,
        )
    print(
        f"test_auc={metrics['auc']:.4f} test_f1={metrics['f1']:.4f} "
        f"threshold={threshold:.6f}"
    )
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
