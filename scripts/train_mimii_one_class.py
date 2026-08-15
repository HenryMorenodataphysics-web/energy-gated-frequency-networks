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
    add_hybrid_anomaly_partitions,
    AnomalyAudioRecord,
    AnomalyWindowDataset,
    ConditionBatchSampler,
    HybridConditionBatchSampler,
    find_dcase2020_development_split,
    find_folder_anomaly_recordings,
    find_mimii_recordings,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
    validate_hybrid_anomaly_split,
)
from src.models import Conv1DAnomalyEncoder, HierarchicalAnomalyDetector
from src.utils.binary_evaluation import metrics_at_threshold


def parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EGFN or Conv1D with a shared one-class audio protocol."
    )
    parser.add_argument("--model", choices=("egfn", "conv1d"), required=True)
    parser.add_argument(
        "--training-mode",
        choices=("one_class", "hybrid"),
        default="one_class",
        help="Hybrid keeps normal-only representation learning and adds labeled BCE.",
    )
    parser.add_argument(
        "--egfn-embedding-channels",
        type=int,
        default=8,
        help="Internal EGFN width; use 22 for the approximately 1,000-parameter variant.",
    )
    parser.add_argument(
        "--conv1d-channels",
        type=parse_int_tuple,
        default=(4, 8, 8),
        help="Three Conv1D encoder widths; use 2,2,7 for the 304-parameter control.",
    )
    parser.add_argument(
        "--dataset-format", choices=("mimii", "folders", "dcase2020"), default="mimii"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument("--normal-dir", type=Path, default=None)
    parser.add_argument("--anomalous-dir", type=Path, default=None)
    parser.add_argument("--dcase-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="folder_audio")
    parser.add_argument(
        "--condition-mode", choices=("global", "parent"), default="global"
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument(
        "--macro-edges-hz",
        type=parse_float_tuple,
        default=None,
        help="Four comma-separated macro-band edges. Defaults scale with Nyquist.",
    )
    parser.add_argument(
        "--subbands-per-macro", type=parse_int_tuple, default=(4, 8, 4)
    )
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
    parser.add_argument(
        "--filter-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Epochs that train the encoder and gates with fixed subband weights "
            "before filter adaptation. Required with --learnable-subband-weights."
        ),
    )
    parser.add_argument(
        "--filter-adaptation-epochs",
        type=int,
        default=0,
        help=(
            "Epochs that update only the subband weights. The normal profile is "
            "refitted from training-normal audio after every adaptation epoch."
        ),
    )
    parser.add_argument(
        "--gate-mode",
        choices=("none", "macro", "subband", "hierarchical"),
        default="hierarchical",
    )
    parser.add_argument("--normalize-gate-inputs", action="store_true")
    parser.add_argument("--conditional-subgates", action="store_true")
    parser.add_argument("--harmonic-context", action="store_true")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--variance-target", type=float, default=0.05)
    parser.add_argument("--invariance-weight", type=float, default=25.0)
    parser.add_argument("--variance-weight", type=float, default=25.0)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--view-noise-fraction", type=float, default=0.005)
    parser.add_argument("--view-max-shift-fraction", type=float, default=0.05)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--anomaly-train-ratio", type=float, default=0.6)
    parser.add_argument("--anomaly-validation-ratio", type=float, default=0.2)
    parser.add_argument("--pretrained-one-class-checkpoint", type=Path, default=None)
    parser.add_argument("--head-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--hybrid-max-validation-fpr",
        type=float,
        default=None,
        help="Select the hybrid threshold under this validation false-positive-rate cap.",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--recording-quantile", type=float, default=0.95)
    parser.add_argument("--normal-threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--primary-score",
        choices=(
            "auto",
            "reconstruction_score",
            "memory_score",
            "embedding_score",
            "profile_score",
            "sustained_score",
            "event_score",
            "subband_score",
            "supervised_score",
            "hybrid_score",
        ),
        default="auto",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=512,
        help="Maximum normal feature vectors retained per condition and subband.",
    )
    parser.add_argument(
        "--memory-representation",
        choices=("encoder", "gated_profile", "activation_signature"),
        default="encoder",
    )
    parser.add_argument(
        "--objective-representation",
        choices=("encoder", "memory"),
        default="encoder",
        help=(
            "Representation optimized and used for checkpoint selection. "
            "Use memory to optimize a pooled proxy of --memory-representation."
        ),
    )
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
    return parser.parse_args(argv)


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
    egfn_embedding_channels: int = 8,
    conv1d_channels: tuple[int, ...] = (4, 8, 8),
    supervised_anomaly_head: bool = False,
    learnable_subband_weights: bool = False,
    gate_mode: str = "hierarchical",
    normalize_gate_inputs: bool = False,
    conditional_subgates: bool = False,
    harmonic_context: bool = False,
) -> torch.nn.Module:
    if model_name == "conv1d":
        if len(conv1d_channels) != 3:
            raise ValueError("Conv1D requires exactly three channel widths.")
        return Conv1DAnomalyEncoder(
            embedding_channels=8,
            channels=tuple(conv1d_channels),
        )
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
        gate_mode=gate_mode,
        normalize_gate_inputs=normalize_gate_inputs,
        conditional_subgates=conditional_subgates,
        harmonic_context=harmonic_context,
    )
    return HierarchicalAnomalyDetector(
        frontend,
        profile,
        embedding_channels=egfn_embedding_channels,
        supervised_anomaly_head=supervised_anomaly_head,
    )


def bootstrap_normal_profile(
    records: tuple[AnomalyAudioRecord, ...],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    macro_edges_hz: tuple[float, ...] | None,
    subbands_per_macro: tuple[int, ...],
) -> ConditionedNormalProfile:
    """Create structural placeholders that are refitted from training normals."""
    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("frontend sample rate and STFT values must be positive.")
    nyquist = sample_rate / 2
    edges = macro_edges_hz or (0.0, nyquist / 12, 2 * nyquist / 3, nyquist)
    if len(edges) != 4 or len(subbands_per_macro) != 3:
        raise ValueError("the hierarchical frontend requires three macro bands.")
    frontend = HierarchicalSpectralFrontend(
        sample_rate=sample_rate,
        macro_edges_hz=edges,
        subbands_per_macro=subbands_per_macro,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    condition_ids = tuple(
        sorted({f"{record.dataset_name}/{record.condition_id}" for record in records})
    )
    counts = torch.tensor(
        [
            sum(
                f"{record.dataset_name}/{record.condition_id}" == condition_id
                for record in records
            )
            for condition_id in condition_ids
        ]
    )
    shape = (len(condition_ids), frontend.num_subbands, 2)
    return ConditionedNormalProfile(
        condition_ids=condition_ids,
        mean=torch.zeros(shape),
        std=torch.ones(shape),
        record_counts=counts,
        fallback_mean=torch.zeros(frontend.num_subbands, 2),
        fallback_std=torch.ones(frontend.num_subbands, 2),
        metadata={
            "dataset": records[0].dataset_name,
            "source_split": "bootstrap_before_train_calibration",
            "frontend": {
                "sample_rate": sample_rate,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "macro_edges_hz": list(edges),
                "subbands_per_macro": list(subbands_per_macro),
                "subband_edges_hz": frontend.subband_edges_hz.detach().cpu().tolist(),
            },
        },
    )


def training_stage(
    epoch: int,
    learnable_subband_weights: bool,
    filter_warmup_epochs: int,
    filter_adaptation_epochs: int,
    head_warmup_epochs: int = 0,
) -> str:
    """Return the active stage for a one-indexed training epoch."""
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    if epoch <= head_warmup_epochs:
        return "head_warmup"
    if not learnable_subband_weights:
        return "representation"
    if epoch <= filter_warmup_epochs:
        return "filter_warmup"
    if epoch <= filter_warmup_epochs + filter_adaptation_epochs:
        return "filter_adaptation"
    return "representation_finetune"


def configure_training_stage(
    model: torch.nn.Module,
    model_name: str,
    stage: str,
) -> int:
    """Freeze parameters so filter learning cannot drift the downstream model."""
    allowed = {
        "representation",
        "filter_warmup",
        "filter_adaptation",
        "representation_finetune",
        "head_warmup",
    }
    if stage not in allowed:
        raise ValueError(f"unknown training stage: {stage}.")
    if model_name != "egfn" and stage != "representation":
        raise ValueError("filter training stages require an EGFN model.")

    if stage == "head_warmup":
        anomaly_head = getattr(model, "anomaly_head", None)
        if anomaly_head is None:
            raise ValueError("head warmup requires a supervised anomaly head.")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in anomaly_head.parameters():
            parameter.requires_grad_(True)
        return sum(parameter.numel() for parameter in anomaly_head.parameters())

    filter_parameter = (
        model.frontend.subband_weight_logits if model_name == "egfn" else None
    )
    for parameter in model.parameters():
        parameter.requires_grad_(stage != "filter_adaptation")
    if filter_parameter is not None:
        filter_parameter.requires_grad_(stage == "filter_adaptation")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"training stage {stage} has no trainable parameters.")
    return sum(parameter.numel() for parameter in trainable)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("optimizer received no trainable parameters.")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def load_pretrained_one_class_backbone(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> int:
    """Load every shared one-class tensor while leaving the new head random."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("model") != "egfn":
        raise ValueError("pretrained checkpoint must contain an EGFN model.")
    if checkpoint_args.get("training_mode", "one_class") != "one_class":
        raise ValueError("pretrained checkpoint must use one-class training.")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("pretrained checkpoint does not contain model_state.")
    target_state = model.state_dict()
    shared_names = [
        name for name in target_state if not name.startswith("anomaly_head.")
    ]
    missing = [name for name in shared_names if name not in source_state]
    mismatched = [
        name
        for name in shared_names
        if name in source_state and source_state[name].shape != target_state[name].shape
    ]
    if missing or mismatched:
        raise ValueError(
            "pretrained backbone is incompatible: "
            f"missing={missing[:3]} mismatched={mismatched[:3]}"
        )
    model.load_state_dict(
        {name: source_state[name] for name in shared_names},
        strict=False,
    )
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("anomaly_head.")
    )


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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise_fraction < 0 or not 0 <= max_shift_fraction < 1:
        raise ValueError("view augmentation values are outside their valid ranges.")
    maximum_shift = int(waveform.shape[-1] * max_shift_fraction)
    shifts = torch.randint(
        -maximum_shift,
        maximum_shift + 1,
        (waveform.shape[0],),
        device=waveform.device,
        generator=generator,
    )
    shifted_items = []
    for item, sampled_shift in zip(waveform, shifts, strict=True):
        shift = int(sampled_shift.item())
        shifted_item = torch.zeros_like(item)
        if shift > 0:
            shifted_item[..., shift:] = item[..., :-shift]
        elif shift < 0:
            shifted_item[..., :shift] = item[..., -shift:]
        else:
            shifted_item.copy_(item)
        original_rms = item.square().mean().sqrt().clamp_min(1e-8)
        shifted_rms = shifted_item.square().mean().sqrt().clamp_min(1e-8)
        shifted_items.append(shifted_item * (original_rms / shifted_rms))
    shifted = torch.stack(shifted_items)
    if noise_fraction == 0:
        return shifted
    rms = shifted.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    noise = torch.randn(
        shifted.shape,
        dtype=shifted.dtype,
        device=shifted.device,
        generator=generator,
    )
    return shifted + noise * rms * noise_fraction


def complementary_subband_masks(
    batch_size: int,
    num_subbands: int,
    mask_fraction: float,
    device: torch.device,
) -> list[torch.Tensor]:
    """Cover every subband exactly once with deterministic inference masks."""
    if batch_size <= 0 or num_subbands <= 1 or not 0 < mask_fraction < 1:
        raise ValueError("complementary mask configuration is invalid.")
    mask_count = min(num_subbands - 1, max(1, round(num_subbands * mask_fraction)))
    masks = []
    for start in range(0, num_subbands, mask_count):
        mask = torch.zeros(batch_size, num_subbands, dtype=torch.bool, device=device)
        mask[:, start : start + mask_count] = True
        masks.append(mask)
    return masks


def reconstruction_anomaly_scores(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    condition_ids: list[str],
    mask_fraction: float,
    top_fraction: float,
) -> torch.Tensor:
    """Score masked spectral prediction error using complementary subband masks."""
    if not hasattr(model, "frontend"):
        raise ValueError("reconstruction scoring requires an EGFN frontend.")
    local_error = None
    coverage = None
    for mask in complementary_subband_masks(
        waveform.shape[0],
        model.frontend.num_subbands,
        mask_fraction,
        waveform.device,
    ):
        output = forward_model(
            model,
            "egfn",
            waveform,
            condition_ids,
            progress=1.0,
            masked_subbands=mask,
        )
        prediction = output["reconstructed_log_energy_z"]
        target = output["z_scores"][:, :, 0, :].detach().clamp(-10.0, 10.0)
        error = torch.nn.functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )
        expanded_mask = mask.unsqueeze(-1).expand_as(error)
        if local_error is None:
            local_error = torch.zeros_like(error)
            coverage = torch.zeros_like(error)
        local_error = local_error + error * expanded_mask
        coverage = coverage + expanded_mask
    if local_error is None or coverage is None or torch.any(coverage != 1):
        raise RuntimeError("complementary masks did not cover each subband exactly once.")
    flattened = local_error.flatten(1)
    top_count = max(1, int(np.ceil(flattened.shape[1] * top_fraction)))
    return flattened.topk(top_count, dim=1).values.mean(dim=1)


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


def local_feature_map(
    output: dict[str, torch.Tensor],
    representation: str = "encoder",
) -> torch.Tensor:
    if representation == "gated_profile":
        if "gated_profile_features" not in output:
            raise ValueError("model did not expose gated profile features.")
        return output["gated_profile_features"]
    if representation == "activation_signature":
        if "activation_signature_features" not in output:
            raise ValueError("model did not expose activation signature features.")
        return output["activation_signature_features"]
    if representation != "encoder":
        raise ValueError(
            "memory representation must be encoder, gated_profile, or "
            "activation_signature."
        )
    if "embedding_map" in output:
        return output["embedding_map"]
    if "feature_map" in output:
        feature_map = output["feature_map"]
        if feature_map.ndim == 3:
            return feature_map.unsqueeze(2)
    raise ValueError("model did not expose a local feature map.")


def objective_embedding(
    output: dict[str, torch.Tensor],
    objective_representation: str,
    memory_representation: str,
) -> torch.Tensor:
    """Return the exact encoder embedding or a pooled proxy of memory features."""
    if objective_representation == "encoder":
        return output["embedding"]
    if objective_representation != "memory":
        raise ValueError("objective representation must be encoder or memory.")
    feature_map = local_feature_map(output, memory_representation)
    temporal_mean = feature_map.mean(dim=-1)
    if feature_map.shape[-1] == 1:
        return temporal_mean.flatten(1)
    temporal_peak = feature_map.amax(dim=-1)
    return torch.cat((temporal_mean, temporal_peak), dim=1).flatten(1)


def representation_geometry(embeddings: torch.Tensor) -> dict[str, float | int]:
    """Measure rank and redundancy; marginal standard deviation alone is insufficient."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must have shape [example, dimension] with n >= 2.")
    values = embeddings.detach().float().cpu()
    centered = values - values.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= 0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        effective_rank = float(
            torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            ).item()
        )
    safe_std = std.clamp_min(1e-12)
    correlation = covariance / (safe_std[:, None] * safe_std[None, :])
    off_diagonal = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    mean_abs_correlation = (
        float(correlation[off_diagonal].abs().mean().item())
        if off_diagonal.any()
        else 0.0
    )
    return {
        "examples": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
        "mean_dimension_std": float(std.mean().item()),
        "active_dimensions": int((std > 1e-3).sum().item()),
        "effective_rank": effective_rank,
        "mean_absolute_off_diagonal_correlation": mean_abs_correlation,
    }


def estimate_model_diagnostics(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    objective_representation: str,
    memory_representation: str,
    collapse_threshold: float = 0.02,
) -> tuple[dict[str, float | int], dict[str, dict[str, float]] | None]:
    collected = []
    gate_totals = {
        name: {"sum": 0.0, "square_sum": 0.0, "dead": 0.0, "open": 0.0, "n": 0}
        for name in ("macro", "subband", "joint")
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = forward_model(
                model,
                model_name,
                batch["waveform"].to(device, non_blocking=True),
                list(batch["condition_id"]),
                progress=1.0,
            )
            collected.append(
                objective_embedding(
                    output, objective_representation, memory_representation
                ).cpu()
            )
            if model_name == "egfn":
                for name, key in (
                    ("macro", "macro_gates"),
                    ("subband", "subband_gates"),
                    ("joint", "joint_gates"),
                ):
                    gates = output[key].detach().float()
                    gate_totals[name]["sum"] += float(gates.sum().item())
                    gate_totals[name]["square_sum"] += float(
                        gates.square().sum().item()
                    )
                    gate_totals[name]["dead"] += float(
                        (gates <= collapse_threshold).sum().item()
                    )
                    gate_totals[name]["open"] += float(
                        (gates >= 1.0 - collapse_threshold).sum().item()
                    )
                    gate_totals[name]["n"] += gates.numel()
    geometry = representation_geometry(torch.cat(collected, dim=0))
    if model_name != "egfn":
        return geometry, None
    result = {}
    for name, values in gate_totals.items():
        count = int(values["n"])
        mean = float(values["sum"]) / count
        variance = max(float(values["square_sum"]) / count - mean * mean, 0.0)
        result[name] = {
            "mean": mean,
            "std": variance**0.5,
            "dead_fraction": float(values["dead"]) / count,
            "always_open_fraction": float(values["open"]) / count,
        }
    return geometry, result


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
    representation: str,
    label_filter: int | None = None,
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
            feature_map = local_feature_map(output, representation)
            condition_ids = list(batch["condition_id"])
            if label_filter is not None:
                selected = batch["label"].eq(label_filter)
                if not bool(selected.any()):
                    continue
                feature_map = feature_map[selected.to(feature_map.device)]
                condition_ids = [
                    condition_id
                    for condition_id, keep in zip(
                        condition_ids, selected.tolist(), strict=True
                    )
                    if keep
                ]
            estimator.update(feature_map, condition_ids)
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
    objective_representation: str = "encoder",
    memory_representation: str = "encoder",
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
            objective_embedding(
                first_output, objective_representation, memory_representation
            ),
            objective_embedding(
                second_output, objective_representation, memory_representation
            ),
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


def supervised_anomaly_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if "anomaly_logit" not in output:
        raise ValueError("hybrid training requires a supervised anomaly head.")
    if positive_weight <= 0:
        raise ValueError("positive class weight must be positive.")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        output["anomaly_logit"],
        labels.float(),
        pos_weight=output["anomaly_logit"].new_tensor(positive_weight),
    )


def pairwise_anomaly_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if margin < 0:
        raise ValueError("ranking margin must be non-negative.")
    normal_logits = logits[labels == 0]
    anomalous_logits = logits[labels == 1]
    if normal_logits.numel() == 0 or anomalous_logits.numel() == 0:
        raise ValueError("ranking loss requires normal and anomalous examples.")
    differences = anomalous_logits[:, None] - normal_logits[None, :]
    return torch.nn.functional.softplus(margin - differences).mean()


def supervised_anomaly_objective(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    positive_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> dict[str, torch.Tensor]:
    if ranking_weight < 0:
        raise ValueError("ranking weight must be non-negative.")
    bce_loss = supervised_anomaly_loss(output, labels, positive_weight)
    ranking_loss = (
        pairwise_anomaly_ranking_loss(
            output["anomaly_logit"], labels, ranking_margin
        )
        if ranking_weight > 0
        else bce_loss.new_zeros(())
    )
    return {
        "supervised_loss": bce_loss + ranking_weight * ranking_loss,
        "bce_loss": bce_loss,
        "ranking_loss": ranking_loss,
    }


def run_supervised_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    positive_weight: float,
    supervised_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    noise_fraction: float,
    max_shift_fraction: float,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    correct = 0
    example_count = 0
    for batch in loader:
        waveform = batch["waveform"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = forward_model(
            model,
            "egfn",
            make_audio_view(waveform, noise_fraction, max_shift_fraction),
            list(batch["condition_id"]),
            progress=1.0,
        )
        objective = supervised_anomaly_objective(
            output,
            labels,
            positive_weight,
            ranking_weight,
            ranking_margin,
        )
        (supervised_weight * objective["supervised_loss"]).backward()
        optimizer.step()
        batch_size = waveform.shape[0]
        for name, value in objective.items():
            totals[name] += value.item() * batch_size
        predictions = (output["anomaly_logit"] >= 0).long()
        correct += int((predictions == labels).sum().item())
        example_count += batch_size
    return {
        **{name: value / example_count for name, value in totals.items()},
        "supervised_accuracy": correct / example_count,
    }


def evaluate_supervised_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> dict[str, float]:
    model.eval()
    collected_logits = []
    collected_labels = []
    collected_conditions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            output = forward_model(
                model,
                "egfn",
                waveform,
                list(batch["condition_id"]),
                progress=1.0,
            )
            collected_logits.append(output["anomaly_logit"])
            collected_labels.append(labels)
            collected_conditions.extend(batch["condition_id"])
    logits = torch.cat(collected_logits)
    labels = torch.cat(collected_labels)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=logits.new_tensor(positive_weight),
    )
    condition_losses = []
    if ranking_weight > 0:
        condition_array = np.asarray(collected_conditions)
        for condition in sorted(set(collected_conditions)):
            indices = torch.from_numpy(condition_array == condition).to(logits.device)
            condition_losses.append(
                pairwise_anomaly_ranking_loss(
                    logits[indices], labels[indices], ranking_margin
                )
            )
    ranking_loss = (
        torch.stack(condition_losses).mean()
        if condition_losses
        else bce_loss.new_zeros(())
    )
    supervised_loss = bce_loss + ranking_weight * ranking_loss
    predictions = (logits >= 0).long()
    return {
        "supervised_loss": float(supervised_loss.item()),
        "bce_loss": float(bce_loss.item()),
        "ranking_loss": float(ranking_loss.item()),
        "supervised_accuracy": float((predictions == labels).float().mean().item()),
    }


def collect_window_scores(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    embedding_profiles: dict[str, ConditionedEmbeddingProfile],
    feature_memory: ConditionedFeatureMemory,
    device: torch.device,
    reconstruction_mask_fraction: float,
    reconstruction_top_fraction: float,
    memory_representation: str,
    anomaly_feature_memory: ConditionedFeatureMemory | None = None,
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
                local_feature_map(output, memory_representation), batch_conditions
            )
            memory_score = memory_output["recording_memory_score"]
            scores["memory_score"].extend(memory_score.cpu().tolist())
            if anomaly_feature_memory is not None:
                anomaly_memory_output = anomaly_feature_memory.score(
                    local_feature_map(output, memory_representation), batch_conditions
                )
                anomaly_distance = anomaly_memory_output["recording_memory_score"]
                reference_score = memory_score / (
                    memory_score + anomaly_distance + 1e-8
                )
                scores["anomaly_memory_distance"].extend(
                    anomaly_distance.cpu().tolist()
                )
                scores["reference_score"].extend(reference_score.cpu().tolist())
            if "recording_score" in output:
                scores["profile_score"].extend(output["recording_score"].cpu().tolist())
            if "anomaly_logit" in output:
                scores["supervised_score"].extend(
                    output["anomaly_logit"].sigmoid().cpu().tolist()
                )
            if model_name == "egfn":
                reconstruction_score = reconstruction_anomaly_scores(
                    model,
                    waveform,
                    batch_conditions,
                    reconstruction_mask_fraction,
                    reconstruction_top_fraction,
                )
                scores["reconstruction_score"].extend(
                    reconstruction_score.cpu().tolist()
                )
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


def resolve_primary_score(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "reconstruction_score" if model_name == "egfn" else "memory_score"


def fit_condition_thresholds(
    outputs: dict[str, object],
    score_name: str,
    quantile: float,
) -> tuple[dict[str, float], float]:
    labels = outputs["labels"]
    if np.any(labels != 0):
        raise ValueError("threshold calibration must use only normal recordings.")
    conditions = np.asarray(outputs["condition_ids"])
    scores = outputs["scores"][score_name]
    thresholds = {
        condition_id: float(np.quantile(scores[conditions == condition_id], quantile))
        for condition_id in sorted(set(conditions.tolist()))
    }
    return thresholds, float(np.quantile(scores, quantile))


def apply_condition_thresholds(
    outputs: dict[str, object],
    score_name: str,
    thresholds: dict[str, float],
    fallback_threshold: float,
) -> np.ndarray:
    raw_scores = outputs["scores"][score_name]
    calibrated = []
    for score, condition_id in zip(
        raw_scores, outputs["condition_ids"], strict=True
    ):
        scale = max(float(thresholds.get(condition_id, fallback_threshold)), 1e-12)
        calibrated.append(float(score) / scale)
    return np.asarray(calibrated)


def normal_score_outputs(outputs: dict[str, object]) -> dict[str, object]:
    labels = outputs["labels"]
    mask = labels == 0
    return {
        "recording_ids": [
            item for item, keep in zip(outputs["recording_ids"], mask, strict=True) if keep
        ],
        "condition_ids": [
            item for item, keep in zip(outputs["condition_ids"], mask, strict=True) if keep
        ],
        "labels": labels[mask],
        "scores": {
            name: values[mask] for name, values in outputs["scores"].items()
        },
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("threshold selection requires both validation labels.")
    candidates = np.unique(scores)
    best = max(
        (
            (metrics_at_threshold(labels, scores, float(threshold))["f1"], -index, threshold)
            for index, threshold in enumerate(candidates)
        ),
        key=lambda item: (item[0], item[1]),
    )
    return float(best[2])


def threshold_at_max_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_fpr: float,
) -> tuple[float, dict[str, float | int]]:
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum FPR must be in [0, 1).")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("FPR calibration requires both validation labels.")
    candidates = np.concatenate(
        ([np.nextafter(scores.max(), np.inf)], np.unique(scores))
    )
    feasible = []
    for threshold in candidates:
        metrics = metrics_at_threshold(labels, scores, float(threshold))
        normal_count = int(metrics["tn"]) + int(metrics["fp"])
        fpr = int(metrics["fp"]) / normal_count
        if fpr <= maximum_fpr + 1e-12:
            feasible.append(
                (
                    float(metrics["recall"]),
                    float(metrics["f1"]),
                    float(metrics["precision"]),
                    -float(threshold),
                    float(threshold),
                    metrics,
                )
            )
    if not feasible:
        raise RuntimeError("no validation threshold satisfies the requested FPR.")
    *_, threshold, metrics = max(feasible, key=lambda item: item[:4])
    return threshold, metrics


def fit_hybrid_calibration(
    validation_outputs: dict[str, object],
    one_class_score_name: str,
    normal_quantile: float,
    maximum_validation_fpr: float | None = None,
) -> dict[str, object]:
    if "supervised_score" not in validation_outputs["scores"]:
        raise ValueError("hybrid calibration requires supervised scores.")
    normal_outputs = normal_score_outputs(validation_outputs)
    score_names = [one_class_score_name, "supervised_score"]
    if "reference_score" in validation_outputs["scores"]:
        score_names.append("reference_score")
    component_calibrations = {}
    calibrated_components = {}
    for score_name in score_names:
        thresholds, fallback = fit_condition_thresholds(
            normal_outputs, score_name, normal_quantile
        )
        component_calibrations[score_name] = {
            "condition_thresholds": thresholds,
            "fallback_threshold": fallback,
        }
        calibrated_components[score_name] = apply_condition_thresholds(
            validation_outputs, score_name, thresholds, fallback
        )

    labels = validation_outputs["labels"]
    best_key = None
    best_configuration = None
    if "reference_score" in score_names:
        candidate_weights = []
        for normal_weight in np.linspace(0.0, 1.0, 5):
            for reference_weight in np.linspace(0.0, 1.0 - normal_weight, 5):
                supervised_weight = 1.0 - normal_weight - reference_weight
                candidate_weights.append(
                    {
                        one_class_score_name: float(normal_weight),
                        "reference_score": float(reference_weight),
                        "supervised_score": float(supervised_weight),
                    }
                )
    else:
        candidate_weights = [
            {
                one_class_score_name: float(alpha),
                "supervised_score": float(1.0 - alpha),
            }
            for alpha in np.linspace(0.0, 1.0, 5)
        ]
    for weights in candidate_weights:
        scores = sum(
            weight * calibrated_components[score_name]
            for score_name, weight in weights.items()
        )
        if maximum_validation_fpr is None:
            threshold = best_f1_threshold(labels, scores)
            metrics = metrics_at_threshold(labels, scores, threshold)
            key = (
                metrics["auc"],
                metrics["f1"],
                -max(weights.values()),
            )
        else:
            threshold, metrics = threshold_at_max_fpr(
                labels, scores, maximum_validation_fpr
            )
            key = (
                metrics["recall"],
                metrics["f1"],
                metrics["auc"],
                -max(weights.values()),
            )
        if best_key is None or key > best_key:
            best_key = key
            best_configuration = (weights, threshold, metrics)
    if best_configuration is None:
        raise RuntimeError("hybrid calibration did not produce an operating point.")
    weights, threshold, validation_metrics = best_configuration
    result = {
        "one_class_score": one_class_score_name,
        "weights": weights,
        "threshold": float(threshold),
        "maximum_validation_fpr": maximum_validation_fpr,
        "validation_metrics": validation_metrics,
        "components": component_calibrations,
    }
    if "reference_score" not in score_names:
        result["alpha"] = float(weights[one_class_score_name])
    return result


def apply_hybrid_calibration(
    outputs: dict[str, object],
    calibration: dict[str, object],
) -> np.ndarray:
    one_class_score = str(calibration["one_class_score"])
    weights = calibration.get("weights")
    if weights is None:
        alpha = float(calibration["alpha"])
        weights = {
            one_class_score: alpha,
            "supervised_score": 1.0 - alpha,
        }
    calibrated = {}
    for score_name in weights:
        component = calibration["components"][score_name]
        calibrated[score_name] = apply_condition_thresholds(
            outputs,
            score_name,
            component["condition_thresholds"],
            float(component["fallback_threshold"]),
        )
    return sum(
        float(weight) * calibrated[score_name]
        for score_name, weight in weights.items()
    )


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
    validation_seed: int,
    objective_representation: str = "encoder",
    memory_representation: str = "encoder",
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    example_count = 0
    view_generator = torch.Generator(device=device).manual_seed(validation_seed)
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
                make_audio_view(
                    waveform,
                    noise_fraction,
                    max_shift_fraction,
                    generator=view_generator,
                ),
                conditions,
                progress=1.0,
                masked_subbands=first_mask,
            )
            second_output = forward_model(
                model,
                model_name,
                make_audio_view(
                    waveform,
                    noise_fraction,
                    max_shift_fraction,
                    generator=view_generator,
                ),
                conditions,
                progress=1.0,
                masked_subbands=second_mask,
            )
            objective = anti_collapse_loss(
                objective_embedding(
                    first_output, objective_representation, memory_representation
                ),
                objective_embedding(
                    second_output, objective_representation, memory_representation
                ),
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
            batch_size = waveform.shape[0]
            totals["loss"] += loss.item() * batch_size
            totals["representation_loss"] += (
                objective["representation_loss"].item() * batch_size
            )
            totals["invariance_loss"] += objective["invariance_loss"].item() * batch_size
            totals["variance_loss"] += objective["variance_loss"].item() * batch_size
            totals["covariance_loss"] += objective["covariance_loss"].item() * batch_size
            totals["gate_loss"] += gate_loss.item() * batch_size
            totals["reconstruction_loss"] += reconstruction_loss.item() * batch_size
            totals["embedding_std"] += objective["embedding_std"].item() * batch_size
            example_count += batch_size
    if example_count == 0:
        raise RuntimeError("validation loader did not contain a complete batch.")
    return {name: value / example_count for name, value in totals.items()}


def metrics_and_distributions_by_condition(
    outputs: dict[str, object],
    threshold: float,
    score_name: str = "primary_score",
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, dict[str, float | int]]]]:
    conditions = np.asarray(outputs["condition_ids"])
    labels = outputs["labels"]
    scores = outputs["scores"][score_name]
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


def harmonic_context_summary(model: torch.nn.Module) -> dict[str, object] | None:
    frontend = getattr(model, "frontend", None)
    if frontend is None:
        return None
    enabled = bool(frontend.harmonic_context_enabled)
    return {
        "enabled": enabled,
        "second_harmonic_pairs": int(
            torch.count_nonzero(frontend.second_harmonic_matrix).item()
        ),
        "third_harmonic_pairs": int(
            torch.count_nonzero(frontend.third_harmonic_matrix).item()
        ),
        "second_harmonic_scale": (
            None
            if not enabled
            else frontend.second_harmonic_scale.detach().cpu().tolist()
        ),
        "third_harmonic_scale": (
            None
            if not enabled
            else frontend.third_harmonic_scale.detach().cpu().tolist()
        ),
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


def save_history(history: list[dict[str, object]], path: Path) -> None:
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
    primary_score: str | None = None,
    condition_thresholds: dict[str, float] | None = None,
    fallback_threshold: float | None = None,
    anomaly_feature_memory: ConditionedFeatureMemory | None = None,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "protocol_version": 7,
            "model_state": model_state,
            "epoch": int(epoch),
            "validation_selection_loss": float(validation_loss),
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
            "anomaly_feature_memory": (
                None
                if anomaly_feature_memory is None
                else anomaly_feature_memory.to_dict()
            ),
            "spectral_profile": (
                None if spectral_profile is None else spectral_profile.to_dict()
            ),
            "threshold": threshold,
            "primary_score": primary_score,
            "condition_thresholds": condition_thresholds,
            "fallback_threshold": fallback_threshold,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    stage_epochs = args.filter_warmup_epochs + args.filter_adaptation_epochs
    if min(args.filter_warmup_epochs, args.filter_adaptation_epochs) < 0:
        raise ValueError("filter stage epochs must be non-negative.")
    if args.learnable_subband_weights and not args.evaluate_only:
        if args.filter_warmup_epochs <= 0 or args.filter_adaptation_epochs <= 0:
            raise ValueError(
                "learnable subband weights require positive warmup and adaptation "
                "epochs."
            )
        if stage_epochs >= args.epochs:
            raise ValueError(
                "filter warmup plus adaptation must leave at least one final "
                "fine-tuning epoch."
            )
    if not args.learnable_subband_weights and stage_epochs:
        raise ValueError(
            "filter stage epochs require --learnable-subband-weights."
        )
    if args.model != "egfn" and (
        args.memory_representation != "encoder"
        or args.objective_representation != "encoder"
    ):
        raise ValueError(
            "non-encoder memory/objective representations require --model egfn."
        )
    if (
        args.objective_representation == "memory"
        and args.memory_representation != "encoder"
        and args.gate_mode == "none"
    ):
        raise ValueError(
            "a fixed ungated spectral representation has no trainable path; use "
            "--objective-representation encoder or an active gate mode."
        )
    if args.training_mode == "hybrid":
        if args.model != "egfn":
            raise ValueError("hybrid training is currently available only for EGFN.")
        if args.supervised_weight <= 0:
            raise ValueError("hybrid training requires a positive --supervised-weight.")
        if args.ranking_weight < 0 or args.ranking_margin < 0:
            raise ValueError("ranking weight and margin must be non-negative.")
        if args.primary_score not in {"auto", "hybrid_score"}:
            raise ValueError("hybrid training requires --primary-score auto or hybrid_score.")
        if (
            args.anomaly_train_ratio <= 0
            or args.anomaly_validation_ratio <= 0
            or args.anomaly_train_ratio + args.anomaly_validation_ratio >= 1
        ):
            raise ValueError("hybrid anomaly split ratios are invalid.")
        if args.hybrid_max_validation_fpr is not None and not (
            0 <= args.hybrid_max_validation_fpr < 1
        ):
            raise ValueError("--hybrid-max-validation-fpr must be in [0, 1).")
        requested_primary_score = "hybrid_score"
        representation_primary_score = "memory_score"
    else:
        if args.ranking_weight:
            raise ValueError("--ranking-weight requires --training-mode hybrid.")
        if args.hybrid_max_validation_fpr is not None:
            raise ValueError(
                "--hybrid-max-validation-fpr requires --training-mode hybrid."
            )
        requested_primary_score = resolve_primary_score(args.model, args.primary_score)
        representation_primary_score = requested_primary_score
    if args.head_warmup_epochs < 0 or args.head_warmup_epochs > args.epochs:
        raise ValueError("--head-warmup-epochs must be between zero and --epochs.")
    if args.pretrained_one_class_checkpoint is not None:
        if args.training_mode != "hybrid" or args.evaluate_only:
            raise ValueError(
                "pretrained one-class transfer requires trainable hybrid mode."
            )
        if args.head_warmup_epochs <= 0:
            raise ValueError(
                "pretrained one-class transfer requires positive head warmup epochs."
            )
        if not args.pretrained_one_class_checkpoint.is_file():
            raise FileNotFoundError(args.pretrained_one_class_checkpoint)
    elif args.head_warmup_epochs:
        raise ValueError("head warmup requires --pretrained-one-class-checkpoint.")
    if args.head_warmup_epochs and args.learnable_subband_weights:
        raise ValueError("head warmup cannot be combined with filter adaptation.")
    if representation_primary_score == "memory_score":
        expected_objective = (
            "encoder" if args.memory_representation == "encoder" else "memory"
        )
        if args.objective_representation != expected_objective:
            raise ValueError(
                "memory_score must train and select the same representation: "
                f"use --objective-representation {expected_objective}."
            )
    if (
        representation_primary_score == "reconstruction_score"
        and args.reconstruction_weight <= 0
    ):
        raise ValueError(
            "reconstruction_score requires a positive --reconstruction-weight."
        )
    if args.variance_target <= 0 or min(
        args.invariance_weight,
        args.variance_weight,
        args.covariance_weight,
    ) < 0:
        raise ValueError("representation objective values are invalid.")
    if args.view_noise_fraction < 0 or not 0 <= args.view_max_shift_fraction < 1:
        raise ValueError("view augmentation values are invalid.")
    if args.dataset_format == "folders" and args.normal_dir is None:
        raise ValueError("--normal-dir is required with --dataset-format folders.")
    if args.dataset_format == "dcase2020" and args.dcase_dir is None:
        raise ValueError("--dcase-dir is required with --dataset-format dcase2020.")
    if args.egfn_embedding_channels <= 0:
        raise ValueError("--egfn-embedding-channels must be positive.")
    if len(args.conv1d_channels) != 3:
        raise ValueError("--conv1d-channels requires exactly three values.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    default_output_name = {
        "mimii": "mimii_one_class",
        "folders": f"{args.dataset_name}_one_class",
        "dcase2020": "dcase2020_one_class",
    }[args.dataset_format]
    if args.training_mode == "hybrid":
        default_output_name = default_output_name.replace("one_class", "hybrid")
    output_dir = args.output_dir or (
        ROOT / "outputs" / default_output_name / f"{args.model}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_checkpoint_path = args.checkpoint or output_dir / "best_checkpoint.pt"
    recovery_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset_format == "mimii":
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
    elif args.dataset_format == "folders":
        records = find_folder_anomaly_recordings(
            args.normal_dir,
            args.anomalous_dir,
            dataset_name=args.dataset_name,
            condition_mode=args.condition_mode,
        )
        split = split_anomaly_records(
            records,
            validation_normal_ratio=args.validation_normal_ratio,
            test_normal_ratio=args.test_normal_ratio,
            seed=args.seed,
        )
    else:
        split = find_dcase2020_development_split(
            args.dcase_dir,
            validation_normal_ratio=args.validation_normal_ratio,
            seed=args.seed,
        )
    validate_anomaly_split(split)
    if args.training_mode == "hybrid":
        split = add_hybrid_anomaly_partitions(
            split,
            anomaly_train_ratio=args.anomaly_train_ratio,
            anomaly_validation_ratio=args.anomaly_validation_ratio,
            seed=args.seed,
        )
        validate_hybrid_anomaly_split(split)
    train_records = cap_by_label(split.train, args.max_train_records)
    validation_records = cap_by_label(split.validation, args.max_eval_records_per_label)
    test_records = cap_by_label(split.test, args.max_eval_records_per_label)
    train_normal_records = tuple(record for record in train_records if record.is_normal)
    train_anomalous_records = tuple(
        record for record in train_records if not record.is_normal
    )
    validation_normal_records = tuple(
        record for record in validation_records if record.is_normal
    )
    validation_anomalous_records = tuple(
        record for record in validation_records if not record.is_normal
    )
    print(
        f"train_normal={len(train_normal_records)} "
        f"train_anomalous={len(train_anomalous_records)} "
        f"validation_normal={len(validation_normal_records)} "
        f"validation_anomalous={len(validation_anomalous_records)} "
        f"test_records={len(test_records)}"
    )

    if not train_normal_records or not validation_normal_records or not test_records:
        raise RuntimeError(
            "the split must contain training, validation, and test normal audio; "
            "provide at least three normal recordings per condition."
        )
    if args.training_mode == "hybrid" and (
        not train_anomalous_records or not validation_anomalous_records
    ):
        raise RuntimeError("hybrid training requires anomalous train and validation audio.")
    if args.dataset_format == "mimii":
        profile = ConditionedNormalProfile.load_json(args.normal_profile)
    else:
        profile = bootstrap_normal_profile(
            train_normal_records,
            args.sample_rate,
            args.n_fft,
            args.hop_length,
            args.macro_edges_hz,
            args.subbands_per_macro,
        )
    sample_rate = int(profile.metadata["frontend"]["sample_rate"])
    train_set = AnomalyWindowDataset(
        train_normal_records,
        target_sample_rate=sample_rate,
        duration_seconds=args.duration,
        crop_mode="random",
    )
    calibration_set = AnomalyWindowDataset(
        train_normal_records,
        target_sample_rate=sample_rate,
        duration_seconds=args.duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    validation_set = AnomalyWindowDataset(
        validation_normal_records,
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
    hybrid_train_loader = None
    hybrid_validation_loader = None
    hybrid_train_batch_sampler = None
    hybrid_positive_weight = 1.0
    if args.training_mode == "hybrid":
        hybrid_train_set = AnomalyWindowDataset(
            train_records,
            target_sample_rate=sample_rate,
            duration_seconds=args.duration,
            crop_mode="random",
        )
        hybrid_validation_set = AnomalyWindowDataset(
            validation_records,
            target_sample_rate=sample_rate,
            duration_seconds=args.duration,
            crop_mode="grid",
            evaluation_windows=args.evaluation_windows,
        )
        hybrid_train_batch_sampler = HybridConditionBatchSampler(
            hybrid_train_set,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        hybrid_train_loader = DataLoader(
            hybrid_train_set,
            batch_sampler=hybrid_train_batch_sampler,
            **loader_options,
        )
        hybrid_validation_loader = DataLoader(
            hybrid_validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options,
        )
        hybrid_positive_weight = 1.0
    calibration_loader = DataLoader(
        calibration_set,
        batch_size=args.batch_size,
        **loader_options,
    )
    train_batch_sampler = ConditionBatchSampler(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    if len(train_batch_sampler) == 0:
        raise RuntimeError(
            "no complete training batch can be formed within a condition; "
            "reduce --batch-size or add normal recordings."
        )
    validation_batch_sampler = ConditionBatchSampler(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_sampler=validation_batch_sampler,
        **loader_options,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_options)

    model = build_model(
        args.model,
        profile,
        egfn_embedding_channels=args.egfn_embedding_channels,
        conv1d_channels=args.conv1d_channels,
        learnable_subband_weights=args.learnable_subband_weights,
        gate_mode=args.gate_mode,
        normalize_gate_inputs=args.normalize_gate_inputs,
        conditional_subgates=args.conditional_subgates,
        harmonic_context=args.harmonic_context,
        supervised_anomaly_head=args.training_mode == "hybrid",
    ).to(device)
    if not args.evaluate_only and args.model == "egfn":
        profile_metadata = dict(profile.metadata)
        profile_metadata.update(
            {
                "source_split": "current_train",
                "source_recordings": len(train_normal_records),
                "split_seed": args.seed,
                "validation_normal_ratio": args.validation_normal_ratio,
                "test_normal_ratio": args.test_normal_ratio,
            }
        )
        model.normal_profile = estimate_spectral_profile(
            model,
            calibration_loader,
            device,
            profile_metadata,
        )
    if args.pretrained_one_class_checkpoint is not None:
        loaded_parameters = load_pretrained_one_class_backbone(
            model,
            args.pretrained_one_class_checkpoint,
        )
        model.to(device)
        print(
            f"loaded_one_class_backbone={args.pretrained_one_class_checkpoint} "
            f"shared_parameters={loaded_parameters}"
        )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"trainable_parameters={trainable_parameters}")
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    embedding_profiles: dict[str, ConditionedEmbeddingProfile] | None = None
    feature_memory: ConditionedFeatureMemory | None = None
    anomaly_feature_memory: ConditionedFeatureMemory | None = None
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
        if checkpoint.get("protocol_version") != 7:
            raise ValueError("checkpoint does not use the aligned one-class v7 protocol.")
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("model") != args.model:
            raise ValueError("checkpoint model does not match --model.")
        if int(checkpoint_args.get("seed", -1)) != args.seed:
            raise ValueError("checkpoint seed does not match --seed.")
        if bool(checkpoint_args.get("learnable_subband_weights", False)) != (
            args.learnable_subband_weights
        ):
            raise ValueError("checkpoint learnable-subband configuration does not match.")
        for name, legacy_default in (
            ("gate_mode", "hierarchical"),
            ("normalize_gate_inputs", False),
            ("conditional_subgates", False),
            ("harmonic_context", False),
            ("memory_representation", "encoder"),
            ("objective_representation", "encoder"),
            ("training_mode", "one_class"),
        ):
            if checkpoint_args.get(name, legacy_default) != getattr(args, name):
                raise ValueError(f"checkpoint {name} configuration does not match.")
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
        stored_anomaly_memory = checkpoint.get("anomaly_feature_memory")
        if stored_anomaly_memory:
            anomaly_feature_memory = ConditionedFeatureMemory.from_dict(
                stored_anomaly_memory
            ).to(device)
        best_epoch = int(checkpoint.get("epoch", 0))
        best_validation_loss = float(
            checkpoint.get("validation_selection_loss", float("nan"))
        )
        print(
            f"loaded_checkpoint={recovery_checkpoint_path} "
            f"epoch={best_epoch or 'unknown'}"
        )
    else:
        current_stage = training_stage(
            1,
            args.learnable_subband_weights,
            args.filter_warmup_epochs,
            args.filter_adaptation_epochs,
            args.head_warmup_epochs,
        )
        stage_trainable_parameters = configure_training_stage(
            model, args.model, current_stage
        )
        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        print(
            f"training_stage={current_stage} "
            f"stage_trainable_parameters={stage_trainable_parameters}"
        )
        early_stopping_reference = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            next_stage = training_stage(
                epoch,
                args.learnable_subband_weights,
                args.filter_warmup_epochs,
                args.filter_adaptation_epochs,
                args.head_warmup_epochs,
            )
            if next_stage != current_stage:
                if best_state is None:
                    raise RuntimeError(
                        f"training stage {current_stage} produced no checkpoint."
                    )
                model.load_state_dict(best_state)
                model.to(device)
                current_stage = next_stage
                stage_trainable_parameters = configure_training_stage(
                    model, args.model, current_stage
                )
                if args.model == "egfn":
                    stage_metadata = dict(profile_metadata)
                    stage_metadata["calibration_stage"] = current_stage
                    model.normal_profile = estimate_spectral_profile(
                        model,
                        calibration_loader,
                        device,
                        stage_metadata,
                    )
                optimizer = build_optimizer(
                    model, args.learning_rate, args.weight_decay
                )
                best_state = None
                best_epoch = 0
                best_validation_loss = float("inf")
                early_stopping_reference = float("inf")
                epochs_without_improvement = 0
                print(
                    f"training_stage={current_stage} "
                    f"stage_trainable_parameters={stage_trainable_parameters}"
                )
            train_batch_sampler.set_epoch(epoch)
            if hybrid_train_batch_sampler is not None:
                hybrid_train_batch_sampler.set_epoch(epoch)
            progress = (epoch - 1) / max(args.epochs - 1, 1)
            if current_stage == "head_warmup":
                train_metrics = {
                    "loss": 0.0,
                    "representation_loss": 0.0,
                    "embedding_std": 0.0,
                }
            else:
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
                    args.objective_representation,
                    args.memory_representation,
                )
            if args.training_mode == "hybrid":
                if hybrid_train_loader is None:
                    raise RuntimeError("hybrid training loader is unavailable.")
                supervised_train_metrics = run_supervised_epoch(
                        model,
                        hybrid_train_loader,
                        optimizer,
                        device,
                        hybrid_positive_weight,
                        args.supervised_weight,
                        args.ranking_weight,
                        args.ranking_margin,
                        args.view_noise_fraction,
                        args.view_max_shift_fraction,
                    )
                train_metrics.update(supervised_train_metrics)
                train_metrics["loss"] += (
                    args.supervised_weight
                    * supervised_train_metrics["supervised_loss"]
                )
            if current_stage == "filter_adaptation":
                stage_metadata = dict(profile_metadata)
                stage_metadata["calibration_stage"] = current_stage
                stage_metadata["calibration_epoch"] = epoch
                model.normal_profile = estimate_spectral_profile(
                    model,
                    calibration_loader,
                    device,
                    stage_metadata,
                )
            validation_metrics = evaluate_representation_epoch(
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
                args.seed + 10_000,
                args.objective_representation,
                args.memory_representation,
            )
            if args.training_mode == "hybrid":
                if hybrid_validation_loader is None:
                    raise RuntimeError("hybrid validation loader is unavailable.")
                supervised_validation = evaluate_supervised_epoch(
                    model,
                    hybrid_validation_loader,
                    device,
                    hybrid_positive_weight,
                    args.ranking_weight,
                    args.ranking_margin,
                )
                validation_metrics.update(supervised_validation)
                validation_metrics["hybrid_loss"] = (
                    validation_metrics["representation_loss"]
                    + args.supervised_weight
                    * supervised_validation["supervised_loss"]
                )
                selection_metric_name = "hybrid_loss"
            else:
                selection_metric_name = (
                    "reconstruction_loss"
                    if requested_primary_score == "reconstruction_score"
                    else "representation_loss"
                )
            validation_loss = validation_metrics[selection_metric_name]
            row = {
                "epoch": float(epoch),
                "stage": current_stage,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
                "validation_selection_loss": validation_loss,
            }
            history.append(row)
            save_history(history, output_dir / "history.csv")
            print(
                f"epoch={epoch:02d} stage={current_stage} "
                f"loss={train_metrics['loss']:.6f} "
                f"embedding_std={train_metrics['embedding_std']:.6f} "
                f"val_{selection_metric_name}={validation_loss:.6f}"
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
                    spectral_profile=(
                        model.normal_profile if args.model == "egfn" else None
                    ),
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
            final_stage = current_stage in {
                "representation",
                "representation_finetune",
            }
            if (
                final_stage
                and args.patience > 0
                and epochs_without_improvement >= args.patience
            ):
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
            calibration_loader,
            device,
            dict(profile_metadata),
        )
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    if embedding_profiles is None:
        embedding_profiles = estimate_embedding_profiles(
            model, args.model, calibration_loader, device
        )
    if feature_memory is None:
        feature_memory = estimate_feature_memory(
            model,
            args.model,
            calibration_loader,
            device,
            args.memory_size,
            args.memory_temporal_pool,
            args.memory_top_fraction,
            args.memory_query_chunk_size,
            args.seed,
            args.memory_representation,
        )
    if args.training_mode == "hybrid" and anomaly_feature_memory is None:
        if hybrid_train_loader is None:
            raise RuntimeError("hybrid anomaly memory requires a training loader.")
        anomaly_feature_memory = estimate_feature_memory(
            model,
            args.model,
            hybrid_train_loader,
            device,
            args.memory_size,
            args.memory_temporal_pool,
            args.memory_top_fraction,
            args.memory_query_chunk_size,
            args.seed + 10_000,
            args.memory_representation,
            label_filter=1,
        ).with_statistics_from(feature_memory).to(device)
    geometry, gate_diagnostics = estimate_model_diagnostics(
        model,
        args.model,
        calibration_loader,
        device,
        args.objective_representation,
        args.memory_representation,
    )

    score_validation_loader = (
        hybrid_validation_loader
        if args.training_mode == "hybrid"
        else validation_loader
    )
    if score_validation_loader is None:
        raise RuntimeError("score validation loader is unavailable.")
    validation_windows = collect_window_scores(
        model,
        args.model,
        score_validation_loader,
        embedding_profiles,
        feature_memory,
        device,
        args.mask_fraction,
        args.memory_top_fraction,
        args.memory_representation,
        anomaly_feature_memory,
    )
    validation_outputs = aggregate_score_outputs(
        validation_windows, args.recording_quantile
    )
    hybrid_calibration = None
    if args.training_mode == "hybrid":
        primary_score_name = "hybrid_score"
        hybrid_calibration = fit_hybrid_calibration(
            validation_outputs,
            "memory_score",
            args.normal_threshold_quantile,
            args.hybrid_max_validation_fpr,
        )
        validation_outputs["scores"]["primary_score"] = apply_hybrid_calibration(
            validation_outputs, hybrid_calibration
        )
        threshold = float(hybrid_calibration["threshold"])
        memory_calibration = hybrid_calibration["components"]["memory_score"]
        condition_thresholds = memory_calibration["condition_thresholds"]
        fallback_threshold = float(memory_calibration["fallback_threshold"])
    else:
        primary_score_name = resolve_primary_score(args.model, args.primary_score)
        if primary_score_name not in validation_outputs["scores"]:
            raise ValueError(
                f"primary score {primary_score_name!r} is unavailable for "
                f"model {args.model!r}."
            )
        condition_thresholds, fallback_threshold = fit_condition_thresholds(
            validation_outputs,
            primary_score_name,
            args.normal_threshold_quantile,
        )
        validation_outputs["scores"]["primary_score"] = apply_condition_thresholds(
            validation_outputs,
            primary_score_name,
            condition_thresholds,
            fallback_threshold,
        )
        threshold = 1.0
    test_windows = collect_window_scores(
        model,
        args.model,
        test_loader,
        embedding_profiles,
        feature_memory,
        device,
        args.mask_fraction,
        args.memory_top_fraction,
        args.memory_representation,
        anomaly_feature_memory,
    )
    test_outputs = aggregate_score_outputs(test_windows, args.recording_quantile)
    test_labels = test_outputs["labels"]
    raw_component_scores = dict(test_outputs["scores"])
    if hybrid_calibration is not None:
        test_outputs["scores"]["primary_score"] = apply_hybrid_calibration(
            test_outputs, hybrid_calibration
        )
    else:
        test_outputs["scores"]["primary_score"] = apply_condition_thresholds(
            test_outputs,
            primary_score_name,
            condition_thresholds,
            fallback_threshold,
        )
    test_scores = test_outputs["scores"]["primary_score"]
    supervised_metrics_available = bool(np.any(test_labels == 1))
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    condition_metrics, score_distributions = metrics_and_distributions_by_condition(
        test_outputs, threshold
    )
    result: dict[str, object] = {
        "model": args.model,
        "protocol": (
            "hybrid_disjoint_anomaly_v1"
            if args.training_mode == "hybrid"
            else "normal_only_aligned_representation_v7"
        ),
        "fit_labels": (
            ["normal", "anomalous"]
            if args.training_mode == "hybrid"
            else ["normal"]
        ),
        "supervised_metrics_available": supervised_metrics_available,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_validation_selection_loss": best_validation_loss,
        "embedding_profiles": {
            name: profile.to_dict() for name, profile in embedding_profiles.items()
        },
        "feature_memory": feature_memory.summary(),
        "anomaly_feature_memory": (
            None
            if anomaly_feature_memory is None
            else anomaly_feature_memory.summary()
        ),
        "frontend_filters": frontend_filter_summary(model),
        "harmonic_context": harmonic_context_summary(model),
        "representation_geometry": geometry,
        "gate_diagnostics": gate_diagnostics,
        "primary_score": primary_score_name,
        "threshold_source": (
            (
                f"hybrid_validation_fpr_cap_{args.hybrid_max_validation_fpr}"
                if args.hybrid_max_validation_fpr is not None
                else "hybrid_validation_f1"
            )
            if args.training_mode == "hybrid"
            else f"condition_validation_normal_quantile_{args.normal_threshold_quantile}"
        ),
        "hybrid_calibration": hybrid_calibration,
        "condition_thresholds": condition_thresholds,
        "fallback_threshold": fallback_threshold,
        "metrics": metrics,
        "component_auc": {
            name: metrics_at_threshold(test_labels, values, threshold)["auc"]
            for name, values in raw_component_scores.items()
        },
        "metrics_by_condition": condition_metrics,
        "score_distributions_by_condition": score_distributions,
        "split_counts": {
            "train_normal": len(train_normal_records),
            "train_anomalous": len(train_anomalous_records),
            "validation_normal": len(validation_normal_records),
            "validation_anomalous": len(validation_anomalous_records),
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
        primary_score_name,
        condition_thresholds,
        fallback_threshold,
        anomaly_feature_memory=anomaly_feature_memory,
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
            primary_score_name,
            condition_thresholds,
            fallback_threshold,
            anomaly_feature_memory=anomaly_feature_memory,
        )
    if supervised_metrics_available:
        print(
            f"test_auc={metrics['auc']:.4f} test_f1={metrics['f1']:.4f} "
            f"threshold={threshold:.6f}"
        )
    else:
        print(
            "supervised_test_metrics=unavailable reason=no_anomalous_test_audio "
            f"normal_false_positives={metrics['fp']} threshold={threshold:.6f}"
        )
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
