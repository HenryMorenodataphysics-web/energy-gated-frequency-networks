from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mimii_one_class import (  # noqa: E402
    aggregate_score_outputs,
    apply_condition_thresholds,
    build_model,
    estimate_feature_memory,
    fit_condition_thresholds,
    local_feature_map,
)
from src.anomaly import ConditionedNormalProfile  # noqa: E402
from src.data import (  # noqa: E402
    AnomalyWindowDataset,
    find_mimii_recordings,
    split_anomaly_records,
    to_anomaly_audio_record,
    validate_anomaly_split,
)
from src.utils.binary_evaluation import metrics_at_threshold  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalibrate and benchmark inference-only hard macro routing."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--memory-size", type=int, default=512)
    parser.add_argument("--memory-temporal-pool", type=int, default=4)
    parser.add_argument("--memory-top-fraction", type=float, default=0.05)
    parser.add_argument("--policies", default="soft,top2,top1")
    parser.add_argument("--benchmark-devices", default="cpu,cuda")
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-repetitions", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "mimii_hard_routing_v9",
    )
    return parser.parse_args()


def profile_from_checkpoint(checkpoint: dict[str, object]) -> ConditionedNormalProfile:
    payload = checkpoint.get("spectral_profile")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain a calibrated spectral profile.")
    return ConditionedNormalProfile(
        condition_ids=payload["condition_ids"],
        mean=torch.tensor(payload["mean"]),
        std=torch.tensor(payload["std"]),
        record_counts=torch.tensor(payload["record_counts"]),
        fallback_mean=torch.tensor(payload["fallback_mean"]),
        fallback_std=torch.tensor(payload["fallback_std"]),
        epsilon=float(payload["epsilon"]),
        metadata=payload.get("metadata", {}),
    )


def policy_top_k(policy: str) -> int | None:
    if policy == "soft":
        return None
    if policy.startswith("top") and policy[3:].isdigit():
        return int(policy[3:])
    raise ValueError(f"invalid routing policy: {policy}.")


def build_checkpoint_model(
    checkpoint: dict[str, object],
    device: torch.device,
    top_k: int | None,
) -> torch.nn.Module:
    checkpoint_args = checkpoint["args"]
    profile = profile_from_checkpoint(checkpoint).to(device)
    model = build_model(
        "egfn",
        profile,
        learnable_subband_weights=bool(
            checkpoint_args.get("learnable_subband_weights", False)
        ),
        gate_mode=str(checkpoint_args.get("gate_mode", "hierarchical")),
        normalize_gate_inputs=bool(
            checkpoint_args.get("normalize_gate_inputs", False)
        ),
        conditional_subgates=bool(
            checkpoint_args.get("conditional_subgates", False)
        ),
        harmonic_context=bool(checkpoint_args.get("harmonic_context", False)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    model.frontend.set_hard_routing_top_k(top_k)
    return model


def collect_memory_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_memory,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, float]]:
    recording_ids: list[str] = []
    condition_ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    active_macro = 0
    total_macro = 0
    active_subband = 0
    total_subband = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            conditions = list(batch["condition_id"])
            output = model(waveform, conditions, regularization_progress=1.0)
            memory_output = feature_memory.score(
                local_feature_map(output, "encoder"), conditions
            )
            recording_ids.extend(batch["recording_id"])
            condition_ids.extend(conditions)
            labels.extend(batch["label"].tolist())
            scores.extend(memory_output["recording_memory_score"].cpu().tolist())
            macro_mask = output["active_macro_mask"]
            subband_mask = output["active_subband_mask"]
            active_macro += int(macro_mask.sum().item())
            total_macro += macro_mask.numel()
            active_subband += int(subband_mask.sum().item())
            total_subband += subband_mask.numel()
    return (
        {
            "recording_ids": recording_ids,
            "condition_ids": condition_ids,
            "labels": np.asarray(labels),
            "scores": {"memory_score": np.asarray(scores)},
        },
        {
            "active_macro_fraction": active_macro / total_macro,
            "active_subband_fraction": active_subband / total_subband,
        },
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_samples(function, device: torch.device, warmup: int, repetitions: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter()
            function()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1_000.0)
    return samples


def latency_summary(samples: list[float], batch_size: int) -> dict[str, float]:
    p50 = float(np.quantile(samples, 0.50))
    p95 = float(np.quantile(samples, 0.95))
    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "throughput_examples_per_second": batch_size / (statistics.mean(samples) / 1_000.0),
    }


def benchmark_policy(
    checkpoint: dict[str, object],
    waveform: torch.Tensor,
    conditions: list[str],
    policy: str,
    requested_device: str,
    warmup: int,
    repetitions: int,
) -> dict[str, object] | None:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return None
    device = torch.device(requested_device)
    model = build_checkpoint_model(checkpoint, device, policy_top_k(policy))
    local_waveform = waveform.to(device)
    frontend_samples = latency_samples(
        lambda: model.frontend(local_waveform), device, warmup, repetitions
    )
    model_samples = latency_samples(
        lambda: model(local_waveform, conditions), device, warmup, repetitions
    )
    return {
        "device": requested_device,
        "policy": policy,
        "batch_size": int(local_waveform.shape[0]),
        "frontend": latency_summary(frontend_samples, local_waveform.shape[0]),
        "end_to_end": latency_summary(model_samples, local_waveform.shape[0]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.memory_size <= 0:
        raise ValueError("batch size and memory size must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("model") != "egfn":
        raise ValueError("checkpoint must contain an EGFN model.")
    if checkpoint_args.get("gate_mode") != "macro":
        raise ValueError("hard routing benchmark requires a macro-gated checkpoint.")

    seed = int(checkpoint_args["seed"])
    machine_type = str(checkpoint_args.get("machine_type", "valve"))
    machine_id = str(checkpoint_args.get("machine_id", "all"))
    snr = str(checkpoint_args.get("snr", "all"))
    duration = float(checkpoint_args.get("duration", 2.0))
    recording_quantile = float(checkpoint_args.get("recording_quantile", 0.95))
    threshold_quantile = float(
        checkpoint_args.get("normal_threshold_quantile", 0.95)
    )
    validation_ratio = float(checkpoint_args.get("validation_normal_ratio", 0.15))
    test_ratio = float(checkpoint_args.get("test_normal_ratio", 0.15))
    sample_rate = int(
        profile_from_checkpoint(checkpoint).metadata["frontend"]["sample_rate"]
    )

    records = [
        to_anomaly_audio_record(record)
        for record in find_mimii_recordings(
            args.data_dir,
            machine_type=machine_type,
            machine_id=machine_id,
            snr=snr,
        )
    ]
    split = split_anomaly_records(
        records,
        validation_normal_ratio=validation_ratio,
        test_normal_ratio=test_ratio,
        seed=seed,
    )
    validate_anomaly_split(split)
    calibration_set = AnomalyWindowDataset(
        split.train,
        target_sample_rate=sample_rate,
        duration_seconds=duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    validation_set = AnomalyWindowDataset(
        split.validation,
        target_sample_rate=sample_rate,
        duration_seconds=duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    test_set = AnomalyWindowDataset(
        split.test,
        target_sample_rate=sample_rate,
        duration_seconds=duration,
        crop_mode="grid",
        evaluation_windows=args.evaluation_windows,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda",
    }
    calibration_loader = DataLoader(calibration_set, **loader_options)
    validation_loader = DataLoader(validation_set, **loader_options)
    test_loader = DataLoader(test_set, **loader_options)
    device = torch.device(args.device)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    quality_rows = []
    full_results: dict[str, object] = {"quality": {}, "benchmark": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        print(f"quality_policy={policy}", flush=True)
        model = build_checkpoint_model(checkpoint, device, policy_top_k(policy))
        memory = estimate_feature_memory(
            model,
            "egfn",
            calibration_loader,
            device,
            args.memory_size,
            args.memory_temporal_pool,
            args.memory_top_fraction,
            2048,
            seed,
            "encoder",
        )
        validation_windows, _ = collect_memory_scores(
            model, validation_loader, memory, device
        )
        test_windows, routing = collect_memory_scores(
            model, test_loader, memory, device
        )
        validation = aggregate_score_outputs(validation_windows, recording_quantile)
        test = aggregate_score_outputs(test_windows, recording_quantile)
        thresholds, fallback = fit_condition_thresholds(
            validation, "memory_score", threshold_quantile
        )
        calibrated = apply_condition_thresholds(
            test, "memory_score", thresholds, fallback
        )
        metrics = metrics_at_threshold(test["labels"], calibrated, 1.0)
        row = {
            "policy": policy,
            **routing,
            **{name: metrics[name] for name in ("auc", "accuracy", "precision", "recall", "f1")},
        }
        quality_rows.append(row)
        full_results["quality"][policy] = {
            "metrics": metrics,
            "routing": routing,
            "condition_thresholds": thresholds,
            "fallback_threshold": fallback,
        }
        write_csv(args.output_dir / "hard_routing_quality.csv", quality_rows)
        (args.output_dir / "hard_routing_results.json").write_text(
            json.dumps(full_results, indent=2), encoding="utf-8"
        )
        print(
            f"policy={policy} auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
            f"active_macro={routing['active_macro_fraction']:.3f}",
            flush=True,
        )

    benchmark_batch = next(iter(test_loader))
    waveform = benchmark_batch["waveform"]
    conditions = list(benchmark_batch["condition_id"])
    benchmark_devices = [
        item.strip() for item in args.benchmark_devices.split(",") if item.strip()
    ]
    for requested_device in benchmark_devices:
        for policy in policies:
            print(f"benchmark_device={requested_device} policy={policy}", flush=True)
            result = benchmark_policy(
                checkpoint,
                waveform,
                conditions,
                policy,
                requested_device,
                args.benchmark_warmup,
                args.benchmark_repetitions,
            )
            if result is not None:
                full_results["benchmark"].append(result)

    (args.output_dir / "hard_routing_results.json").write_text(
        json.dumps(full_results, indent=2), encoding="utf-8"
    )
    flattened_benchmark = []
    for result in full_results["benchmark"]:
        flattened_benchmark.append(
            {
                "device": result["device"],
                "policy": result["policy"],
                "batch_size": result["batch_size"],
                **{f"frontend_{key}": value for key, value in result["frontend"].items()},
                **{f"end_to_end_{key}": value for key, value in result["end_to_end"].items()},
            }
        )
    if flattened_benchmark:
        write_csv(args.output_dir / "hard_routing_benchmark.csv", flattened_benchmark)
    print(f"saved={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
