from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE_MODES = ("none", "macro", "subband", "hierarchical")
CONDITIONAL_MODE = "hierarchical_conditional"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a normal-only causal ablation of hierarchical spectral gating."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--machine-id", default="all")
    parser.add_argument("--snr", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--experiments",
        default=",".join(GATE_MODES),
        help=(
            "Comma-separated experiments from none, macro, subband, hierarchical, "
            "or hierarchical_conditional."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--memory-size", type=int, default=512)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records-per-label", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "mimii_hierarchical_gating_ablation_v7",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-conditional",
        action="store_true",
        help=(
            "Run hierarchical_conditional as a separate fifth experiment instead "
            "of confounding it with the routing ablation."
        ),
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def parse_experiments(value: str) -> list[str]:
    experiments = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {*GATE_MODES, CONDITIONAL_MODE}
    invalid = sorted(set(experiments) - allowed)
    if not experiments or invalid:
        raise ValueError(
            "--experiments must contain only " + ", ".join(sorted(allowed)) + "."
        )
    if len(set(experiments)) != len(experiments):
        raise ValueError("--experiments must not contain duplicates.")
    return experiments


def run_mode(args: argparse.Namespace, experiment: str, seed: int) -> Path:
    gate_mode = "hierarchical" if experiment == CONDITIONAL_MODE else experiment
    run_dir = args.output_dir / "runs" / experiment / f"seed_{seed}"
    result_path = run_dir / "results.json"
    if result_path.exists() and not args.force:
        print(f"skip_completed={result_path}", flush=True)
        return result_path
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_mimii_one_class.py"),
        "--model", "egfn",
        "--data-dir", str(args.data_dir),
        "--machine-type", args.machine_type,
        "--machine-id", args.machine_id,
        "--snr", args.snr,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--evaluation-windows", str(args.evaluation_windows),
        "--memory-size", str(args.memory_size),
        "--seed", str(seed),
        "--gate-mode", gate_mode,
        "--normalize-gate-inputs",
        "--memory-representation", "encoder",
        "--objective-representation", "encoder",
        "--primary-score", "memory_score",
        "--gate-regularization-weight", "0.0",
        "--reconstruction-weight", "0.0",
        "--covariance-weight", "1.0",
        "--output-dir", str(run_dir),
    ]
    if experiment == CONDITIONAL_MODE:
        command.append("--conditional-subgates")
    if args.max_train_records is not None:
        command.extend(("--max-train-records", str(args.max_train_records)))
    if args.max_eval_records_per_label is not None:
        command.extend(
            ("--max-eval-records-per-label", str(args.max_eval_records_per_label))
        )
    print("running=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return result_path


def validate_payload(
    payload: dict[str, object], source: str, mode: str
) -> None:
    if payload.get("fit_labels") != ["normal"]:
        raise ValueError(f"run {source} was not fitted exclusively on normal audio.")
    if payload.get("protocol") != "normal_only_aligned_representation_v7":
        raise ValueError(f"run {source} does not use the aligned v7 protocol.")
    run_args = payload.get("args", {})
    if not isinstance(run_args, dict):
        raise ValueError(f"run {source} has invalid serialized arguments.")
    expected_gate_mode = "hierarchical" if mode == CONDITIONAL_MODE else mode
    expected_conditional = mode == CONDITIONAL_MODE
    expected = {
        "gate_mode": expected_gate_mode,
        "conditional_subgates": expected_conditional,
        "memory_representation": "encoder",
        "objective_representation": "encoder",
        "gate_regularization_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    mismatches = {
        name: (run_args.get(name), value)
        for name, value in expected.items()
        if run_args.get(name) != value
    }
    if mismatches:
        raise ValueError(f"run {source} violates the causal ablation: {mismatches}")


def load_row(path: Path, mode: str, seed: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_payload(payload, str(path), mode)
    metrics = payload["metrics"]
    return {
        "gate_mode": mode,
        "seed": seed,
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "run_dir": str(path.parent),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiments = parse_experiments(args.experiments)
    if args.include_conditional:
        if CONDITIONAL_MODE not in experiments:
            experiments.append(CONDITIONAL_MODE)
    rows = [
        load_row(run_mode(args, mode, seed), mode, seed)
        for seed in parse_seeds(args.seeds)
        for mode in experiments
    ]
    summary = []
    for mode in experiments:
        selected = [row for row in rows if row["gate_mode"] == mode]
        summary.append(
            {
                "gate_mode": mode,
                "n_seeds": len(selected),
                **{
                    f"{metric}_{statistic}": function(
                        [float(row[metric]) for row in selected]
                    )
                    for metric in ("auc", "accuracy", "precision", "recall", "f1")
                    for statistic, function in (
                        ("mean", statistics.mean),
                        (
                            "std",
                            lambda values: statistics.stdev(values)
                            if len(values) > 1
                            else 0.0,
                        ),
                    )
                },
            }
        )
    write_csv(args.output_dir / "gating_ablation_runs.csv", rows)
    write_csv(args.output_dir / "gating_ablation_summary.csv", summary)
    (args.output_dir / "gating_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(
            f"gate_mode={row['gate_mode']} auc={row['auc_mean']:.4f} "
            f"f1={row['f1_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
