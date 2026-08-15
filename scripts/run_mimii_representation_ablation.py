from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("encoder", "gated_profile", "activation_signature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aligned encoder, profile-evidence, and activation-signature "
            "memories without learnable spectral filters."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--seeds", default="42")
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
        default=ROOT / "outputs" / "mimii_representation_ablation_v7",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def run_representation(
    args: argparse.Namespace, representation: str, seed: int
) -> Path:
    run_dir = args.output_dir / "runs" / representation / f"seed_{seed}"
    result_path = run_dir / "results.json"
    if result_path.exists() and not args.force:
        print(f"skip_completed={result_path}", flush=True)
        return result_path
    objective = "encoder" if representation == "encoder" else "memory"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_mimii_one_class.py"),
        "--model", "egfn",
        "--data-dir", str(args.data_dir),
        "--machine-type", args.machine_type,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--evaluation-windows", str(args.evaluation_windows),
        "--memory-size", str(args.memory_size),
        "--seed", str(seed),
        "--gate-mode", "hierarchical",
        "--normalize-gate-inputs",
        "--memory-representation", representation,
        "--objective-representation", objective,
        "--primary-score", "memory_score",
        "--gate-regularization-weight", "0.0",
        "--reconstruction-weight", "0.0",
        "--covariance-weight", "1.0",
        "--output-dir", str(run_dir),
    ]
    if args.max_train_records is not None:
        command.extend(("--max-train-records", str(args.max_train_records)))
    if args.max_eval_records_per_label is not None:
        command.extend(
            ("--max-eval-records-per-label", str(args.max_eval_records_per_label))
        )
    print("running=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return result_path


def load_row(path: Path, representation: str, seed: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_args = payload.get("args", {})
    expected_objective = "encoder" if representation == "encoder" else "memory"
    if (
        payload.get("protocol") != "normal_only_aligned_representation_v7"
        or payload.get("fit_labels") != ["normal"]
        or run_args.get("memory_representation") != representation
        or run_args.get("objective_representation") != expected_objective
        or run_args.get("primary_score") != "memory_score"
    ):
        raise ValueError(f"run {path} does not match the aligned representation test.")
    metrics = payload["metrics"]
    geometry = payload["representation_geometry"]
    return {
        "representation": representation,
        "seed": seed,
        "auc": metrics["auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "effective_rank": geometry["effective_rank"],
        "mean_absolute_correlation": geometry[
            "mean_absolute_off_diagonal_correlation"
        ],
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
    rows = [
        load_row(
            run_representation(args, representation, seed),
            representation,
            seed,
        )
        for seed in parse_seeds(args.seeds)
        for representation in REPRESENTATIONS
    ]
    summary = []
    for representation in REPRESENTATIONS:
        selected = [row for row in rows if row["representation"] == representation]
        summary.append(
            {
                "representation": representation,
                "n_seeds": len(selected),
                **{
                    f"{metric}_{statistic}": function(
                        [float(row[metric]) for row in selected]
                    )
                    for metric in (
                        "auc",
                        "precision",
                        "recall",
                        "f1",
                        "effective_rank",
                        "mean_absolute_correlation",
                    )
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
    write_csv(args.output_dir / "representation_ablation_runs.csv", rows)
    write_csv(args.output_dir / "representation_ablation_summary.csv", summary)
    (args.output_dir / "representation_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(
            f"representation={row['representation']} "
            f"auc={row['auc_mean']:.4f} f1={row['f1_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
