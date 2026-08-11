from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CONTROLLED_CONFIGS = {
    "conv1d_matched": [
        "--model", "conv1d_matched",
    ],
    "egfn_sinc": [
        "--model", "egfn_temporal",
        "--filter-mode", "sinc",
        "--gate-mode", "independent",
    ],
}

GATING_ABLATION_CONFIGS = {
    "filterbank_free_nogate": [
        "--model", "egfn_temporal",
        "--filter-mode", "free",
        "--gate-mode", "none",
    ],
    "egfn_free_gated": [
        "--model", "egfn_temporal",
        "--filter-mode", "free",
        "--gate-mode", "independent",
    ],
    "filterbank_sinc_nogate": [
        "--model", "egfn_temporal",
        "--filter-mode", "sinc",
        "--gate-mode", "none",
    ],
}

SHARED_CONFIG = [
    "--filter-bank", "fine",
    "--frontend-channels", "32",
    "--temporal-channels", "64,128",
    "--dropout", "0.25",
    "--label-smoothing", "0.05",
    "--scheduler", "cosine",
    "--augment",
    "--balanced-loss",
    "--patience", "10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the capacity-matched MIMII EGFN V2 experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mimii_v2_controlled")
    parser.add_argument(
        "--include-gating-ablation",
        action="store_true",
        help="Also run matched free/Sinc frontends with and without energy gating.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def free_egfn_run(seed: int) -> Path:
    if seed == 42:
        return ROOT / "outputs" / "mimii_valve_egfn_temporal_wide"
    return ROOT / "outputs" / "mimii_valve_multiseed" / "runs" / "egfn_temporal_wide" / f"seed_{seed}"


def run_command(command: list[str]) -> None:
    print("running=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_controlled_run(
    args: argparse.Namespace,
    experiment: str,
    seed: int,
    configs: dict[str, list[str]],
) -> Path:
    run_dir = args.output_dir / "runs" / experiment / f"seed_{seed}"
    diagnostics = run_dir / "metrics" / "test_diagnostics.pt"
    report = run_dir / "analysis" / "threshold_report.json"
    if args.force or not diagnostics.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_mimii.py"),
            "--data-dir", str(args.data_dir),
            "--machine-type", args.machine_type,
            "--epochs", str(args.epochs),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--seed", str(seed),
            "--output-dir", str(run_dir),
            *SHARED_CONFIG,
            *configs[experiment],
        ]
        run_command(command)
    else:
        print(f"skip_training completed={run_dir}", flush=True)

    if args.force or not report.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_mimii_results.py"),
            "--run-dir", str(run_dir),
            "--data-dir", str(args.data_dir),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ]
        run_command(command)
    else:
        print(f"skip_analysis completed={run_dir}", flush=True)
    return run_dir


def load_rows(experiment: str, seed: int, run_dir: Path) -> list[dict]:
    report_path = run_dir / "analysis" / "threshold_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing analysis report: {report_path}")
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    rows = []
    for point, key in (("default", "test_default"), ("calibrated", "test_calibrated")):
        metrics = report[key]
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "operating_point": point,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    summary = []
    metrics = ("threshold", "accuracy", "precision", "recall", "f1", "auc")
    for experiment in sorted({row["experiment"] for row in rows}):
        for point in ("default", "calibrated"):
            group = [
                row for row in rows
                if row["experiment"] == experiment and row["operating_point"] == point
            ]
            result = {
                "experiment": experiment,
                "operating_point": point,
                "n_seeds": len(group),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    configs = dict(CONTROLLED_CONFIGS)
    if args.include_gating_ablation:
        configs.update(GATING_ABLATION_CONFIGS)
    rows = []
    for seed in seeds:
        free_run = free_egfn_run(seed)
        print(f"reference experiment=egfn_free seed={seed} run_dir={free_run}", flush=True)
        rows.extend(load_rows("egfn_free", seed, free_run))
        for experiment in configs:
            print(f"experiment={experiment} seed={seed}", flush=True)
            run_dir = ensure_controlled_run(args, experiment, seed, configs)
            rows.extend(load_rows(experiment, seed, run_dir))

    summary = aggregate(rows)
    write_csv(args.output_dir / "v2_runs.csv", rows)
    write_csv(args.output_dir / "v2_summary.csv", summary)
    with (args.output_dir / "v2_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    for row in summary:
        print(
            f"experiment={row['experiment']} point={row['operating_point']} n={row['n_seeds']} "
            f"acc={row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f} "
            f"f1={row['f1_mean']:.3f}+/-{row['f1_std']:.3f} "
            f"auc={row['auc_mean']:.3f}+/-{row['auc_std']:.3f}"
        )
    print(f"saved_summary={args.output_dir}")


if __name__ == "__main__":
    main()
