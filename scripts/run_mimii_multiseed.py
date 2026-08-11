from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MODEL_CONFIGS = {
    "conv1d": [
        "--model", "conv1d",
        "--augment",
        "--balanced-loss",
    ],
    "egfn_temporal_wide": [
        "--model", "egfn_temporal",
        "--filter-bank", "fine",
        "--learnable-filters",
        "--gate-mode", "independent",
        "--frontend-channels", "32",
        "--temporal-channels", "64,128",
        "--dropout", "0.25",
        "--label-smoothing", "0.05",
        "--scheduler", "cosine",
        "--augment",
        "--balanced-loss",
        "--patience", "10",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and summarize matched-seed Conv1D vs EGFN MIMII experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument(
        "--models",
        default="conv1d,egfn_temporal_wide",
        help="Comma-separated subset of: conv1d, egfn_temporal_wide.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mimii_valve_multiseed")
    parser.add_argument("--force", action="store_true", help="Rerun completed training and analysis.")
    parser.add_argument(
        "--no-reuse-seed-42",
        action="store_true",
        help="Do not reuse the original seed-42 valve runs.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only aggregate existing reports; do not launch training or analysis.",
    )
    return parser.parse_args()


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def legacy_seed_42_dir(model_name: str) -> Path:
    if model_name == "conv1d":
        return ROOT / "outputs" / "mimii_valve_conv1d"
    return ROOT / "outputs" / "mimii_valve_egfn_temporal_wide"


def run_dir_for(args: argparse.Namespace, model_name: str, seed: int) -> Path:
    legacy = legacy_seed_42_dir(model_name)
    if seed == 42 and not args.no_reuse_seed_42 and legacy.exists():
        return legacy
    return args.output_dir / "runs" / model_name / f"seed_{seed}"


def run_command(command: list[str]) -> None:
    print("running=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_run(args: argparse.Namespace, model_name: str, seed: int, run_dir: Path) -> None:
    diagnostics = run_dir / "metrics" / "test_diagnostics.pt"
    report = run_dir / "analysis" / "threshold_report.json"
    if args.summarize_only:
        return

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
            *MODEL_CONFIGS[model_name],
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


def load_rows(model_name: str, seed: int, run_dir: Path) -> list[dict]:
    report_path = run_dir / "analysis" / "threshold_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path}. Run without --summarize-only to complete it."
        )
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    rows = []
    for operating_point, key in (("default", "test_default"), ("calibrated", "test_calibrated")):
        metrics = report[key]
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "operating_point": operating_point,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    summary = []
    models = sorted({row["model"] for row in rows})
    operating_points = ("default", "calibrated")
    metrics = ("threshold", "accuracy", "precision", "recall", "f1", "auc")
    for model_name in models:
        for operating_point in operating_points:
            group = [
                row for row in rows
                if row["model"] == model_name and row["operating_point"] == operating_point
            ]
            result = {
                "model": model_name,
                "operating_point": operating_point,
                "n_seeds": len(group),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(result)
    return summary


def main() -> None:
    args = parse_args()
    seeds = parse_csv_list(args.seeds, int)
    models = parse_csv_list(args.models)
    unknown = set(models) - set(MODEL_CONFIGS)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if len(seeds) < 3:
        print("warning=fewer than three seeds gives a weak variance estimate", flush=True)

    rows = []
    for seed in seeds:
        for model_name in models:
            run_dir = run_dir_for(args, model_name, seed)
            print(f"experiment model={model_name} seed={seed} run_dir={run_dir}", flush=True)
            ensure_run(args, model_name, seed, run_dir)
            rows.extend(load_rows(model_name, seed, run_dir))

    summary = aggregate(rows)
    write_csv(args.output_dir / "multiseed_runs.csv", rows)
    write_csv(args.output_dir / "multiseed_summary.csv", summary)
    with (args.output_dir / "multiseed_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    for row in summary:
        print(
            f"model={row['model']} point={row['operating_point']} n={row['n_seeds']} "
            f"acc={row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f} "
            f"f1={row['f1_mean']:.3f}+/-{row['f1_std']:.3f} "
            f"auc={row['auc_mean']:.3f}+/-{row['auc_std']:.3f}"
        )
    print(f"saved_summary={args.output_dir}")


if __name__ == "__main__":
    main()
