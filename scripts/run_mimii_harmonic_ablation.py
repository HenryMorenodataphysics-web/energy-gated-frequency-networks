from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the fixed macro EGFN with sparse 2x/3x harmonic context."
    )
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mimii")
    parser.add_argument("--normal-profile", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-windows", type=int, default=5)
    parser.add_argument("--memory-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "mimii_harmonic_ablation_v10",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_result(payload: dict[str, object], harmonic: bool) -> None:
    if payload.get("protocol") != "normal_only_aligned_representation_v7":
        raise ValueError("result does not use the aligned normal-only v7 protocol.")
    if payload.get("fit_labels") != ["normal"]:
        raise ValueError("result was not fitted exclusively on normal audio.")
    arguments = payload.get("args", {})
    expected = {
        "model": "egfn",
        "gate_mode": "macro",
        "normalize_gate_inputs": True,
        "learnable_subband_weights": False,
        "harmonic_context": harmonic,
        "primary_score": "memory_score",
        "memory_representation": "encoder",
        "objective_representation": "encoder",
        "gate_regularization_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    mismatches = {
        key: (arguments.get(key, False if key == "harmonic_context" else None), value)
        for key, value in expected.items()
        if arguments.get(key, False if key == "harmonic_context" else None) != value
    }
    if mismatches:
        raise ValueError(f"result violates the harmonic ablation contract: {mismatches}")


def row(name: str, payload: dict[str, object], path: Path) -> dict[str, object]:
    metrics = payload["metrics"]
    return {
        "variant": name,
        "seed": int(payload["args"]["seed"]),
        "trainable_parameters": int(payload["trainable_parameters"]),
        "auc": float(metrics["auc"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "result_path": str(path),
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    validate_result(baseline, harmonic=False)
    if int(baseline["args"]["seed"]) != args.seed:
        raise ValueError("baseline seed does not match --seed.")

    run_dir = args.output_dir / f"harmonic_seed{args.seed}"
    result_path = run_dir / "results.json"
    if args.force or not result_path.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_mimii_one_class.py"),
            "--model", "egfn",
            "--data-dir", str(args.data_dir),
            "--normal-profile", str(args.normal_profile),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--evaluation-windows", str(args.evaluation_windows),
            "--memory-size", str(args.memory_size),
            "--gate-mode", "macro",
            "--normalize-gate-inputs",
            "--harmonic-context",
            "--primary-score", "memory_score",
            "--objective-representation", "encoder",
            "--memory-representation", "encoder",
            "--gate-regularization-weight", "0.0",
            "--reconstruction-weight", "0.0",
            "--covariance-weight", "1.0",
            "--seed", str(args.seed),
            "--output-dir", str(run_dir),
        ]
        print("running=" + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print(f"skip_completed={result_path}", flush=True)

    harmonic = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result(harmonic, harmonic=True)
    rows = [
        row("macro_fixed", baseline, args.baseline_results),
        row("macro_harmonic", harmonic, result_path),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "harmonic_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "harmonic_ablation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(
        f"baseline_auc={rows[0]['auc']:.4f} harmonic_auc={rows[1]['auc']:.4f} "
        f"baseline_f1={rows[0]['f1']:.4f} harmonic_f1={rows[1]['f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
