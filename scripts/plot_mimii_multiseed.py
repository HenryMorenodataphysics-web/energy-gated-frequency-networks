from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "outputs" / "mimii_valve_multiseed" / "multiseed_runs.csv"
MODEL_SPECS = {
    "conv1d": {
        "label": "Conv1D",
        "parameters": 27_594,
        "color": "#3A9D5D",
        "marker": "o",
    },
    "egfn_temporal_wide": {
        "label": "Temporal EGFN Wide",
        "parameters": 189_802,
        "color": "#E07B39",
        "marker": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the three-seed MIMII valve comparison."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "mimii_multiseed",
    )
    return parser.parse_args()


def read_runs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_rows(
    rows: list[dict[str, str]], model: str, operating_point: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["model"] == model and row["operating_point"] == operating_point
    ]
    return sorted(selected, key=lambda row: int(row["seed"]))


def values_by_model(
    rows: list[dict[str, str]], metric: str, operating_point: str = "calibrated"
) -> dict[str, np.ndarray]:
    return {
        model: np.asarray(
            [float(row[metric]) for row in select_rows(rows, model, operating_point)]
        )
        for model in MODEL_SPECS
    }


def seeds(rows: list[dict[str, str]]) -> np.ndarray:
    first_model = next(iter(MODEL_SPECS))
    return np.asarray(
        [int(row["seed"]) for row in select_rows(rows, first_model, "calibrated")]
    )


def save_summary(output_path: Path, rows: list[dict[str, str]]) -> None:
    metrics = ("accuracy", "precision", "recall", "f1", "auc", "threshold")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "trainable_parameters",
                "n_seeds",
                *[item for metric in metrics for item in (f"{metric}_mean", f"{metric}_std")],
            ]
        )
        for model, spec in MODEL_SPECS.items():
            model_rows = select_rows(rows, model, "calibrated")
            summary: list[float] = []
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in model_rows])
                summary.extend([float(values.mean()), float(values.std(ddof=1))])
            writer.writerow(
                [model, spec["parameters"], len(model_rows), *summary]
            )


def plot_seed_metric(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
) -> None:
    seed_values = seeds(rows)
    for model, spec in MODEL_SPECS.items():
        metric_values = values_by_model(rows, metric)[model]
        axis.plot(
            seed_values,
            metric_values,
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            linewidth=1.7,
            markersize=6,
            label=str(spec["label"]),
        )
    axis.set_xticks(seed_values, [str(value) for value in seed_values])
    axis.set_xlabel("Training seed")
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")


def plot_multiseed(output_dir: Path, rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.8, 8.5), constrained_layout=True)

    plot_seed_metric(
        axes[0, 0],
        rows,
        "accuracy",
        "Calibrated accuracy",
        "(a) Accuracy across seeds",
        (0.90, 1.0),
    )
    plot_seed_metric(
        axes[0, 1],
        rows,
        "f1",
        "Calibrated F1",
        "(b) F1 across seeds",
        (0.62, 0.97),
    )
    plot_seed_metric(
        axes[0, 2],
        rows,
        "auc",
        "AUC",
        "(c) Threshold-free AUC",
        (0.94, 1.0),
    )

    metrics = ("accuracy", "precision", "recall", "f1", "auc")
    labels = ("Accuracy", "Precision", "Recall", "F1", "AUC")
    positions = np.arange(len(metrics))
    width = 0.38
    for index, (model, spec) in enumerate(MODEL_SPECS.items()):
        means = []
        stds = []
        for metric in metrics:
            values = values_by_model(rows, metric)[model]
            means.append(values.mean())
            stds.append(values.std(ddof=1))
        axes[1, 0].bar(
            positions + (index - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=str(spec["color"]),
            label=str(spec["label"]),
        )
    axes[1, 0].set_title("(d) Calibrated mean $\\pm$ SD", loc="left", fontweight="bold")
    axes[1, 0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0.60, 1.02)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8, loc="lower left")

    seed_values = seeds(rows)
    for model, spec in MODEL_SPECS.items():
        thresholds = values_by_model(rows, "threshold")[model]
        axes[1, 1].plot(
            seed_values,
            thresholds,
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            linewidth=1.7,
            markersize=6,
            label=str(spec["label"]),
        )
    axes[1, 1].set_title("(e) Validation-selected threshold", loc="left", fontweight="bold")
    axes[1, 1].set_xticks(seed_values, [str(value) for value in seed_values])
    axes[1, 1].set_xlabel("Training seed")
    axes[1, 1].set_ylabel("Threshold")
    axes[1, 1].set_ylim(0.15, 0.85)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8, loc="lower right")

    for model, spec in MODEL_SPECS.items():
        f1_values = values_by_model(rows, "f1")[model]
        parameters = int(spec["parameters"])
        axes[1, 2].errorbar(
            parameters,
            f1_values.mean(),
            yerr=f1_values.std(ddof=1),
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            markersize=8,
            capsize=4,
            linestyle="none",
        )
        axes[1, 2].annotate(
            f"{spec['label']}\n{parameters / 1000:.1f}k params",
            (parameters, f1_values.mean()),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 2].set_title("(f) Performance-capacity trade-off", loc="left", fontweight="bold")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel("Trainable parameters (log scale)")
    axes[1, 2].set_ylabel("Calibrated F1, mean $\\pm$ SD")
    axes[1, 2].set_xlim(18_000, 320_000)
    axes[1, 2].set_ylim(0.68, 0.96)
    axes[1, 2].grid(alpha=0.25, which="both")

    figure.savefig(output_dir / "mimii_multiseed.png", dpi=300)
    figure.savefig(output_dir / "mimii_multiseed.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = read_runs(RUNS_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(args.output_dir / "mimii_multiseed_paper_summary.csv", rows)
    plot_multiseed(args.output_dir, rows)

    for model, spec in MODEL_SPECS.items():
        f1_values = values_by_model(rows, "f1")[model]
        auc_values = values_by_model(rows, "auc")[model]
        accuracy_values = values_by_model(rows, "accuracy")[model]
        print(
            f"model={spec['label']} n={len(f1_values)} "
            f"accuracy={accuracy_values.mean():.4f}+/-{accuracy_values.std(ddof=1):.4f} "
            f"f1={f1_values.mean():.4f}+/-{f1_values.std(ddof=1):.4f} "
            f"auc={auc_values.mean():.4f}+/-{auc_values.std(ddof=1):.4f}"
        )
    print(f"saved_figure={args.output_dir / 'mimii_multiseed.png'}")
    print(f"saved_pdf={args.output_dir / 'mimii_multiseed.pdf'}")


if __name__ == "__main__":
    main()
