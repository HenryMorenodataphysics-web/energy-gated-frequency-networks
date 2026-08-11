from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "conv1d_matched": {
        "label": "Matched Conv1D",
        "parameters": 188_754,
        "color": "#3A9D5D",
        "run_dir": ROOT
        / "outputs"
        / "mimii_v2_controlled"
        / "runs"
        / "conv1d_matched"
        / "seed_42",
    },
    "egfn_free": {
        "label": "EGFN-Free",
        "parameters": 188_770,
        "color": "#E07B39",
        "run_dir": ROOT / "outputs" / "mimii_valve_egfn_temporal_wide",
    },
    "egfn_sinc": {
        "label": "EGFN-Sinc",
        "parameters": 187_978,
        "color": "#7A5195",
        "run_dir": ROOT
        / "outputs"
        / "mimii_v2_controlled"
        / "runs"
        / "egfn_sinc"
        / "seed_42",
    },
}
INITIAL_BANDS = [
    ("< 250 Hz", 0.0, 250.0),
    ("250-500 Hz", 250.0, 500.0),
    ("500-750 Hz", 500.0, 750.0),
    ("750-1000 Hz", 750.0, 1000.0),
    ("1000-1500 Hz", 1000.0, 1500.0),
    ("1500-2500 Hz", 1500.0, 2500.0),
    ("2500-4000 Hz", 2500.0, 4000.0),
    ("> 4000 Hz", 4000.0, 8000.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the capacity-matched MIMII V2 screening experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "mimii_v2_controlled",
    )
    return parser.parse_args()


def read_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        "target": np.asarray([int(row["target"]) for row in rows]),
        "score": np.asarray([float(row["anomaly_probability"]) for row in rows]),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_summary(
    output_path: Path,
    reports: dict[str, dict[str, object]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "experiment",
                "trainable_parameters",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "false_positives",
                "false_negatives",
            ]
        )
        for model, spec in MODEL_SPECS.items():
            metrics = reports[model]["test_calibrated"]
            writer.writerow(
                [
                    model,
                    spec["parameters"],
                    metrics["threshold"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["auc"],
                    metrics["fp"],
                    metrics["fn"],
                ]
            )


def plot_initial_intervals(axis: plt.Axes) -> None:
    for position, (_, low, high) in enumerate(INITIAL_BANDS):
        axis.hlines(position, low, high, color="0.78", linewidth=7, zorder=1)


def style_filter_axis(axis: plt.Axes, title: str) -> None:
    labels = [band[0] for band in INITIAL_BANDS]
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlim(0, 8000)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Initialized band")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()


def plot_controlled(
    output_dir: Path,
    reports: dict[str, dict[str, object]],
    predictions: dict[str, dict[str, np.ndarray]],
    free_filters: list[dict[str, str]],
    sinc_filters: list[dict[str, str]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True)
    model_names = list(MODEL_SPECS)
    labels = [str(MODEL_SPECS[name]["label"]) for name in model_names]
    colors = [str(MODEL_SPECS[name]["color"]) for name in model_names]

    parameters = np.asarray([int(MODEL_SPECS[name]["parameters"]) for name in model_names])
    bars = axes[0, 0].bar(labels, parameters, color=colors, width=0.68)
    for bar, value in zip(bars, parameters):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2500,
            f"{value:,}",
            ha="center",
            fontsize=8,
        )
    axes[0, 0].set_title("(a) Capacity-matched architectures", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Trainable parameters")
    axes[0, 0].set_ylim(0, 210_000)
    axes[0, 0].tick_params(axis="x", labelrotation=18)
    axes[0, 0].grid(axis="y", alpha=0.25)

    for model, spec in MODEL_SPECS.items():
        values = predictions[model]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            values["target"], values["score"]
        )
        auc_value = reports[model]["test_calibrated"]["auc"]
        axes[0, 1].plot(
            false_positive_rate,
            true_positive_rate,
            color=str(spec["color"]),
            linewidth=1.8,
            label=f"{spec['label']} ({auc_value:.3f})",
        )
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    axes[0, 1].set_title("(b) Test ROC", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("False-positive rate")
    axes[0, 1].set_ylabel("True-positive rate")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, loc="lower right")

    metric_names = ("accuracy", "precision", "recall", "f1", "auc")
    metric_labels = ("Accuracy", "Precision", "Recall", "F1", "AUC")
    positions = np.arange(len(metric_names))
    width = 0.25
    for index, (model, spec) in enumerate(MODEL_SPECS.items()):
        metrics = reports[model]["test_calibrated"]
        axes[0, 2].bar(
            positions + (index - 1) * width,
            [metrics[name] for name in metric_names],
            width,
            color=str(spec["color"]),
            label=str(spec["label"]),
        )
    axes[0, 2].set_title("(c) Calibrated test metrics", loc="left", fontweight="bold")
    axes[0, 2].set_xticks(positions, metric_labels, rotation=25, ha="right")
    axes[0, 2].set_ylabel("Score")
    axes[0, 2].set_ylim(0.60, 1.01)
    axes[0, 2].grid(axis="y", alpha=0.25)
    axes[0, 2].legend(fontsize=7.5, loc="lower left")

    error_positions = np.arange(len(model_names))
    false_positives = [reports[name]["test_calibrated"]["fp"] for name in model_names]
    false_negatives = [reports[name]["test_calibrated"]["fn"] for name in model_names]
    axes[1, 0].bar(
        error_positions - 0.19,
        false_positives,
        0.38,
        color="#4C78A8",
        label="False positives",
    )
    axes[1, 0].bar(
        error_positions + 0.19,
        false_negatives,
        0.38,
        color="#C44E52",
        label="False negatives",
    )
    axes[1, 0].set_title("(d) Calibrated test errors", loc="left", fontweight="bold")
    axes[1, 0].set_xticks(error_positions, labels, rotation=18, ha="right")
    axes[1, 0].set_ylabel("Number of recordings")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    plot_initial_intervals(axes[1, 1])
    for position, row in enumerate(free_filters):
        axes[1, 1].scatter(
            float(row["learned_peak_hz"]),
            position,
            color=str(MODEL_SPECS["egfn_free"]["color"]),
            marker="x",
            s=55,
            linewidth=2,
            zorder=2,
        )
    axes[1, 1].plot([], [], color="0.78", linewidth=7, label="Initialized interval")
    axes[1, 1].scatter([], [], color=str(MODEL_SPECS["egfn_free"]["color"]), marker="x", label="Learned spectral peak")
    style_filter_axis(axes[1, 1], "(e) Free FIR kernels lose band constraint")
    axes[1, 1].legend(fontsize=8, loc="lower right")

    plot_initial_intervals(axes[1, 2])
    for position, row in enumerate(sinc_filters):
        low = float(row["learned_low_hz"])
        high = float(row["learned_high_hz"])
        axes[1, 2].hlines(
            position,
            low,
            high,
            color=str(MODEL_SPECS["egfn_sinc"]["color"]),
            linewidth=4,
            zorder=2,
        )
        axes[1, 2].scatter(
            [low, high],
            [position, position],
            color=str(MODEL_SPECS["egfn_sinc"]["color"]),
            s=18,
            zorder=3,
        )
    axes[1, 2].plot([], [], color="0.78", linewidth=7, label="Initialized interval")
    axes[1, 2].plot([], [], color=str(MODEL_SPECS["egfn_sinc"]["color"]), linewidth=4, label="Learned Sinc passband")
    style_filter_axis(axes[1, 2], "(f) Sinc preserves explicit passbands")
    axes[1, 2].legend(fontsize=8, loc="lower right")

    figure.savefig(output_dir / "mimii_v2_controlled.png", dpi=300)
    figure.savefig(output_dir / "mimii_v2_controlled.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    reports: dict[str, dict[str, object]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for model, spec in MODEL_SPECS.items():
        run_dir = Path(spec["run_dir"])
        reports[model] = read_report(run_dir / "analysis" / "threshold_report.json")
        predictions[model] = read_predictions(
            run_dir / "analysis" / "test_predictions.csv"
        )
    free_filters = read_rows(
        Path(MODEL_SPECS["egfn_free"]["run_dir"])
        / "analysis"
        / "learned_filter_statistics.csv"
    )
    sinc_filters = read_rows(
        Path(MODEL_SPECS["egfn_sinc"]["run_dir"])
        / "analysis"
        / "learned_filter_statistics.csv"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(args.output_dir / "mimii_v2_controlled_summary.csv", reports)
    plot_controlled(output_dir=args.output_dir, reports=reports, predictions=predictions, free_filters=free_filters, sinc_filters=sinc_filters)

    for model, spec in MODEL_SPECS.items():
        metrics = reports[model]["test_calibrated"]
        print(
            f"model={spec['label']} parameters={spec['parameters']} "
            f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f}"
        )
    print(f"saved_figure={args.output_dir / 'mimii_v2_controlled.png'}")
    print(f"saved_pdf={args.output_dir / 'mimii_v2_controlled.pdf'}")


if __name__ == "__main__":
    main()
