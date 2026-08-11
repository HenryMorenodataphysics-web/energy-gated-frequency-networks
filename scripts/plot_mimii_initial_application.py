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
    "Conv1D": {
        "run_dir": ROOT / "outputs" / "mimii_valve_conv1d",
        "color": "#3A9D5D",
    },
    "Temporal EGFN Wide": {
        "run_dir": ROOT / "outputs" / "mimii_valve_egfn_temporal_wide",
        "color": "#E07B39",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the first MIMII valve application of EGFN."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "mimii_initial_application",
    )
    return parser.parse_args()


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
        if key != "path" and key != "machine_id"
    }


def read_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_band_statistics(path: Path) -> list[dict[str, str]]:
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
                "model",
                "operating_point",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
        for model_name, report in reports.items():
            for operating_point in ("test_default", "test_calibrated"):
                metrics = report[operating_point]
                writer.writerow(
                    [
                        model_name,
                        operating_point.removeprefix("test_"),
                        metrics["threshold"],
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        metrics["auc"],
                        metrics["tn"],
                        metrics["fp"],
                        metrics["fn"],
                        metrics["tp"],
                    ]
                )


def plot_grouped_bars(
    axis: plt.Axes,
    labels: list[str],
    first: np.ndarray,
    second: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    positions = np.arange(len(labels))
    width = 0.38
    axis.bar(positions - width / 2, first, width, label="Normal", color="#2878B5")
    axis.bar(positions + width / 2, second, width, label="Abnormal", color="#C44E52")
    axis.set_xticks(positions, labels, rotation=32, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)


def plot_application(
    output_dir: Path,
    histories: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    reports: dict[str, dict[str, object]],
    band_rows: list[dict[str, str]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True)

    for model_name, spec in MODEL_SPECS.items():
        history = histories[model_name]
        best_index = int(np.argmax(history["val_acc"]))
        color = str(spec["color"])
        axes[0, 0].plot(
            history["epoch"],
            history["val_acc"],
            color=color,
            linewidth=1.8,
            label=model_name,
        )
        axes[0, 0].scatter(
            history["epoch"][best_index],
            history["val_acc"][best_index],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axes[0, 0].set_title("(a) Validation on MIMII valve", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation accuracy")
    axes[0, 0].set_ylim(0.10, 0.97)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="lower right")

    for model_name, spec in MODEL_SPECS.items():
        values = predictions[model_name]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            values["target"], values["anomaly_probability"]
        )
        auc_value = reports[model_name]["test_default"]["auc"]
        axes[0, 1].plot(
            false_positive_rate,
            true_positive_rate,
            color=str(spec["color"]),
            linewidth=1.9,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    axes[0, 1].set_title("(b) Test ROC", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("False-positive rate")
    axes[0, 1].set_ylabel("True-positive rate")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, loc="lower right")

    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    positions = np.arange(len(metric_names))
    width = 0.38
    for index, (model_name, spec) in enumerate(MODEL_SPECS.items()):
        metrics = reports[model_name]["test_calibrated"]
        values = [metrics[name] for name in metric_names]
        axes[0, 2].bar(
            positions + (index - 0.5) * width,
            values,
            width,
            color=str(spec["color"]),
            label=model_name,
        )
    axes[0, 2].set_title("(c) Calibrated test metrics", loc="left", fontweight="bold")
    axes[0, 2].set_xticks(positions, metric_labels, rotation=25, ha="right")
    axes[0, 2].set_ylabel("Score")
    axes[0, 2].set_ylim(0.65, 1.01)
    axes[0, 2].grid(axis="y", alpha=0.25)
    axes[0, 2].legend(fontsize=8, loc="lower left")

    operating_points = ["Default", "Calibrated"]
    positions = np.arange(len(operating_points))
    for index, (model_name, spec) in enumerate(MODEL_SPECS.items()):
        f1_values = [
            reports[model_name]["test_default"]["f1"],
            reports[model_name]["test_calibrated"]["f1"],
        ]
        bars = axes[1, 0].bar(
            positions + (index - 0.5) * width,
            f1_values,
            width,
            color=str(spec["color"]),
            label=model_name,
        )
        threshold = reports[model_name]["test_calibrated"]["threshold"]
        axes[1, 0].text(
            bars[1].get_x() + bars[1].get_width() / 2,
            f1_values[1] + 0.012,
            f"t={threshold:.2f}",
            ha="center",
            fontsize=8,
        )
    axes[1, 0].set_title("(d) Effect of validation-selected threshold", loc="left", fontweight="bold")
    axes[1, 0].set_xticks(positions, operating_points)
    axes[1, 0].set_ylabel("Test F1")
    axes[1, 0].set_ylim(0.0, 0.92)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8, loc="lower left")

    band_labels = [row["band"] for row in band_rows]
    normal_gates = np.asarray([float(row["normal_gate_mean"]) for row in band_rows])
    abnormal_gates = np.asarray([float(row["abnormal_gate_mean"]) for row in band_rows])
    plot_grouped_bars(
        axes[1, 1],
        band_labels,
        normal_gates,
        abnormal_gates,
        "Mean gate",
        "(e) Gates by initialized band",
    )
    axes[1, 1].set_ylim(0.48, 0.58)

    normal_energy = np.asarray(
        [float(row["normal_log_energy_mean"]) for row in band_rows]
    )
    abnormal_energy = np.asarray(
        [float(row["abnormal_log_energy_mean"]) for row in band_rows]
    )
    plot_grouped_bars(
        axes[1, 2],
        band_labels,
        normal_energy,
        abnormal_energy,
        "Mean log-energy",
        "(f) Energy by initialized band",
    )

    figure.savefig(output_dir / "mimii_initial_application.png", dpi=300)
    figure.savefig(output_dir / "mimii_initial_application.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories: dict[str, dict[str, np.ndarray]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, dict[str, object]] = {}
    for model_name, spec in MODEL_SPECS.items():
        run_dir = Path(spec["run_dir"])
        histories[model_name] = read_numeric_csv(run_dir / "metrics" / "history.csv")
        predictions[model_name] = read_numeric_csv(
            run_dir / "analysis" / "test_predictions.csv"
        )
        reports[model_name] = read_report(
            run_dir / "analysis" / "threshold_report.json"
        )
    band_rows = read_band_statistics(
        Path(MODEL_SPECS["Temporal EGFN Wide"]["run_dir"])
        / "analysis"
        / "band_statistics.csv"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(args.output_dir / "mimii_initial_metrics.csv", reports)
    plot_application(args.output_dir, histories, predictions, reports, band_rows)

    for model_name in MODEL_SPECS:
        metrics = reports[model_name]["test_calibrated"]
        print(
            f"model={model_name} threshold={metrics['threshold']:.2f} "
            f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f}"
        )
    print(f"saved_figure={args.output_dir / 'mimii_initial_application.png'}")
    print(f"saved_pdf={args.output_dir / 'mimii_initial_application.pdf'}")


if __name__ == "__main__":
    main()
