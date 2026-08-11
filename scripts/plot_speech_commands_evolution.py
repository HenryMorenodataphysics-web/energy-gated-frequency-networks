from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "Conv1D": {
        "history": ROOT
        / "outputs"
        / "speech_commands_conv1d"
        / "metrics"
        / "history.csv",
        "diagnostics": ROOT
        / "outputs"
        / "speech_commands_conv1d"
        / "metrics"
        / "test_diagnostics.pt",
        "parameters": 27_594,
        "test_gate_mean": np.nan,
        "color": "#3A9D5D",
    },
    "Temporal EGFN": {
        "history": ROOT
        / "outputs"
        / "speech_commands_egfn_temporal"
        / "metrics"
        / "history.csv",
        "diagnostics": ROOT
        / "outputs"
        / "speech_commands_egfn_temporal"
        / "metrics"
        / "test_diagnostics.pt",
        "parameters": 44_714,
        "test_gate_mean": 0.547,
        "color": "#2878B5",
    },
    "Temporal EGFN Wide": {
        "history": ROOT
        / "outputs"
        / "speech_commands_egfn_temporal_wide"
        / "metrics"
        / "history.csv",
        "diagnostics": ROOT
        / "outputs"
        / "speech_commands_egfn_temporal_wide"
        / "metrics"
        / "test_diagnostics.pt",
        "parameters": 189_802,
        "test_gate_mean": 0.444,
        "color": "#E07B39",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Speech Commands evolution of Temporal EGFN."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "speech_commands_evolution",
    )
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "train_acc", "val_acc", "train_gate", "val_gate")
    }


def read_diagnostics(path: Path) -> dict[str, np.ndarray | float | list[str]]:
    diagnostics = torch.load(path, map_location="cpu", weights_only=True)
    confusion = diagnostics["confusion"].float()
    normalized = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1)
    labels = [str(label) for label in diagnostics["labels"]]
    return {
        "test_acc": float(diagnostics["test_acc"]),
        "confusion": confusion.numpy(),
        "normalized_confusion": normalized.numpy(),
        "labels": labels,
    }


def save_summary(
    output_path: Path,
    histories: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[str, np.ndarray | float | list[str]]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "trainable_parameters",
                "best_validation_accuracy",
                "best_epoch",
                "final_train_accuracy",
                "final_validation_accuracy",
                "test_accuracy",
                "test_gate_mean",
                "test_examples",
            ]
        )
        for model_name, spec in MODEL_SPECS.items():
            history = histories[model_name]
            result = diagnostics[model_name]
            best_index = int(np.argmax(history["val_acc"]))
            writer.writerow(
                [
                    model_name,
                    spec["parameters"],
                    history["val_acc"][best_index],
                    history["epoch"][best_index],
                    history["train_acc"][-1],
                    history["val_acc"][-1],
                    result["test_acc"],
                    spec["test_gate_mean"],
                    int(np.asarray(result["confusion"]).sum()),
                ]
            )


def annotate_confusion(axis: plt.Axes, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value >= 0.15:
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="black" if value >= 0.65 else "white",
                )


def plot_comparison(
    output_dir: Path,
    histories: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[str, np.ndarray | float | list[str]]],
) -> None:
    figure = plt.figure(figsize=(15.0, 8.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.92, 1.08))
    curve_axis = figure.add_subplot(grid[0, :2])
    bar_axis = figure.add_subplot(grid[0, 2])

    for model_name, spec in MODEL_SPECS.items():
        history = histories[model_name]
        best_index = int(np.argmax(history["val_acc"]))
        color = str(spec["color"])
        curve_axis.plot(
            history["epoch"],
            history["val_acc"],
            label=model_name,
            color=color,
            linewidth=1.9,
        )
        curve_axis.scatter(
            history["epoch"][best_index],
            history["val_acc"][best_index],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    curve_axis.set_title("(a) Official validation split", loc="left", fontweight="bold")
    curve_axis.set_xlabel("Epoch")
    curve_axis.set_ylabel("Validation accuracy")
    curve_axis.set_ylim(0.55, 0.93)
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=9, ncols=3, loc="lower right")

    model_names = list(MODEL_SPECS)
    test_accuracies = [float(diagnostics[name]["test_acc"]) for name in model_names]
    colors = [str(MODEL_SPECS[name]["color"]) for name in model_names]
    bars = bar_axis.bar(model_names, test_accuracies, color=colors, width=0.68)
    for bar, model_name, accuracy in zip(bars, model_names, test_accuracies):
        parameters = int(MODEL_SPECS[model_name]["parameters"])
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.006,
            f"{accuracy:.3f}\n{parameters / 1000:.1f}k params",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    bar_axis.set_title("(b) Official test split", loc="left", fontweight="bold")
    bar_axis.set_ylabel("Test accuracy")
    bar_axis.set_ylim(0.0, 0.96)
    bar_axis.tick_params(axis="x", labelrotation=17)
    bar_axis.grid(axis="y", alpha=0.25)

    confusion_image = None
    panel_letters = ("c", "d", "e")
    for column, (model_name, panel_letter) in enumerate(
        zip(model_names, panel_letters)
    ):
        axis = figure.add_subplot(grid[1, column])
        result = diagnostics[model_name]
        normalized = np.asarray(result["normalized_confusion"])
        labels = list(result["labels"])
        confusion_image = axis.imshow(
            normalized,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        annotate_confusion(axis, normalized)
        axis.set_title(
            f"({panel_letter}) {model_name} test confusion",
            loc="left",
            fontweight="bold",
        )
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted command")
        axis.set_ylabel("True command")

    if confusion_image is not None:
        figure.colorbar(
            confusion_image,
            ax=figure.axes[-3:],
            pad=0.015,
            shrink=0.88,
            label="Row-normalized rate",
        )
    figure.savefig(output_dir / "speech_commands_evolution.png", dpi=300)
    figure.savefig(output_dir / "speech_commands_evolution.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = {
        model_name: read_history(Path(spec["history"]))
        for model_name, spec in MODEL_SPECS.items()
    }
    diagnostics = {
        model_name: read_diagnostics(Path(spec["diagnostics"]))
        for model_name, spec in MODEL_SPECS.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(
        args.output_dir / "speech_commands_evolution_summary.csv",
        histories,
        diagnostics,
    )
    plot_comparison(args.output_dir, histories, diagnostics)

    for model_name, spec in MODEL_SPECS.items():
        history = histories[model_name]
        result = diagnostics[model_name]
        print(
            f"model={model_name} parameters={spec['parameters']} "
            f"best_val={history['val_acc'].max():.4f} "
            f"test_acc={float(result['test_acc']):.4f} "
            f"test_gate_mean={spec['test_gate_mean']}"
        )
    print(f"saved_figure={args.output_dir / 'speech_commands_evolution.png'}")
    print(f"saved_pdf={args.output_dir / 'speech_commands_evolution.pdf'}")


if __name__ == "__main__":
    main()
