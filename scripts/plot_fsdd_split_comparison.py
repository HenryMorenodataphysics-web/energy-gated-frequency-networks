from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DIGIT_LABELS = [str(digit) for digit in range(10)]
SPLIT_SPECS = {
    "Random split": {
        "history": ROOT / "outputs" / "fsdd_egfn_random" / "metrics" / "history.csv",
        "diagnostics": ROOT
        / "outputs"
        / "fsdd_egfn_random"
        / "metrics"
        / "test_diagnostics.pt",
        "description": "Recordings mixed across speakers",
    },
    "Speaker-disjoint split": {
        "history": ROOT / "outputs" / "fsdd_egfn" / "metrics" / "history.csv",
        "diagnostics": ROOT
        / "outputs"
        / "fsdd_egfn"
        / "metrics"
        / "test_diagnostics.pt",
        "description": "Unseen speaker in the test set",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EGFN on random and speaker-disjoint FSDD splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "fsdd_split_comparison",
    )
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "train_acc", "val_acc", "train_loss", "val_loss")
    }


def read_diagnostics(path: Path) -> dict[str, np.ndarray | float]:
    diagnostics = torch.load(path, map_location="cpu", weights_only=True)
    confusion = diagnostics["confusion"].float()
    normalized = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1)
    return {
        "test_acc": float(diagnostics["test_acc"]),
        "confusion": confusion.numpy(),
        "normalized_confusion": normalized.numpy(),
    }


def save_summary(
    output_path: Path,
    histories: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[str, np.ndarray | float]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "split",
                "best_validation_accuracy",
                "best_epoch",
                "final_train_accuracy",
                "final_validation_accuracy",
                "test_accuracy",
                "test_examples",
            ]
        )
        for split_name in SPLIT_SPECS:
            history = histories[split_name]
            result = diagnostics[split_name]
            best_index = int(np.argmax(history["val_acc"]))
            writer.writerow(
                [
                    split_name,
                    history["val_acc"][best_index],
                    history["epoch"][best_index],
                    history["train_acc"][-1],
                    history["val_acc"][-1],
                    result["test_acc"],
                    int(np.asarray(result["confusion"]).sum()),
                ]
            )


def annotate_confusion(axis: plt.Axes, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value >= 0.10:
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if value >= 0.65 else "white",
                )


def plot_comparison(
    output_dir: Path,
    histories: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[str, np.ndarray | float]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 9.0), constrained_layout=True)
    split_names = list(SPLIT_SPECS)

    for column, split_name in enumerate(split_names):
        history = histories[split_name]
        best_index = int(np.argmax(history["val_acc"]))
        axis = axes[0, column]
        axis.plot(history["epoch"], history["train_acc"], label="Train", linewidth=1.6)
        axis.plot(history["epoch"], history["val_acc"], label="Validation", linewidth=1.6)
        axis.scatter(
            history["epoch"][best_index],
            history["val_acc"][best_index],
            color="C1",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            f"best = {history['val_acc'][best_index]:.3f}",
            (history["epoch"][best_index], history["val_acc"][best_index]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )
        panel_letter = "a" if column == 0 else "b"
        axis.set_title(
            f"({panel_letter}) {split_name}", loc="left", fontweight="bold"
        )
        axis.text(
            0.02,
            0.04,
            str(SPLIT_SPECS[split_name]["description"]),
            transform=axis.transAxes,
            fontsize=8,
            color="0.3",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.0, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")

    confusion_image = None
    for column, split_name in enumerate(split_names):
        result = diagnostics[split_name]
        normalized = np.asarray(result["normalized_confusion"])
        axis = axes[1, column]
        confusion_image = axis.imshow(
            normalized,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        annotate_confusion(axis, normalized)
        panel_letter = "c" if column == 0 else "d"
        axis.set_title(
            f"({panel_letter}) Test confusion\nAccuracy = {float(result['test_acc']):.3f}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xticks(range(10), DIGIT_LABELS)
        axis.set_yticks(range(10), DIGIT_LABELS)
        axis.set_xlabel("Predicted digit")
        axis.set_ylabel("True digit")

    if confusion_image is not None:
        figure.colorbar(
            confusion_image,
            ax=axes[1, :],
            pad=0.02,
            label="Row-normalized rate",
        )
    figure.savefig(output_dir / "fsdd_random_vs_speaker_split.png", dpi=300)
    figure.savefig(output_dir / "fsdd_random_vs_speaker_split.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = {
        split_name: read_history(Path(spec["history"]))
        for split_name, spec in SPLIT_SPECS.items()
    }
    diagnostics = {
        split_name: read_diagnostics(Path(spec["diagnostics"]))
        for split_name, spec in SPLIT_SPECS.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(args.output_dir / "fsdd_split_summary.csv", histories, diagnostics)
    plot_comparison(args.output_dir, histories, diagnostics)

    for split_name in SPLIT_SPECS:
        history = histories[split_name]
        result = diagnostics[split_name]
        print(
            f"split={split_name} best_val={history['val_acc'].max():.4f} "
            f"test_acc={float(result['test_acc']):.4f}"
        )
    print(f"saved_figure={args.output_dir / 'fsdd_random_vs_speaker_split.png'}")
    print(f"saved_pdf={args.output_dir / 'fsdd_random_vs_speaker_split.pdf'}")


if __name__ == "__main__":
    main()
