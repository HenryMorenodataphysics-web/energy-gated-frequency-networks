from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.models import FrequencyGatedClassifier, get_filter_bands
from src.utils.signal_utils import SyntheticConfig, SyntheticFrequencyDataset


CLASS_NAMES = ["low", "mid", "high", "low + high"]
BAND_LABELS = [
    "<250 Hz",
    "250-500 Hz",
    "500-750 Hz",
    "750-1000 Hz",
    "1-1.5 kHz",
    "1.5-2.5 kHz",
    "2.5-4 kHz",
    ">4 kHz",
]
MODEL_SPECS = {
    "Independent": {
        "checkpoint": ROOT
        / "outputs"
        / "hard_learnable"
        / "models"
        / "frequency_gated_synthetic_best.pt",
        "history": ROOT
        / "outputs"
        / "hard_learnable"
        / "metrics"
        / "synthetic_history.csv",
        "gate_mode": "independent",
    },
    "Contextual": {
        "checkpoint": ROOT
        / "outputs"
        / "hard_contextual"
        / "models"
        / "frequency_gated_synthetic_best.pt",
        "history": ROOT
        / "outputs"
        / "hard_contextual"
        / "metrics"
        / "synthetic_history.csv",
        "gate_mode": "contextual",
    },
    "Sparse ($\\lambda=0.05$)": {
        "checkpoint": ROOT
        / "outputs"
        / "hard_sparse_gate_l005"
        / "models"
        / "frequency_gated_synthetic_best.pt",
        "history": ROOT
        / "outputs"
        / "hard_sparse_gate_l005"
        / "metrics"
        / "synthetic_history.csv",
        "gate_mode": "independent",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent, contextual, and sparsity-regularized gates."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "gate_evolution",
    )
    parser.add_argument("--num-samples", type=int, default=640)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        "epoch": np.asarray([float(row["epoch"]) for row in rows]),
        "val_acc": np.asarray([float(row["val_acc"]) for row in rows]),
    }


def load_model(spec: dict[str, Path | str], sample_rate: int) -> FrequencyGatedClassifier:
    checkpoint = Path(spec["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gate_mode = str(spec["gate_mode"])
    gate_hidden_dim = None
    if gate_mode == "contextual":
        gate_hidden_dim = int(state["frontend.context_gate.0.weight"].shape[0])
    model = FrequencyGatedClassifier(
        num_classes=int(state["classifier.3.weight"].shape[0]),
        sample_rate=sample_rate,
        hidden_dim=int(state["classifier.0.weight"].shape[0]),
        learnable_filters=True,
        filter_bands=get_filter_bands("fine"),
        gate_mode=gate_mode,
        gate_hidden_dim=gate_hidden_dim,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_validation_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SyntheticFrequencyDataset(
        SyntheticConfig(
            num_samples=args.num_samples,
            sample_rate=args.sample_rate,
            duration=args.duration,
            snr_db=args.snr_db,
            difficulty="hard",
            seed=args.dataset_seed,
        )
    )
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    _, validation_set = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    return DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)


def evaluate_model(
    model: FrequencyGatedClassifier,
    loader: DataLoader,
) -> dict[str, np.ndarray | float]:
    gate_batches = []
    label_batches = []
    prediction_batches = []
    with torch.no_grad():
        for waveforms, labels in loader:
            outputs = model(waveforms)
            gate_batches.append(outputs["gates"].cpu())
            label_batches.append(labels.cpu())
            prediction_batches.append(outputs["logits"].argmax(dim=1).cpu())
    gates = torch.cat(gate_batches).numpy()
    labels = torch.cat(label_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    class_gate_means = np.zeros((len(CLASS_NAMES), len(BAND_LABELS)))
    for class_index in range(len(CLASS_NAMES)):
        class_gate_means[class_index] = gates[labels == class_index].mean(axis=0)
    return {
        "gates": gates,
        "labels": labels,
        "predictions": predictions,
        "accuracy": float((labels == predictions).mean()),
        "class_gate_means": class_gate_means,
    }


def save_measurements(
    output_dir: Path,
    results: dict[str, dict[str, np.ndarray | float]],
    thresholds: np.ndarray,
) -> None:
    with (output_dir / "gate_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model", "example_index", "class", "prediction", "band", "gate"]
        )
        for model_name, result in results.items():
            gates = np.asarray(result["gates"])
            labels = np.asarray(result["labels"])
            predictions = np.asarray(result["predictions"])
            for example_index in range(len(labels)):
                for band_index, band_name in enumerate(BAND_LABELS):
                    writer.writerow(
                        [
                            model_name,
                            example_index,
                            CLASS_NAMES[int(labels[example_index])],
                            CLASS_NAMES[int(predictions[example_index])],
                            band_name,
                            gates[example_index, band_index],
                        ]
                    )

    with (output_dir / "gate_selectivity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "accuracy",
                "gate_mean",
                "gate_std",
                *[f"active_bands_at_{threshold:.2f}" for threshold in thresholds],
            ]
        )
        for model_name, result in results.items():
            gates = np.asarray(result["gates"])
            writer.writerow(
                [
                    model_name,
                    result["accuracy"],
                    gates.mean(),
                    gates.std(),
                    *[(gates > threshold).sum(axis=1).mean() for threshold in thresholds],
                ]
            )


def add_heatmap_values(axis: plt.Axes, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if values[row, column] < 0.66 else "black",
            )


def plot_gate_evolution(
    output_dir: Path,
    histories: dict[str, dict[str, np.ndarray]],
    results: dict[str, dict[str, np.ndarray | float]],
) -> None:
    thresholds = np.linspace(0.50, 0.90, 41)
    model_names = list(results)
    figure = plt.figure(figsize=(15.2, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    accuracy_axis = figure.add_subplot(grid[0, 0])
    distribution_axis = figure.add_subplot(grid[0, 1])
    threshold_axis = figure.add_subplot(grid[0, 2])
    heatmap_axes = [figure.add_subplot(grid[1, column]) for column in range(3)]

    colors = ["C0", "C1", "C2"]
    for model_name, color in zip(model_names, colors):
        history = histories[model_name]
        accuracy_axis.plot(
            history["epoch"],
            history["val_acc"],
            label=model_name,
            color=color,
            linewidth=1.5,
        )
        best_index = int(np.argmax(history["val_acc"]))
        accuracy_axis.scatter(
            history["epoch"][best_index],
            history["val_acc"][best_index],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    accuracy_axis.set_title("(a) Validation accuracy", loc="left", fontweight="bold")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.25, 0.76)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(fontsize=8, loc="lower right")

    violin_values = [np.asarray(results[name]["gates"]).ravel() for name in model_names]
    violin = distribution_axis.violinplot(
        violin_values,
        positions=np.arange(1, len(model_names) + 1),
        showmeans=True,
        showextrema=True,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    distribution_axis.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    distribution_axis.set_xticks(
        np.arange(1, len(model_names) + 1),
        ["Independent", "Contextual", "Sparse"],
    )
    distribution_axis.set_ylim(0.40, 1.0)
    distribution_axis.set_ylabel("Gate value")
    distribution_axis.set_title("(b) Gate distributions", loc="left", fontweight="bold")
    distribution_axis.grid(axis="y", alpha=0.2)

    for model_name, color in zip(model_names, colors):
        gates = np.asarray(results[model_name]["gates"])
        active_counts = np.asarray(
            [(gates > threshold).sum(axis=1).mean() for threshold in thresholds]
        )
        threshold_axis.plot(
            thresholds,
            active_counts,
            label=model_name,
            color=color,
            linewidth=1.7,
        )
    threshold_axis.set_title(
        "(c) Selectivity depends on threshold", loc="left", fontweight="bold"
    )
    threshold_axis.set_xlabel("Gate threshold")
    threshold_axis.set_ylabel("Mean active bands")
    threshold_axis.set_xlim(0.50, 0.90)
    threshold_axis.set_ylim(0.0, len(BAND_LABELS) + 0.2)
    threshold_axis.grid(alpha=0.25)
    threshold_axis.legend(fontsize=8)

    global_gate_min = min(
        float(np.asarray(result["class_gate_means"]).min()) for result in results.values()
    )
    global_gate_max = max(
        float(np.asarray(result["class_gate_means"]).max()) for result in results.values()
    )
    color_min = math.floor(global_gate_min * 20.0) / 20.0
    color_max = math.ceil(global_gate_max * 20.0) / 20.0
    heatmap_image = None
    panel_letters = ["d", "e", "f"]
    for axis, model_name, letter in zip(heatmap_axes, model_names, panel_letters):
        values = np.asarray(results[model_name]["class_gate_means"])
        heatmap_image = axis.imshow(
            values,
            aspect="auto",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_title(f"({letter}) {model_name}", loc="left", fontweight="bold")
        axis.set_xticks(
            range(len(BAND_LABELS)), BAND_LABELS, rotation=35, ha="right"
        )
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Initialized band")
        add_heatmap_values(axis, values)
    heatmap_axes[0].set_ylabel("Signal class")
    for axis in heatmap_axes[1:]:
        axis.tick_params(labelleft=False)
    if heatmap_image is not None:
        figure.colorbar(
            heatmap_image,
            ax=heatmap_axes,
            pad=0.012,
            label="Mean gate value",
        )

    figure.savefig(output_dir / "hard_gate_evolution.png", dpi=300)
    figure.savefig(output_dir / "hard_gate_evolution.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validation_loader = build_validation_loader(args)
    histories = {}
    results = {}
    for model_name, spec in MODEL_SPECS.items():
        model = load_model(spec, sample_rate=args.sample_rate)
        histories[model_name] = read_history(Path(spec["history"]))
        results[model_name] = evaluate_model(model, validation_loader)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_thresholds = np.asarray([0.50, 0.60, 0.70, 0.80])
    save_measurements(args.output_dir, results, summary_thresholds)
    plot_gate_evolution(args.output_dir, histories, results)

    for model_name, result in results.items():
        gates = np.asarray(result["gates"])
        print(
            f"model={model_name} accuracy={float(result['accuracy']):.4f} "
            f"gate_mean={gates.mean():.4f} gate_std={gates.std():.4f} "
            f"active_at_0.5={(gates > 0.5).sum(axis=1).mean():.2f} "
            f"active_at_0.7={(gates > 0.7).sum(axis=1).mean():.2f}"
        )
    print(f"saved_figure={args.output_dir / 'hard_gate_evolution.png'}")
    print(f"saved_pdf={args.output_dir / 'hard_gate_evolution.pdf'}")


if __name__ == "__main__":
    main()
