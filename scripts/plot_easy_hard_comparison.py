from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import FrequencyGatedClassifier, get_filter_bands
from src.utils.signal_utils import SyntheticConfig, SyntheticFrequencyDataset


CLASS_NAMES = ["low", "mid", "high", "low + high"]
BAND_LABELS = ["<300 Hz", "300-800 Hz", "0.8-2 kHz", "2-4 kHz", ">4 kHz"]
BAND_CUTOFFS_HZ = [300.0, 800.0, 2_000.0, 4_000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one fixed-filter EGFN checkpoint on easy and hard signals."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "outputs" / "models" / "frequency_gated_synthetic.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "easy_vs_hard",
    )
    parser.add_argument("--num-samples", type=int, default=2_048)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--dataset-seed", type=int, default=2_026)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_v0_model(model_path: Path, sample_rate: int) -> FrequencyGatedClassifier:
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    num_filters = int(state["frontend.filters"].shape[0])
    if num_filters != len(BAND_LABELS):
        raise ValueError(
            f"The V0 comparison expects five fixed filters, found {num_filters}."
        )
    model = FrequencyGatedClassifier(
        num_classes=int(state["classifier.3.weight"].shape[0]),
        sample_rate=sample_rate,
        hidden_dim=int(state["classifier.0.weight"].shape[0]),
        filter_bands=get_filter_bands("default"),
        gate_mode="independent",
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_difficulty(
    model: FrequencyGatedClassifier,
    difficulty: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float | int]:
    dataset = SyntheticFrequencyDataset(
        SyntheticConfig(
            num_samples=args.num_samples,
            sample_rate=args.sample_rate,
            duration=args.duration,
            snr_db=args.snr_db,
            difficulty=difficulty,
            seed=args.dataset_seed,
        )
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_classes = len(CLASS_NAMES)
    num_bands = len(BAND_LABELS)
    num_frequency_bins = int(args.sample_rate * args.duration) // 2 + 1

    spectrum_sums = torch.zeros(num_classes, num_frequency_bins)
    energy_sums = torch.zeros(num_classes, num_bands)
    gate_sums = torch.zeros(num_classes, num_bands)
    class_counts = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for waveforms, labels in loader:
            outputs = model(waveforms)
            predictions = outputs["logits"].argmax(dim=1)
            spectra = torch.fft.rfft(waveforms.squeeze(1), dim=-1).abs()

            for class_index in range(num_classes):
                mask = labels == class_index
                if mask.any():
                    spectrum_sums[class_index] += spectra[mask].sum(dim=0)
                    energy_sums[class_index] += torch.log1p(
                        outputs["energy"][mask]
                    ).sum(dim=0)
                    gate_sums[class_index] += outputs["gates"][mask].sum(dim=0)
                    class_counts[class_index] += mask.sum()

            for target, prediction in zip(labels, predictions):
                confusion[target, prediction] += 1

    safe_counts = class_counts.clamp_min(1.0)
    normalized_confusion = confusion.float() / confusion.sum(dim=1, keepdim=True).clamp_min(1)
    accuracy = confusion.diag().sum().item() / confusion.sum().item()
    frequencies_hz = torch.fft.rfftfreq(
        int(args.sample_rate * args.duration),
        d=1.0 / args.sample_rate,
    )
    return {
        "num_examples": len(dataset),
        "accuracy": accuracy,
        "frequencies_hz": frequencies_hz.numpy(),
        "spectra": (spectrum_sums / safe_counts[:, None]).numpy(),
        "log_energy": (energy_sums / safe_counts[:, None]).numpy(),
        "gates": (gate_sums / safe_counts[:, None]).numpy(),
        "confusion": confusion.numpy(),
        "normalized_confusion": normalized_confusion.numpy(),
    }


def save_metrics(
    output_dir: Path,
    results: dict[str, dict[str, np.ndarray | float | int]],
) -> None:
    with (output_dir / "difficulty_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["difficulty", "num_examples", "accuracy"])
        for difficulty, result in results.items():
            writer.writerow(
                [difficulty, result["num_examples"], result["accuracy"]]
            )

    with (output_dir / "class_band_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["difficulty", "class", "band", "mean_log_energy", "mean_gate"]
        )
        for difficulty, result in results.items():
            log_energy = np.asarray(result["log_energy"])
            gates = np.asarray(result["gates"])
            for class_index, class_name in enumerate(CLASS_NAMES):
                for band_index, band_name in enumerate(BAND_LABELS):
                    writer.writerow(
                        [
                            difficulty,
                            class_name,
                            band_name,
                            log_energy[class_index, band_index],
                            gates[class_index, band_index],
                        ]
                    )

    with (output_dir / "confusion_matrices.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["difficulty", "true_class", "predicted_class", "count"])
        for difficulty, result in results.items():
            confusion = np.asarray(result["confusion"])
            for target_index, target_name in enumerate(CLASS_NAMES):
                for prediction_index, prediction_name in enumerate(CLASS_NAMES):
                    writer.writerow(
                        [
                            difficulty,
                            target_name,
                            prediction_name,
                            confusion[target_index, prediction_index],
                        ]
                    )


def add_heatmap_values(axis: plt.Axes, values: np.ndarray, value_format: str) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.72 * values.max() else "black",
            )


def plot_comparison(
    output_dir: Path,
    results: dict[str, dict[str, np.ndarray | float | int]],
) -> None:
    all_spectra = np.concatenate(
        [np.asarray(result["spectra"]) for result in results.values()], axis=0
    )
    spectrum_reference = max(float(all_spectra.max()), 1e-12)
    all_energy = np.concatenate(
        [np.asarray(result["log_energy"]) for result in results.values()], axis=0
    )
    energy_max = float(all_energy.max())
    all_gates = np.concatenate(
        [np.asarray(result["gates"]) for result in results.values()], axis=0
    )
    gate_min = min(0.5, float(all_gates.min()))
    gate_max = max(0.7, float(all_gates.max()))

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 7.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.0, 1.0]},
    )
    class_colors = plt.get_cmap("tab10").colors[: len(CLASS_NAMES)]

    for row, (difficulty, result) in enumerate(results.items()):
        frequencies_khz = np.asarray(result["frequencies_hz"]) / 1_000.0
        spectra = np.asarray(result["spectra"])
        spectra_db = 20.0 * np.log10(np.maximum(spectra, 1e-12) / spectrum_reference)
        for class_index, class_name in enumerate(CLASS_NAMES):
            axes[row, 0].plot(
                frequencies_khz,
                spectra_db[class_index],
                label=class_name,
                color=class_colors[class_index],
                linewidth=1.2,
            )
        for cutoff_hz in BAND_CUTOFFS_HZ:
            axes[row, 0].axvline(
                cutoff_hz / 1_000.0,
                color="0.55",
                linewidth=0.7,
                linestyle="--",
            )
        axes[row, 0].set_xlim(0.0, 5.0)
        axes[row, 0].set_ylim(-55.0, 1.0)
        axes[row, 0].set_ylabel("Mean magnitude (dB)")
        axes[row, 0].set_title(f"({chr(97 + row * 4)}) {difficulty.capitalize()} spectra", loc="left", fontweight="bold")
        if row == 0:
            axes[row, 0].legend(fontsize=7, ncol=2, loc="lower right")

        log_energy = np.asarray(result["log_energy"])
        energy_image = axes[row, 1].imshow(
            log_energy,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=energy_max,
        )
        axes[row, 1].set_title(
            f"({chr(98 + row * 4)}) Mean log-energy", loc="left", fontweight="bold"
        )
        add_heatmap_values(axes[row, 1], log_energy, ".2f")

        gates = np.asarray(result["gates"])
        gate_image = axes[row, 2].imshow(
            gates,
            aspect="auto",
            cmap="viridis",
            vmin=gate_min,
            vmax=gate_max,
        )
        axes[row, 2].set_title(
            f"({chr(99 + row * 4)}) Mean gates", loc="left", fontweight="bold"
        )
        add_heatmap_values(axes[row, 2], gates, ".2f")

        normalized_confusion = np.asarray(result["normalized_confusion"])
        confusion_image = axes[row, 3].imshow(
            normalized_confusion,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 3].set_title(
            f"({chr(100 + row * 4)}) Confusion\nAccuracy = {float(result['accuracy']):.3f}",
            loc="left",
            fontweight="bold",
        )
        add_heatmap_values(axes[row, 3], normalized_confusion, ".2f")

        for column in (1, 2):
            axes[row, column].set_xticks(range(len(BAND_LABELS)), BAND_LABELS, rotation=35, ha="right")
            axes[row, column].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axes[row, 3].set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        axes[row, 3].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axes[row, 3].set_xlabel("Predicted class")
        axes[row, 3].set_ylabel("True class")

    axes[-1, 0].set_xlabel("Frequency (kHz)")
    axes[0, 0].set_xlabel("Frequency (kHz)")
    figure.colorbar(energy_image, ax=axes[:, 1], pad=0.015, label="log(1 + energy)")
    figure.colorbar(gate_image, ax=axes[:, 2], pad=0.015, label="Gate value")
    figure.colorbar(confusion_image, ax=axes[:, 3], pad=0.015, label="Row-normalized rate")
    figure.savefig(output_dir / "v0_easy_vs_hard_behavior.png", dpi=300)
    figure.savefig(output_dir / "v0_easy_vs_hard_behavior.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model = load_v0_model(args.model_path, sample_rate=args.sample_rate)
    results = {
        difficulty: evaluate_difficulty(model, difficulty, args)
        for difficulty in ("easy", "hard")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(args.output_dir, results)
    plot_comparison(args.output_dir, results)

    print(f"model={args.model_path}")
    for difficulty, result in results.items():
        print(
            f"difficulty={difficulty} examples={result['num_examples']} "
            f"accuracy={float(result['accuracy']):.4f}"
        )
    print(f"saved_figure={args.output_dir / 'v0_easy_vs_hard_behavior.png'}")
    print(f"saved_pdf={args.output_dir / 'v0_easy_vs_hard_behavior.pdf'}")
    print(f"saved_metrics={args.output_dir}")


if __name__ == "__main__":
    main()
