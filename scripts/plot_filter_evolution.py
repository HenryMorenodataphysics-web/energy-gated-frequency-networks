from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
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
CUTOFFS_HZ = [250.0, 500.0, 750.0, 1_000.0, 1_500.0, 2_500.0, 4_000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the transition from fixed to learnable fine filters."
    )
    parser.add_argument(
        "--fixed-history",
        type=Path,
        default=ROOT / "outputs" / "hard_fixed" / "metrics" / "synthetic_history.csv",
    )
    parser.add_argument(
        "--learnable-history",
        type=Path,
        default=ROOT / "outputs" / "hard_learnable" / "metrics" / "synthetic_history.csv",
    )
    parser.add_argument(
        "--fixed-sweep",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "figures"
            / "hard_fixed_sweep"
            / "v0_frequency_activation_sweep.csv"
        ),
    )
    parser.add_argument(
        "--learnable-sweep",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "figures"
            / "hard_learnable_sweep"
            / "v0_frequency_activation_sweep.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "filter_evolution",
    )
    parser.add_argument(
        "--coarse-fixed-best-val-acc",
        type=float,
        default=0.531,
        help="Historical five-band fixed-filter result used as a reference line.",
    )
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "train_acc", "val_acc", "train_loss", "val_loss")
    }


def read_sweep(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    frequencies = np.asarray(sorted({float(row["frequency_hz"]) for row in rows}))
    num_bands = max(int(row["band_index"]) for row in rows) + 1
    frequency_to_index = {frequency: index for index, frequency in enumerate(frequencies)}
    energy_db = np.zeros((len(frequencies), num_bands))
    gates = np.zeros((len(frequencies), num_bands))
    for row in rows:
        frequency_index = frequency_to_index[float(row["frequency_hz"])]
        band_index = int(row["band_index"])
        energy_db[frequency_index, band_index] = float(row["energy_relative_db"])
        gates[frequency_index, band_index] = float(row["gate"])
    return {"frequencies_hz": frequencies, "energy_db": energy_db, "gates": gates}


def draw_sweep_panel(
    axis: plt.Axes,
    frequencies_hz: np.ndarray,
    values: np.ndarray,
    title: str,
    value_min: float,
    value_max: float,
) -> plt.AxesImage:
    image = axis.imshow(
        values.T,
        aspect="auto",
        interpolation="nearest",
        extent=[
            frequencies_hz[0] / 1_000.0,
            frequencies_hz[-1] / 1_000.0,
            len(BAND_LABELS) - 0.5,
            -0.5,
        ],
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_yticks(range(len(BAND_LABELS)), BAND_LABELS)
    for cutoff_hz in CUTOFFS_HZ:
        axis.axvline(
            cutoff_hz / 1_000.0,
            color="white",
            linewidth=0.7,
            alpha=0.75,
        )
    return image


def save_summary(
    output_path: Path,
    coarse_best: float,
    fixed_history: dict[str, np.ndarray],
    learnable_history: dict[str, np.ndarray],
) -> None:
    rows = [
        ("coarse_fixed_5_band", "historical log", coarse_best),
        (
            "fine_fixed_8_band",
            "outputs/hard_fixed/metrics/synthetic_history.csv",
            float(fixed_history["val_acc"].max()),
        ),
        (
            "fine_learnable_8_band",
            "outputs/hard_learnable/metrics/synthetic_history.csv",
            float(learnable_history["val_acc"].max()),
        ),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["configuration", "source", "best_validation_accuracy"])
        writer.writerows(rows)


def plot_evolution(
    output_dir: Path,
    coarse_best: float,
    fixed_history: dict[str, np.ndarray],
    learnable_history: dict[str, np.ndarray],
    fixed_sweep: dict[str, np.ndarray],
    learnable_sweep: dict[str, np.ndarray],
) -> None:
    figure = plt.figure(figsize=(14.5, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.0])
    accuracy_axis = figure.add_subplot(grid[:, 0])
    fixed_energy_axis = figure.add_subplot(grid[0, 1])
    learnable_energy_axis = figure.add_subplot(grid[0, 2])
    fixed_gate_axis = figure.add_subplot(grid[1, 1])
    learnable_gate_axis = figure.add_subplot(grid[1, 2])

    accuracy_axis.plot(
        fixed_history["epoch"],
        fixed_history["val_acc"],
        label="Fine fixed (8 bands)",
        linewidth=1.7,
    )
    accuracy_axis.plot(
        learnable_history["epoch"],
        learnable_history["val_acc"],
        label="Fine learnable (8 kernels)",
        linewidth=1.7,
    )
    accuracy_axis.axhline(
        coarse_best,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label="Coarse fixed historical best",
    )
    for history, color in ((fixed_history, "C0"), (learnable_history, "C1")):
        best_index = int(np.argmax(history["val_acc"]))
        accuracy_axis.scatter(
            history["epoch"][best_index],
            history["val_acc"][best_index],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        accuracy_axis.annotate(
            f"{history['val_acc'][best_index]:.3f}",
            (history["epoch"][best_index], history["val_acc"][best_index]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    accuracy_axis.set_title("(a) Validation on hard signals", loc="left", fontweight="bold")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Validation accuracy")
    accuracy_axis.set_ylim(0.24, 0.76)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(fontsize=8, loc="lower right")

    energy_image = draw_sweep_panel(
        fixed_energy_axis,
        fixed_sweep["frequencies_hz"],
        fixed_sweep["energy_db"],
        "(b) Fixed: filtered energy",
        -40.0,
        0.0,
    )
    draw_sweep_panel(
        learnable_energy_axis,
        learnable_sweep["frequencies_hz"],
        learnable_sweep["energy_db"],
        "(c) Learnable: filtered energy",
        -40.0,
        0.0,
    )
    gate_image = draw_sweep_panel(
        fixed_gate_axis,
        fixed_sweep["frequencies_hz"],
        fixed_sweep["gates"],
        "(d) Fixed: energy gates",
        0.5,
        1.0,
    )
    draw_sweep_panel(
        learnable_gate_axis,
        learnable_sweep["frequencies_hz"],
        learnable_sweep["gates"],
        "(e) Learnable: energy gates",
        0.5,
        1.0,
    )

    for axis in (fixed_energy_axis, learnable_energy_axis):
        axis.tick_params(labelbottom=False)
    for axis in (fixed_gate_axis, learnable_gate_axis):
        axis.set_xlabel("Input frequency (kHz)")
    for axis in (learnable_energy_axis, learnable_gate_axis):
        axis.tick_params(labelleft=False)
    fixed_energy_axis.set_ylabel("Initialized band")
    fixed_gate_axis.set_ylabel("Initialized band")

    figure.colorbar(
        energy_image,
        ax=[fixed_energy_axis, learnable_energy_axis],
        pad=0.015,
        label="Relative energy (dB)",
    )
    figure.colorbar(
        gate_image,
        ax=[fixed_gate_axis, learnable_gate_axis],
        pad=0.015,
        label="Gate value",
    )
    figure.savefig(output_dir / "hard_filter_evolution.png", dpi=300)
    figure.savefig(output_dir / "hard_filter_evolution.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    fixed_history = read_history(args.fixed_history)
    learnable_history = read_history(args.learnable_history)
    fixed_sweep = read_sweep(args.fixed_sweep)
    learnable_sweep = read_sweep(args.learnable_sweep)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(
        args.output_dir / "filter_evolution_summary.csv",
        args.coarse_fixed_best_val_acc,
        fixed_history,
        learnable_history,
    )
    plot_evolution(
        output_dir=args.output_dir,
        coarse_best=args.coarse_fixed_best_val_acc,
        fixed_history=fixed_history,
        learnable_history=learnable_history,
        fixed_sweep=fixed_sweep,
        learnable_sweep=learnable_sweep,
    )
    print(f"coarse_fixed_best={args.coarse_fixed_best_val_acc:.3f}")
    print(f"fine_fixed_best={fixed_history['val_acc'].max():.3f}")
    print(f"fine_learnable_best={learnable_history['val_acc'].max():.3f}")
    print(f"saved_figure={args.output_dir / 'hard_filter_evolution.png'}")
    print(f"saved_pdf={args.output_dir / 'hard_filter_evolution.pdf'}")


if __name__ == "__main__":
    main()
