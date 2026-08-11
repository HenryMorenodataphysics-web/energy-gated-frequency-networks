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

from src.models import FrequencyGatedClassifier, get_filter_bands


BAND_LABELS = {
    "default": [
        "<300 Hz",
        "300-800 Hz",
        "0.8-2 kHz",
        "2-4 kHz",
        ">4 kHz",
    ],
    "fine": [
        "<250 Hz",
        "250-500 Hz",
        "500-750 Hz",
        "750-1000 Hz",
        "1-1.5 kHz",
        "1.5-2.5 kHz",
        "2.5-4 kHz",
        ">4 kHz",
    ],
}

CUTOFFS_HZ = {
    "default": [300.0, 800.0, 2_000.0, 4_000.0],
    "fine": [250.0, 500.0, 750.0, 1_000.0, 1_500.0, 2_500.0, 4_000.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe EGFN energy, gates, and outputs with sinusoidal inputs."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "outputs" / "models" / "frequency_gated_synthetic.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "v0_frequency_sweep",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--frequency-step", type=float, default=25.0)
    parser.add_argument("--min-frequency", type=float, default=50.0)
    parser.add_argument("--max-frequency", type=float, default=7_950.0)
    parser.add_argument("--target-rms", type=float, default=1.0 / math.sqrt(2.0))
    parser.add_argument("--batch-frequencies", type=int, default=24)
    return parser.parse_args()


def infer_model(state: dict[str, torch.Tensor], sample_rate: int) -> tuple[FrequencyGatedClassifier, str]:
    num_filters = int(state["frontend.filters"].shape[0])
    bank_name = {5: "default", 8: "fine"}.get(num_filters)
    if bank_name is None:
        raise ValueError(f"Unsupported checkpoint with {num_filters} frontend filters.")

    hidden_dim = int(state["classifier.0.weight"].shape[0])
    num_classes = int(state["classifier.3.weight"].shape[0])
    model = FrequencyGatedClassifier(
        num_classes=num_classes,
        sample_rate=sample_rate,
        hidden_dim=hidden_dim,
        filter_bands=get_filter_bands(bank_name),
        gate_mode="independent",
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, bank_name


def probe_frequencies(
    model: FrequencyGatedClassifier,
    frequencies_hz: torch.Tensor,
    sample_rate: int,
    duration: float,
    target_rms: float,
    batch_frequencies: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = int(sample_rate * duration)
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    phases = torch.tensor([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])

    energy_batches = []
    gate_batches = []
    activation_batches = []

    with torch.no_grad():
        for start in range(0, len(frequencies_hz), batch_frequencies):
            frequencies = frequencies_hz[start : start + batch_frequencies]
            waves = torch.sin(
                2.0
                * math.pi
                * frequencies[:, None, None]
                * time[None, None, :]
                + phases[None, :, None]
            )
            rms = waves.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            waves = target_rms * waves / rms
            num_frequencies, num_phases, _ = waves.shape
            waves = waves.reshape(num_frequencies * num_phases, 1, num_samples)

            band_outputs, gates, energy = model.frontend(waves)
            num_filters = gates.shape[-1]
            energy = energy.reshape(num_frequencies, num_phases, num_filters).mean(dim=1)
            gates = gates.reshape(num_frequencies, num_phases, num_filters).mean(dim=1)
            activation_rms = (
                band_outputs.square()
                .mean(dim=-1)
                .sqrt()
                .reshape(num_frequencies, num_phases, num_filters)
                .mean(dim=1)
            )

            energy_batches.append(energy.cpu())
            gate_batches.append(gates.cpu())
            activation_batches.append(activation_rms.cpu())

    return (
        torch.cat(energy_batches),
        torch.cat(gate_batches),
        torch.cat(activation_batches),
    )


def relative_db(values: np.ndarray, power: bool) -> np.ndarray:
    multiplier = 10.0 if power else 20.0
    reference = max(float(values.max()), 1e-12)
    return multiplier * np.log10(np.maximum(values, 1e-12) / reference)


def save_measurements(
    output_path: Path,
    frequencies_hz: np.ndarray,
    band_labels: list[str],
    energy: np.ndarray,
    gates: np.ndarray,
    activation_rms: np.ndarray,
) -> None:
    energy_db = relative_db(energy, power=True)
    activation_db = relative_db(activation_rms, power=False)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frequency_hz",
                "band_index",
                "band_label",
                "energy",
                "energy_relative_db",
                "gate",
                "activation_rms",
                "activation_relative_db",
            ]
        )
        for frequency_index, frequency_hz in enumerate(frequencies_hz):
            for band_index, band_label in enumerate(band_labels):
                writer.writerow(
                    [
                        float(frequency_hz),
                        band_index,
                        band_label,
                        float(energy[frequency_index, band_index]),
                        float(energy_db[frequency_index, band_index]),
                        float(gates[frequency_index, band_index]),
                        float(activation_rms[frequency_index, band_index]),
                        float(activation_db[frequency_index, band_index]),
                    ]
                )


def plot_sweep(
    output_dir: Path,
    frequencies_hz: np.ndarray,
    band_labels: list[str],
    cutoffs_hz: list[float],
    energy: np.ndarray,
    gates: np.ndarray,
    activation_rms: np.ndarray,
) -> None:
    panels = [
        (relative_db(energy, power=True), "(a) Filtered energy", "Relative energy (dB)", -40.0, 0.0),
        (gates, "(b) Energy gates", "Gate value", 0.0, 1.0),
        (
            relative_db(activation_rms, power=False),
            "(c) Gated activation",
            "Relative RMS (dB)",
            -40.0,
            0.0,
        ),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.2), sharex=True, constrained_layout=True)
    extent = [
        frequencies_hz[0] / 1_000.0,
        frequencies_hz[-1] / 1_000.0,
        len(band_labels) - 0.5,
        -0.5,
    ]

    for axis, (values, title, colorbar_label, value_min, value_max) in zip(axes, panels):
        image = axis.imshow(
            values.T,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_yticks(range(len(band_labels)), band_labels)
        axis.set_ylabel("EGFN band")
        for cutoff_hz in cutoffs_hz:
            axis.axvline(cutoff_hz / 1_000.0, color="white", linewidth=0.8, alpha=0.8)
        colorbar = figure.colorbar(image, ax=axis, pad=0.015)
        colorbar.set_label(colorbar_label)

    axes[-1].set_xlabel("Input sinusoid frequency (kHz)")
    axes[-1].set_xlim(frequencies_hz[0] / 1_000.0, frequencies_hz[-1] / 1_000.0)
    figure.savefig(output_dir / "v0_frequency_activation_sweep.png", dpi=300)
    figure.savefig(output_dir / "v0_frequency_activation_sweep.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_frequency >= args.sample_rate / 2:
        raise ValueError("--max-frequency must be lower than the Nyquist frequency.")
    if args.min_frequency <= 0 or args.frequency_step <= 0:
        raise ValueError("Frequency limits and step must be positive.")

    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model, bank_name = infer_model(state, sample_rate=args.sample_rate)
    frequencies_hz = torch.arange(
        args.min_frequency,
        args.max_frequency + 0.5 * args.frequency_step,
        args.frequency_step,
    )
    energy, gates, activation_rms = probe_frequencies(
        model=model,
        frequencies_hz=frequencies_hz,
        sample_rate=args.sample_rate,
        duration=args.duration,
        target_rms=args.target_rms,
        batch_frequencies=args.batch_frequencies,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequencies_np = frequencies_hz.numpy()
    energy_np = energy.numpy()
    gates_np = gates.numpy()
    activation_np = activation_rms.numpy()
    save_measurements(
        args.output_dir / "v0_frequency_activation_sweep.csv",
        frequencies_np,
        BAND_LABELS[bank_name],
        energy_np,
        gates_np,
        activation_np,
    )
    plot_sweep(
        output_dir=args.output_dir,
        frequencies_hz=frequencies_np,
        band_labels=BAND_LABELS[bank_name],
        cutoffs_hz=CUTOFFS_HZ[bank_name],
        energy=energy_np,
        gates=gates_np,
        activation_rms=activation_np,
    )

    print(f"model={args.model_path}")
    print(f"filter_bank={bank_name}")
    print(f"frequencies={len(frequencies_hz)}")
    print(f"gate_range={gates.min().item():.4f},{gates.max().item():.4f}")
    print(f"saved_figure={args.output_dir / 'v0_frequency_activation_sweep.png'}")
    print(f"saved_pdf={args.output_dir / 'v0_frequency_activation_sweep.pdf'}")
    print(f"saved_data={args.output_dir / 'v0_frequency_activation_sweep.csv'}")


if __name__ == "__main__":
    main()
