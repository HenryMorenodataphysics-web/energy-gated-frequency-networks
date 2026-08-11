from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_mimii import build_model, resolve_device
from src.data import MIMIIDataset, find_mimii_recordings, split_records_stratified
from src.utils import metrics_at_threshold, select_f1_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a MIMII classifier and analyze EGFN frequency gates."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict, Path]:
    diagnostics_path = run_dir / "metrics" / "test_diagnostics.pt"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")
    diagnostics = torch.load(diagnostics_path, map_location="cpu", weights_only=False)
    config = dict(diagnostics["args"])
    model_path = run_dir / "models" / f"mimii_{config['model']}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return config, model_path


def collect_outputs(model, loader, device, include_bands: bool) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        "targets": [],
        "probabilities": [],
    }
    if include_bands:
        collected["gates"] = []
        collected["energy"] = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device))
            probabilities = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            collected["targets"].append(y.numpy())
            collected["probabilities"].append(probabilities.cpu().numpy())
            if include_bands:
                collected["gates"].append(outputs["gates"].cpu().numpy())
                collected["energy"].append(outputs["energy"].cpu().numpy())

    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def band_label(band: tuple[str, float | None, float | None]) -> str:
    kind, low, high = band
    if kind == "lowpass":
        return f"< {high:g} Hz"
    if kind == "highpass":
        return f"> {low:g} Hz"
    return f"{low:g}-{high:g} Hz"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_threshold_figure(rows: list[dict], selected: float, path: Path) -> None:
    thresholds = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for metric in ("precision", "recall", "f1"):
        ax.plot(thresholds, [row[metric] for row in rows], label=metric.capitalize())
    ax.axvline(selected, color="black", linestyle="--", label=f"Selected: {selected:.2f}")
    ax.set(xlabel="Anomaly threshold", ylabel="Score", ylim=(0.0, 1.05), title="Validation threshold calibration")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_figure(default: dict, calibrated: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax, title, metrics in zip(
        axes,
        ("Threshold 0.50", f"Calibrated threshold {calibrated['threshold']:.2f}"),
        (default, calibrated),
    ):
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ax.set(
            xticks=(0, 1),
            yticks=(0, 1),
            xticklabels=("Normal", "Abnormal"),
            yticklabels=("Normal", "Abnormal"),
            xlabel="Predicted",
            ylabel="True",
            title=title,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_band_analysis(outputs: dict[str, np.ndarray], filter_bands, analysis_dir: Path) -> None:
    labels = [band_label(band) for band in filter_bands]
    targets = outputs["targets"]
    gates = outputs["gates"]
    log_energy = np.log1p(outputs["energy"])
    rows = []
    for index, label in enumerate(labels):
        normal_gate = gates[targets == 0, index]
        abnormal_gate = gates[targets == 1, index]
        normal_energy = log_energy[targets == 0, index]
        abnormal_energy = log_energy[targets == 1, index]
        rows.append(
            {
                "band": label,
                "normal_gate_mean": float(normal_gate.mean()),
                "abnormal_gate_mean": float(abnormal_gate.mean()),
                "gate_difference": float(abnormal_gate.mean() - normal_gate.mean()),
                "normal_log_energy_mean": float(normal_energy.mean()),
                "abnormal_log_energy_mean": float(abnormal_energy.mean()),
                "log_energy_difference": float(abnormal_energy.mean() - normal_energy.mean()),
            }
        )
    write_csv(analysis_dir / "band_statistics.csv", rows)

    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].bar(x - width / 2, [row["normal_gate_mean"] for row in rows], width, label="Normal")
    axes[0].bar(x + width / 2, [row["abnormal_gate_mean"] for row in rows], width, label="Abnormal")
    axes[0].set(ylabel="Mean gate", title="EGFN gate activation by condition", ylim=(0.0, 1.0))
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].legend()
    axes[1].bar(x - width / 2, [row["normal_log_energy_mean"] for row in rows], width, label="Normal")
    axes[1].bar(x + width / 2, [row["abnormal_log_energy_mean"] for row in rows], width, label="Abnormal")
    axes[1].set(ylabel="Mean log(1 + energy)", title="Band energy by condition")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].legend()
    fig.savefig(analysis_dir / "gate_energy_by_label.png", dpi=180)
    plt.close(fig)


def save_filter_responses(frontend, analysis_dir: Path) -> None:
    filters = frontend.get_filters().detach().cpu().squeeze(1).numpy()
    frequencies = np.fft.rfftfreq(filters.shape[-1], d=1.0 / frontend.sample_rate)
    magnitudes = np.abs(np.fft.rfft(filters, axis=-1))
    normalized = magnitudes / np.maximum(magnitudes.max(axis=1, keepdims=True), 1e-12)
    labels = [band_label(band) for band in frontend.filter_bands]

    cutoff_lows = cutoff_highs = None
    if frontend.filter_mode == "sinc":
        low_tensor, high_tensor = frontend.cutoff_frequencies_hz()
        cutoff_lows = low_tensor.detach().cpu().numpy()
        cutoff_highs = high_tensor.detach().cpu().numpy()

    rows = []
    for index, label in enumerate(labels):
        magnitude_sum = magnitudes[index].sum()
        centroid = float((frequencies * magnitudes[index]).sum() / max(magnitude_sum, 1e-12))
        row = {
            "initialized_band": label,
            "filter_mode": frontend.filter_mode,
            "learned_peak_hz": float(frequencies[np.argmax(magnitudes[index])]),
            "spectral_centroid_hz": centroid,
        }
        if cutoff_lows is not None and cutoff_highs is not None:
            row["learned_low_hz"] = float(cutoff_lows[index])
            row["learned_high_hz"] = float(cutoff_highs[index])
            row["learned_bandwidth_hz"] = float(cutoff_highs[index] - cutoff_lows[index])
        rows.append(row)
    write_csv(analysis_dir / "learned_filter_statistics.csv", rows)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for index, label in enumerate(labels):
        ax.plot(frequencies, normalized[index], label=label)
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="Normalized magnitude",
        title="Learned EGFN filter responses",
        xlim=(0, frontend.sample_rate / 2),
        ylim=(0, 1.05),
    )
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(analysis_dir / "learned_filter_responses.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, model_path = load_run(args.run_dir)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    device = resolve_device(args.device)
    print(f"device={device}")

    records = find_mimii_recordings(
        config["data_dir"],
        machine_type=config["machine_type"],
        machine_id=config["machine_id"],
        snr=config["snr"],
    )
    _, val_records, test_records = split_records_stratified(
        records,
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
    )
    val_set = MIMIIDataset(
        val_records,
        target_sample_rate=config["target_sample_rate"],
        duration=config["duration"],
        crop_mode="center",
    )
    test_set = MIMIIDataset(
        test_records,
        target_sample_rate=config["target_sample_rate"],
        duration=config["duration"],
        crop_mode="center",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    config_namespace = argparse.Namespace(**config)
    model = build_model(config_namespace).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    include_bands = hasattr(model, "frontend")
    val_outputs = collect_outputs(model, val_loader, device, include_bands=False)
    test_outputs = collect_outputs(model, test_loader, device, include_bands=include_bands)

    selected_threshold, sweep_rows = select_f1_threshold(
        val_outputs["targets"], val_outputs["probabilities"]
    )
    val_default = metrics_at_threshold(val_outputs["targets"], val_outputs["probabilities"], 0.5)
    val_calibrated = metrics_at_threshold(
        val_outputs["targets"], val_outputs["probabilities"], selected_threshold
    )
    test_default = metrics_at_threshold(test_outputs["targets"], test_outputs["probabilities"], 0.5)
    test_calibrated = metrics_at_threshold(
        test_outputs["targets"], test_outputs["probabilities"], selected_threshold
    )

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "threshold_sweep_validation.csv", sweep_rows)
    save_threshold_figure(sweep_rows, selected_threshold, analysis_dir / "threshold_calibration.png")
    save_confusion_figure(
        test_default,
        test_calibrated,
        analysis_dir / "test_confusion_default_vs_calibrated.png",
    )
    report = {
        "selection_set": "validation",
        "selection_metric": "f1",
        "selected_threshold": selected_threshold,
        "validation_default": val_default,
        "validation_calibrated": val_calibrated,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
    }
    with (analysis_dir / "threshold_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    prediction_rows = []
    for record, target, probability in zip(
        test_records, test_outputs["targets"], test_outputs["probabilities"]
    ):
        prediction_rows.append(
            {
                "path": str(record.path),
                "machine_id": record.machine_id,
                "target": int(target),
                "anomaly_probability": float(probability),
                "prediction_default": int(probability >= 0.5),
                "prediction_calibrated": int(probability >= selected_threshold),
            }
        )
    write_csv(analysis_dir / "test_predictions.csv", prediction_rows)

    if include_bands:
        save_band_analysis(test_outputs, model.frontend.filter_bands, analysis_dir)
        save_filter_responses(model.frontend, analysis_dir)

    print(f"selected_threshold={selected_threshold:.2f}")
    print(
        "test_default "
        f"acc={test_default['accuracy']:.3f} precision={test_default['precision']:.3f} "
        f"recall={test_default['recall']:.3f} f1={test_default['f1']:.3f} auc={test_default['auc']:.3f}"
    )
    print(
        "test_calibrated "
        f"acc={test_calibrated['accuracy']:.3f} precision={test_calibrated['precision']:.3f} "
        f"recall={test_calibrated['recall']:.3f} f1={test_calibrated['f1']:.3f} auc={test_calibrated['auc']:.3f}"
    )
    print(f"saved_analysis={analysis_dir}")


if __name__ == "__main__":
    main()
