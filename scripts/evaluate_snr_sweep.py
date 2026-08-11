from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import FrequencyGatedClassifier, get_filter_bands
from src.utils.signal_utils import SyntheticConfig, SyntheticFrequencyDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained synthetic model under multiple SNR levels."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "outputs" / "models" / "frequency_gated_synthetic_best.pt",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--filter-bank", choices=["default", "fine"], default="fine")
    parser.add_argument("--learnable-filters", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=101)
    parser.add_argument(
        "--gate-mode",
        choices=["none", "independent", "contextual"],
        default="independent",
    )
    parser.add_argument("--gate-hidden-dim", type=int, default=None)
    parser.add_argument("--active-gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use cuda when available by default.",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def evaluate(
    model: FrequencyGatedClassifier,
    loader: DataLoader,
    active_gate_threshold: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_gate_mean = 0.0
    total_active_bands = 0.0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs["logits"], y)
            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (outputs["logits"].argmax(dim=-1) == y).sum().item()
            total_gate_mean += outputs["gates"].mean().item() * batch_size
            total_active_bands += (
                (outputs["gates"] > active_gate_threshold)
                .float()
                .sum(dim=1)
                .mean()
                .item()
                * batch_size
            )
            total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        total_gate_mean / total_examples,
        total_active_bands / total_examples,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    snr_levels: list[float | None] = [None, 20.0, 10.0, 5.0, 0.0, -5.0]

    model = FrequencyGatedClassifier(
        num_classes=4,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        learnable_filters=args.learnable_filters,
        filter_bands=get_filter_bands(args.filter_bank),
        gate_mode=args.gate_mode,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    rows = []
    for index, snr_db in enumerate(snr_levels):
        dataset = SyntheticFrequencyDataset(
            SyntheticConfig(
                num_samples=args.num_samples,
                duration=args.duration,
                snr_db=snr_db,
                difficulty=args.difficulty,
                seed=args.seed + index,
            )
        )
        loader = DataLoader(dataset, batch_size=args.batch_size)
        loss, acc, gate_mean, active_bands = evaluate(
            model,
            loader,
            active_gate_threshold=args.active_gate_threshold,
            device=device,
        )
        snr_label = "clean" if snr_db is None else f"{snr_db:g}"
        rows.append(
            {
                "snr_db": snr_label,
                "loss": loss,
                "acc": acc,
                "gate_mean": gate_mean,
                "active_bands": active_bands,
            }
        )
        print(
            f"snr_db={snr_label} loss={loss:.4f} acc={acc:.3f} "
            f"gate_mean={gate_mean:.3f} active_bands={active_bands:.2f}"
        )

    metrics_dir = args.output_dir / "metrics"
    figure_dir = args.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / f"snr_sweep_{args.difficulty}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["snr_db", "loss", "acc", "gate_mean", "active_bands"],
        )
        writer.writeheader()
        writer.writerows(rows)

    x_labels = [row["snr_db"] for row in rows]
    y_values = [row["acc"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x_labels, y_values, marker="o")
    axes[0].set_title(f"SNR robustness sweep ({args.difficulty})")
    axes[0].set_xlabel("SNR dB")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].plot(x_labels, [row["gate_mean"] for row in rows], marker="o", label="mean gate")
    axes[1].plot(
        x_labels,
        [row["active_bands"] for row in rows],
        marker="o",
        label="active bands",
    )
    axes[1].set_title("Gate selectivity under noise")
    axes[1].set_xlabel("SNR dB")
    axes[1].legend()
    fig.savefig(figure_dir / f"snr_sweep_{args.difficulty}.png", dpi=150)
    plt.close(fig)

    print(f"saved_metrics={csv_path}")
    print(f"saved_figure={figure_dir / f'snr_sweep_{args.difficulty}.png'}")


if __name__ == "__main__":
    main()
