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
from torch.utils.data import DataLoader, random_split

from src.models import FrequencyGatedClassifier, get_filter_bands
from src.utils.signal_utils import SyntheticConfig, SyntheticFrequencyDataset


CLASS_NAMES = ["low", "mid", "high", "low_high"]
BAND_LABELS = {
    "default": ["lowpass", "300_800", "800_2000", "2000_4000", "highpass"],
    "fine": [
        "lowpass",
        "250_500",
        "500_750",
        "750_1000",
        "1000_1500",
        "1500_2500",
        "2500_4000",
        "highpass",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Energy-Gated Frequency classifier on synthetic signals."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=640)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--filter-bank", choices=["default", "fine"], default="default")
    parser.add_argument("--learnable-filters", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=101)
    parser.add_argument(
        "--gate-mode",
        choices=["none", "independent", "contextual"],
        default="independent",
    )
    parser.add_argument("--gate-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--gate-l1",
        type=float,
        default=0.0,
        help="Penalty strength for sparse/selective gates. Try 0.001 to 0.02.",
    )
    parser.add_argument(
        "--active-gate-threshold",
        type=float,
        default=0.5,
        help="Gate value counted as active for reporting.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use cuda when available by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def run_epoch(
    model: FrequencyGatedClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    gate_l1: float = 0.0,
    active_gate_threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[float, float, float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_gate_mean = 0.0
    total_active_bands = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        if device is not None:
            x = x.to(device)
            y = y.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(x)
            cls_loss = F.cross_entropy(outputs["logits"], y)
            gate_mean = outputs["gates"].mean()
            loss = cls_loss + gate_l1 * gate_mean

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_gate_mean += gate_mean.item() * batch_size
        total_active_bands += (
            (outputs["gates"] > active_gate_threshold)
            .float()
            .sum(dim=1)
            .mean()
            .item()
            * batch_size
        )
        total_correct += (outputs["logits"].argmax(dim=-1) == y).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_cls_loss / total_examples,
        total_correct / total_examples,
        total_gate_mean / total_examples,
        total_active_bands / total_examples,
    )


def collect_diagnostics(
    model: FrequencyGatedClassifier,
    loader: DataLoader,
    num_classes: int,
    active_gate_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    gate_sums = torch.zeros(num_classes, model.frontend.num_filters)
    energy_sums = torch.zeros(num_classes, model.frontend.num_filters)
    active_band_sums = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes).clamp_min(0)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            predictions = outputs["logits"].argmax(dim=-1)

            for true_label, predicted_label in zip(y, predictions):
                confusion[true_label.cpu(), predicted_label.cpu()] += 1

            for class_idx in range(num_classes):
                mask = y == class_idx
                if mask.any():
                    gate_sums[class_idx] += outputs["gates"][mask].sum(dim=0)
                    energy_sums[class_idx] += torch.log1p(outputs["energy"][mask]).sum(dim=0)
                    active_band_sums[class_idx] += (
                        (outputs["gates"][mask] > active_gate_threshold)
                        .float()
                        .sum(dim=1)
                        .sum()
                    )
                    class_counts[class_idx] += mask.sum().item()

    counts = class_counts.clamp_min(1).unsqueeze(1)
    active_bands_by_class = active_band_sums / class_counts.clamp_min(1)
    return confusion, gate_sums / counts, energy_sums / counts, active_bands_by_class


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_history.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    axes[2].plot(epochs, [row["train_gate_mean"] for row in history], label="train mean gate")
    axes[2].plot(epochs, [row["val_gate_mean"] for row in history], label="val mean gate")
    axes[2].plot(
        epochs,
        [row["train_active_bands"] for row in history],
        label="train active bands",
        linestyle="--",
    )
    axes[2].plot(
        epochs,
        [row["val_active_bands"] for row in history],
        label="val active bands",
        linestyle="--",
    )
    axes[2].set_title("Gate Selectivity")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.savefig(output_dir / "synthetic_training_curves.png", dpi=150)
    plt.close(fig)


def plot_matrix(
    matrix: torch.Tensor,
    title: str,
    output_path: Path,
    x_labels: list[str],
    y_labels: list[str],
    value_format: str = ".2f",
) -> None:
    values = matrix.detach().cpu().float()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(values, cmap="viridis")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xticks(range(len(x_labels)), x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)), y_labels)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                format(values[row, col].item(), value_format),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    dataset = SyntheticFrequencyDataset(
        SyntheticConfig(
            num_samples=args.num_samples,
            duration=args.duration,
            snr_db=args.snr_db,
            difficulty=args.difficulty,
            seed=args.dataset_seed,
        )
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2)

    model = FrequencyGatedClassifier(
        num_classes=4,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        learnable_filters=args.learnable_filters,
        filter_bands=get_filter_bands(args.filter_bank),
        gate_mode=args.gate_mode,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cls_loss, train_acc, train_gate_mean, train_active_bands = run_epoch(
            model,
            train_loader,
            optimizer,
            gate_l1=args.gate_l1,
            active_gate_threshold=args.active_gate_threshold,
            device=device,
        )
        val_loss, val_cls_loss, val_acc, val_gate_mean, val_active_bands = run_epoch(
            model,
            val_loader,
            gate_l1=args.gate_l1,
            active_gate_threshold=args.active_gate_threshold,
            device=device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_acc": train_acc,
                "train_gate_mean": train_gate_mean,
                "train_active_bands": train_active_bands,
                "val_loss": val_loss,
                "val_cls_loss": val_cls_loss,
                "val_acc": val_acc,
                "val_gate_mean": val_gate_mean,
                "val_active_bands": val_active_bands,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"train_gate={train_gate_mean:.3f} train_active={train_active_bands:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"val_gate={val_gate_mean:.3f} val_active={val_active_bands:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model_dir = args.output_dir / "models"
    figure_dir = args.output_dir / "figures"
    metrics_dir = args.output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "frequency_gated_synthetic_best.pt")
    save_history(history, metrics_dir)
    plot_history(history, figure_dir)

    confusion, gates_by_class, energy_by_class, active_bands_by_class = collect_diagnostics(
        model,
        val_loader,
        num_classes=4,
        active_gate_threshold=args.active_gate_threshold,
        device=device,
    )
    torch.save(
        {
            "confusion": confusion,
            "gates_by_class": gates_by_class,
            "energy_by_class": energy_by_class,
            "active_bands_by_class": active_bands_by_class,
            "active_gate_threshold": args.active_gate_threshold,
            "gate_l1": args.gate_l1,
        },
        metrics_dir / "synthetic_diagnostics.pt",
    )
    plot_matrix(
        confusion,
        "Synthetic validation confusion matrix",
        figure_dir / "synthetic_confusion_matrix.png",
        x_labels=CLASS_NAMES,
        y_labels=CLASS_NAMES,
        value_format=".0f",
    )
    plot_matrix(
        gates_by_class,
        "Mean gate by class",
        figure_dir / "synthetic_gates_by_class.png",
        x_labels=BAND_LABELS[args.filter_bank],
        y_labels=CLASS_NAMES,
    )
    plot_matrix(
        energy_by_class,
        "Mean log-energy by class",
        figure_dir / "synthetic_energy_by_class.png",
        x_labels=BAND_LABELS[args.filter_bank],
        y_labels=CLASS_NAMES,
    )

    print(f"best_val_acc={best_val_acc:.3f}")
    print(f"final_val_gate_mean={history[-1]['val_gate_mean']:.3f}")
    print(f"final_val_active_bands={history[-1]['val_active_bands']:.2f}")
    print(f"saved_model={model_dir / 'frequency_gated_synthetic_best.pt'}")
    print(f"saved_figures={figure_dir}")
    print(f"saved_metrics={metrics_dir}")


if __name__ == "__main__":
    main()
