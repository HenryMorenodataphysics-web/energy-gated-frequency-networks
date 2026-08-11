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

from src.data.fsdd_dataset import (
    FSDDDataset,
    find_fsdd_recordings,
    split_records_by_speaker,
    split_records_random,
)
from src.models import (
    Conv1DBaseline,
    FrequencyGatedClassifier,
    FrequencyGatedTemporalClassifier,
    get_filter_bands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on Free Spoken Digit Dataset.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "fsdd")
    parser.add_argument("--model", choices=["egfn", "egfn_temporal", "conv1d"], default="egfn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--target-sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-channels", type=str, default="32,64")
    parser.add_argument("--filter-bank", choices=["default", "fine"], default="fine")
    parser.add_argument("--learnable-filters", action="store_true")
    parser.add_argument(
        "--gate-mode",
        choices=["none", "independent", "contextual"],
        default="independent",
    )
    parser.add_argument("--gate-l1", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--min-gain", type=float, default=0.75)
    parser.add_argument("--max-gain", type=float, default=1.25)
    parser.add_argument("--max-shift-fraction", type=float, default=0.08)
    parser.add_argument("--max-mask-fraction", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--split", choices=["random", "speaker"], default="speaker")
    parser.add_argument("--val-speakers", type=str, default="nicolas")
    parser.add_argument("--test-speakers", type=str, default="jackson")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use cuda when available by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fsdd_egfn")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def parse_speakers(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "conv1d":
        return Conv1DBaseline(num_classes=10)

    if args.model == "egfn_temporal":
        temporal_channels = tuple(int(value.strip()) for value in args.temporal_channels.split(","))
        if len(temporal_channels) != 2:
            raise ValueError("--temporal-channels must contain two comma-separated integers.")
        return FrequencyGatedTemporalClassifier(
            num_classes=10,
            sample_rate=args.target_sample_rate,
            temporal_channels=(temporal_channels[0], temporal_channels[1]),
            learnable_filters=args.learnable_filters,
            filter_bands=get_filter_bands(args.filter_bank),
            gate_mode=args.gate_mode,
        )

    return FrequencyGatedClassifier(
        num_classes=10,
        sample_rate=args.target_sample_rate,
        hidden_dim=args.hidden_dim,
        learnable_filters=args.learnable_filters,
        filter_bands=get_filter_bands(args.filter_bank),
        gate_mode=args.gate_mode,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    gate_l1: float = 0.0,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_gate_mean = 0.0
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
            gates = outputs.get("gates")
            gate_mean = gates.mean() if gates is not None else torch.tensor(0.0)
            loss = cls_loss + gate_l1 * gate_mean

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (outputs["logits"].argmax(dim=-1) == y).sum().item()
        total_gate_mean += float(gate_mean.item()) * batch_size
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        total_gate_mean / total_examples,
    )


def collect_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    confusion = torch.zeros(10, 10, dtype=torch.long)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            predictions = outputs["logits"].argmax(dim=-1)
            for true_label, predicted_label in zip(y, predictions):
                confusion[true_label.cpu(), predicted_label.cpu()] += 1

    return confusion


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("FSDD Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("FSDD Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(confusion: torch.Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    values = confusion.float()
    image = ax.imshow(values, cmap="viridis")
    fig.colorbar(image, ax=ax)
    ax.set_title("FSDD test confusion matrix")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(int(confusion[row, col])), ha="center", va="center", fontsize=7)

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

    records = find_fsdd_recordings(args.data_dir)
    speakers = sorted({record.speaker for record in records})
    print(f"records={len(records)} speakers={speakers}")

    if args.split == "speaker":
        train_records, val_records, test_records = split_records_by_speaker(
            records,
            test_speakers=parse_speakers(args.test_speakers),
            val_speakers=parse_speakers(args.val_speakers),
        )
    else:
        train_records, val_records, test_records = split_records_random(
            records,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    if not train_records or not val_records or not test_records:
        raise SystemExit("Empty split detected. Adjust --val-speakers/--test-speakers or use --split random.")

    train_loader = DataLoader(
        FSDDDataset(
            train_records,
            args.target_sample_rate,
            args.duration,
            augment=args.augment,
            noise_std=args.noise_std,
            gain_range=(args.min_gain, args.max_gain),
            max_shift_fraction=args.max_shift_fraction,
            max_mask_fraction=args.max_mask_fraction,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        FSDDDataset(val_records, args.target_sample_rate, args.duration),
        batch_size=args.batch_size * 2,
    )
    test_loader = DataLoader(
        FSDDDataset(test_records, args.target_sample_rate, args.duration),
        batch_size=args.batch_size * 2,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_gate = run_epoch(
            model,
            train_loader,
            optimizer,
            args.gate_l1,
            device,
        )
        val_loss, val_acc, val_gate = run_epoch(
            model,
            val_loader,
            gate_l1=args.gate_l1,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_gate": train_gate,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_gate": val_gate,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_gate={val_gate:.3f}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stopping_epoch={epoch} best_val_acc={best_val_acc:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_gate = run_epoch(
        model,
        test_loader,
        gate_l1=args.gate_l1,
        device=device,
    )
    print(f"best_val_acc={best_val_acc:.3f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} test_gate={test_gate:.3f}")

    model_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figure_dir = args.output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"fsdd_{args.model}_best.pt")
    save_history(history, metrics_dir)
    plot_history(history, figure_dir)

    confusion = collect_confusion(model, test_loader, device)
    torch.save({"confusion": confusion, "test_acc": test_acc}, metrics_dir / "test_diagnostics.pt")
    plot_confusion(confusion, figure_dir / "test_confusion_matrix.png")

    print(f"saved_model={model_dir / f'fsdd_{args.model}_best.pt'}")
    print(f"saved_metrics={metrics_dir}")
    print(f"saved_figures={figure_dir}")


if __name__ == "__main__":
    main()
