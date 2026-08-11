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

from src.data import DEFAULT_SPEECH_COMMANDS_LABELS, SpeechCommandsSubset
from src.models import (
    Conv1DBaseline,
    FrequencyGatedClassifier,
    FrequencyGatedTemporalClassifier,
    get_filter_bands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on Google Speech Commands.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "speech_commands")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", choices=["egfn", "egfn_temporal", "conv1d"], default="egfn_temporal")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_SPEECH_COMMANDS_LABELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--target-sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-channels", type=str, default="32,64")
    parser.add_argument("--frontend-channels", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--filter-bank", choices=["default", "fine"], default="fine")
    parser.add_argument("--learnable-filters", action="store_true")
    parser.add_argument(
        "--gate-mode",
        choices=["none", "independent", "contextual"],
        default="independent",
    )
    parser.add_argument("--gate-l1", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "speech_commands_egfn_temporal")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in value.split(",") if label.strip())
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def build_model(args: argparse.Namespace, num_classes: int) -> torch.nn.Module:
    if args.model == "conv1d":
        return Conv1DBaseline(num_classes=num_classes)

    if args.model == "egfn_temporal":
        temporal_channels = tuple(int(value.strip()) for value in args.temporal_channels.split(","))
        if len(temporal_channels) != 2:
            raise ValueError("--temporal-channels must contain two comma-separated integers.")
        return FrequencyGatedTemporalClassifier(
            num_classes=num_classes,
            sample_rate=args.target_sample_rate,
            temporal_channels=(temporal_channels[0], temporal_channels[1]),
            frontend_channels=args.frontend_channels if args.frontend_channels > 0 else None,
            dropout=args.dropout,
            learnable_filters=args.learnable_filters,
            filter_bands=get_filter_bands(args.filter_bank),
            gate_mode=args.gate_mode,
        )

    return FrequencyGatedClassifier(
        num_classes=num_classes,
        sample_rate=args.target_sample_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learnable_filters=args.learnable_filters,
        filter_bands=get_filter_bands(args.filter_bank),
        gate_mode=args.gate_mode,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gate_l1: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_gate = 0.0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(x)
            cls_loss = F.cross_entropy(
                outputs["logits"],
                y,
                label_smoothing=label_smoothing if is_training else 0.0,
            )
            gates = outputs.get("gates")
            gate_mean = gates.mean() if gates is not None else torch.tensor(0.0, device=device)
            loss = cls_loss + gate_l1 * gate_mean

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (outputs["logits"].argmax(dim=-1) == y).sum().item()
        total_gate += gate_mean.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples, total_gate / total_examples


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
    axes[0].set_title("Speech Commands Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Speech Commands Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def collect_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    labels = parse_labels(args.labels)
    device = resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"labels={labels}")

    train_set = SpeechCommandsSubset(
        args.data_dir,
        subset="training",
        labels=labels,
        download=args.download,
        target_sample_rate=args.target_sample_rate,
        duration=args.duration,
        augment=args.augment,
        noise_std=args.noise_std,
    )
    val_set = SpeechCommandsSubset(
        args.data_dir,
        subset="validation",
        labels=labels,
        download=False,
        target_sample_rate=args.target_sample_rate,
        duration=args.duration,
    )
    test_set = SpeechCommandsSubset(
        args.data_dir,
        subset="testing",
        labels=labels,
        download=False,
        target_sample_rate=args.target_sample_rate,
        duration=args.duration,
    )
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.05,
        )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_gate = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.gate_l1,
            args.label_smoothing,
        )
        val_loss, val_acc, val_gate = run_epoch(
            model,
            val_loader,
            device,
            gate_l1=args.gate_l1,
        )
        if scheduler is not None:
            scheduler.step()
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
        model.to(device)

    test_loss, test_acc, test_gate = run_epoch(model, test_loader, device, gate_l1=args.gate_l1)
    print(f"best_val_acc={best_val_acc:.3f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} test_gate={test_gate:.3f}")

    model_dir = args.output_dir / "models"
    metrics_dir = args.output_dir / "metrics"
    figure_dir = args.output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"speech_commands_{args.model}_best.pt")
    save_history(history, metrics_dir)
    plot_history(history, figure_dir)
    confusion = collect_confusion(model, test_loader, device, num_classes=len(labels))
    torch.save(
        {"confusion": confusion, "test_acc": test_acc, "labels": labels},
        metrics_dir / "test_diagnostics.pt",
    )

    print(f"saved_model={model_dir / f'speech_commands_{args.model}_best.pt'}")
    print(f"saved_metrics={metrics_dir}")
    print(f"saved_figures={figure_dir}")


if __name__ == "__main__":
    main()
