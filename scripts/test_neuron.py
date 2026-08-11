from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.blocks import EnergyGatedFrequencyNeuron
from src.models import FrequencyGatedClassifier
from src.utils import generate_synthetic_frequency_batch


def main() -> None:
    torch.manual_seed(42)

    x, labels = generate_synthetic_frequency_batch(batch_size=8, seed=42)
    neuron = EnergyGatedFrequencyNeuron(sample_rate=16_000, kernel_size=101)
    y, gates, energy = neuron(x)

    model = FrequencyGatedClassifier(num_classes=4)
    outputs = model(x)

    print("Input:", tuple(x.shape))
    print("Labels:", tuple(labels.shape))
    print("Band outputs:", tuple(y.shape))
    print("Gates:", tuple(gates.shape))
    print("Energy:", tuple(energy.shape))
    print("Logits:", tuple(outputs["logits"].shape))

    assert y.shape == (8, neuron.num_filters, x.shape[-1])
    assert gates.shape == (8, neuron.num_filters)
    assert energy.shape == (8, neuron.num_filters)
    assert outputs["logits"].shape == (8, 4)
    assert torch.isfinite(outputs["logits"]).all()

    print("Forward pass OK.")


if __name__ == "__main__":
    main()
