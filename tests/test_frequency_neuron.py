from __future__ import annotations

import torch

from src.blocks import EnergyGatedFrequencyNeuron
from src.models import (
    FrequencyGatedClassifier,
    FrequencyGatedTemporalClassifier,
    MatchedConvTemporalClassifier,
)
from src.utils import generate_synthetic_frequency_batch


def test_neuron_shapes() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=4, duration=0.25, seed=123)
    neuron = EnergyGatedFrequencyNeuron(sample_rate=16_000, kernel_size=101)

    y, gates, energy = neuron(x)

    assert y.shape == (4, neuron.num_filters, x.shape[-1])
    assert gates.shape == (4, neuron.num_filters)
    assert energy.shape == (4, neuron.num_filters)
    assert torch.isfinite(y).all()
    assert torch.isfinite(gates).all()
    assert torch.isfinite(energy).all()


def test_classifier_forward() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=4, duration=0.25, seed=456)
    model = FrequencyGatedClassifier(num_classes=4)

    outputs = model(x)

    assert outputs["logits"].shape == (4, 4)
    assert outputs["gates"].shape[1] == model.frontend.num_filters
    assert torch.isfinite(outputs["logits"]).all()


def test_contextual_gate_forward() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=4, duration=0.25, seed=789)
    model = FrequencyGatedClassifier(
        num_classes=4,
        gate_mode="contextual",
        gate_hidden_dim=16,
    )

    outputs = model(x)

    assert model.frontend.gate_mode == "contextual"
    assert outputs["logits"].shape == (4, 4)
    assert outputs["gates"].shape == (4, model.frontend.num_filters)
    assert torch.isfinite(outputs["gates"]).all()


def test_no_gate_is_an_identity_modulation() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=4, duration=0.25, seed=246)
    neuron = EnergyGatedFrequencyNeuron(
        sample_rate=16_000,
        kernel_size=101,
        gate_mode="none",
    )

    y, gates, energy = neuron(x)
    filtered = torch.nn.functional.conv1d(
        x,
        neuron.get_filters(),
        padding=neuron.kernel_size // 2,
    )
    expected = torch.nn.functional.gelu(
        filtered + neuron.bias.view(1, -1, 1)
    )

    assert neuron.gate_alpha is None
    assert neuron.gate_beta is None
    assert torch.equal(gates, torch.ones_like(energy))
    assert torch.allclose(y, expected)


def test_temporal_classifier_forward() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=4, duration=0.25, seed=987)
    model = FrequencyGatedTemporalClassifier(num_classes=4)

    outputs = model(x)

    assert outputs["logits"].shape == (4, 4)
    assert outputs["band_outputs"].shape[:2] == (4, model.frontend.num_filters)
    assert torch.isfinite(outputs["logits"]).all()


def test_sinc_filters_are_physical_and_differentiable() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=2, duration=0.1, seed=654)
    neuron = EnergyGatedFrequencyNeuron(
        sample_rate=16_000,
        kernel_size=101,
        filter_mode="sinc",
    )

    y, gates, energy = neuron(x)
    low, high = neuron.cutoff_frequencies_hz()
    filters = neuron.get_filters().squeeze(1)
    (y.mean() + gates.mean() + energy.mean()).backward()

    assert torch.all(low > 0)
    assert torch.all(high < 8_000)
    assert torch.all(high - low >= neuron.min_bandwidth_hz)
    assert torch.allclose(filters, filters.flip(-1), atol=1e-6)
    assert neuron.low_cutoff_raw.grad is not None
    assert neuron.bandwidth_raw.grad is not None
    assert torch.isfinite(neuron.low_cutoff_raw.grad).all()
    assert torch.isfinite(neuron.bandwidth_raw.grad).all()


def test_matched_conv_uses_same_head_capacity() -> None:
    x, _ = generate_synthetic_frequency_batch(batch_size=2, duration=0.1, seed=321)
    egfn = FrequencyGatedTemporalClassifier(
        num_classes=2,
        filter_mode="free",
        frontend_channels=32,
        temporal_channels=(64, 128),
    )
    matched = MatchedConvTemporalClassifier(
        num_classes=2,
        num_frontend_filters=egfn.frontend.num_filters,
        frontend_channels=32,
        temporal_channels=(64, 128),
    )

    egfn_outputs = egfn(x)
    matched_outputs = matched(x)
    egfn_parameters = sum(parameter.numel() for parameter in egfn.parameters())
    matched_parameters = sum(parameter.numel() for parameter in matched.parameters())

    assert egfn_outputs["logits"].shape == matched_outputs["logits"].shape == (2, 2)
    assert egfn_outputs["band_outputs"].shape == matched_outputs["band_outputs"].shape
    assert abs(egfn_parameters - matched_parameters) <= 32
