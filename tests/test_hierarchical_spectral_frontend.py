from __future__ import annotations

import numpy as np
import pytest
import torch

from src.blocks import HierarchicalSpectralFrontend


def build_frontend() -> HierarchicalSpectralFrontend:
    return HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
    )


def sine_wave(frequency_hz: float, sample_rate: int = 8_000) -> torch.Tensor:
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    return torch.sin(2.0 * torch.pi * frequency_hz * time).view(1, 1, -1)


def test_frontend_shapes_and_gate_ranges() -> None:
    frontend = build_frontend()
    outputs = frontend(torch.randn(2, 1, 4_000))
    frames = outputs["spectrogram"].shape[-1]

    assert outputs["features"].shape == (2, 3, 7, frames)
    assert outputs["macro_gates"].shape == (2, 3, frames)
    assert outputs["subband_gates"].shape == (2, 7, frames)
    assert outputs["joint_gates"].shape == (2, 7, frames)
    assert outputs["subband_energy"].shape == (2, 7, frames)
    assert torch.all((outputs["macro_gates"] >= 0) & (outputs["macro_gates"] <= 1))
    assert torch.all((outputs["subband_gates"] >= 0) & (outputs["subband_gates"] <= 1))
    assert torch.allclose(
        outputs["joint_gates"],
        outputs["macro_gates"][:, outputs["subband_macro_index"], :]
        * outputs["subband_gates"],
    )


def test_tone_is_localized_inside_its_physical_subband() -> None:
    frontend = build_frontend()
    outputs = frontend(sine_wave(500.0))
    mean_energy = outputs["subband_energy"].mean(dim=-1).squeeze(0)
    dominant = int(mean_energy.argmax())
    low = float(outputs["subband_edges_hz"][dominant])
    high = float(outputs["subband_edges_hz"][dominant + 1])

    assert low <= 500.0 <= high


def test_frontend_is_differentiable() -> None:
    frontend = build_frontend()
    waveform = torch.randn(2, 1, 4_000, requires_grad=True)
    outputs = frontend(waveform)
    loss = outputs["features"].square().mean() + outputs["joint_gates"].mean()
    loss.backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    gradients = [parameter.grad for parameter in frontend.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_neighbor_context_reaches_adjacent_subbands() -> None:
    frontend = build_frontend()
    features = torch.zeros(1, frontend.temporal_channels, frontend.num_subbands, 3)
    features[:, :, 3, :] = 1.0
    with torch.no_grad():
        frontend.neighbor_context.weight.fill_(1.0)
        context = frontend.neighbor_context(features)

    assert torch.all(context[:, :, 2, :] > 0)
    assert torch.all(context[:, :, 4, :] > 0)


def test_frontend_stays_compact() -> None:
    frontend = build_frontend()
    parameter_count = sum(parameter.numel() for parameter in frontend.parameters())

    assert 0 < parameter_count < 1_000


@pytest.mark.parametrize(
    "macro_edges",
    [
        (0.0, 1_000.0, 2_000.0),
        (0.0, 1_000.0, 900.0, 4_000.0),
        (0.0, 1_000.0, 2_000.0, 4_100.0),
    ],
)
def test_invalid_macro_edges_are_rejected(macro_edges: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        HierarchicalSpectralFrontend(
            sample_rate=8_000,
            macro_edges_hz=macro_edges,
            subbands_per_macro=(2, 2, 2),
            n_fft=256,
        )


def test_subband_edges_are_strictly_ordered() -> None:
    frontend = build_frontend()
    edges = frontend.subband_edges_hz.detach().cpu().numpy()

    assert len(edges) == frontend.num_subbands + 1
    assert np.all(np.diff(edges) > 0)
