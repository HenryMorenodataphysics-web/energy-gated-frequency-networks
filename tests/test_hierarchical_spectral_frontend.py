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
    assert outputs["activation_signature"].shape == (2, 7, 5)
    assert torch.all((outputs["macro_gates"] >= 0) & (outputs["macro_gates"] <= 1))
    assert torch.all((outputs["subband_gates"] >= 0) & (outputs["subband_gates"] <= 1))
    assert torch.allclose(
        outputs["joint_gates"],
        outputs["macro_gates"][:, outputs["subband_macro_index"], :]
        * outputs["subband_gates"],
    )


@pytest.mark.parametrize("gate_mode", ["none", "macro", "subband", "hierarchical"])
def test_gate_ablation_modes_apply_the_declared_routing(gate_mode: str) -> None:
    frontend = HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
        gate_mode=gate_mode,
        normalize_gate_inputs=True,
        conditional_subgates=True,
    )
    output = frontend(torch.randn(2, 1, 4_000))
    parent = output["macro_gates"][:, output["subband_macro_index"], :]
    expected = {
        "none": torch.ones_like(output["subband_gates"]),
        "macro": parent,
        "subband": output["subband_gates"],
        "hierarchical": parent * output["subband_gates"],
    }[gate_mode]

    assert frontend.child_gate.in_channels == 4
    assert torch.allclose(output["joint_gates"], expected)


def test_activation_signature_records_soft_activation_and_duration() -> None:
    gates = torch.tensor([[[0.2, 0.6, 0.8, 0.4]]])
    energy = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])

    signature = HierarchicalSpectralFrontend.activation_signature(gates, energy)

    assert signature.shape == (1, 1, 5)
    assert signature[0, 0, 0].item() == pytest.approx(0.5)
    assert signature[0, 0, 1].item() == pytest.approx(0.8)
    assert signature[0, 0, 3].item() == pytest.approx(0.5)


def test_tone_is_localized_inside_its_physical_subband() -> None:
    frontend = build_frontend()
    outputs = frontend(sine_wave(500.0))
    mean_energy = outputs["subband_energy"].mean(dim=-1).squeeze(0)
    dominant = int(mean_energy.argmax())
    low = float(outputs["subband_edges_hz"][dominant])
    high = float(outputs["subband_edges_hz"][dominant + 1])

    assert low <= 500.0 <= high


def test_opposite_phase_channels_are_combined_in_power_domain() -> None:
    frontend = build_frontend()
    tone = sine_wave(500.0)
    opposite_phase = torch.cat((tone, -tone), dim=1)

    mono = frontend(tone)
    multichannel = frontend(opposite_phase)

    assert torch.allclose(multichannel["spectrogram"], mono["spectrogram"], atol=1e-6)
    assert torch.allclose(
        multichannel["subband_energy"],
        mono["subband_energy"],
        atol=1e-6,
    )


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


def test_learnable_subband_weights_start_fixed_and_cannot_cross_boundaries() -> None:
    fixed = build_frontend()
    learnable = HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
        learnable_subband_weights=True,
    )

    assert torch.allclose(
        learnable.effective_subband_weights(), fixed.subband_mask, atol=1e-7
    )
    with torch.no_grad():
        learnable.subband_weight_logits.normal_()
    weights = learnable.effective_subband_weights()
    assert torch.all(weights[~learnable.subband_support] == 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(learnable.num_subbands))


def test_learnable_subband_weights_receive_reconstruction_gradient() -> None:
    frontend = HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
        learnable_subband_weights=True,
    )
    output = frontend(torch.randn(2, 1, 4_000))
    output["subband_log_energy"].square().mean().backward()

    gradient = frontend.subband_weight_logits.grad
    assert gradient is not None
    assert torch.any(gradient[frontend.subband_support] != 0)
    assert torch.all(gradient[~frontend.subband_support] == 0)


def test_hard_macro_routing_executes_only_selected_temporal_branches() -> None:
    frontend = HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
        gate_mode="macro",
    )
    observed_rows: list[int] = []
    handle = frontend.shared_temporal_transform.register_forward_pre_hook(
        lambda _module, inputs: observed_rows.append(inputs[0].shape[0])
    )
    frontend.eval()
    frontend.set_hard_routing_top_k(1)
    output = frontend(torch.randn(2, 1, 4_000))
    handle.remove()

    active_subbands = output["active_subband_mask"]
    assert torch.all(output["active_macro_mask"].sum(dim=1) == 1)
    assert observed_rows == [int(active_subbands.sum())]
    assert observed_rows[0] < 2 * frontend.num_subbands
    assert torch.all(output["features"].permute(0, 2, 1, 3)[~active_subbands] == 0)
    assert torch.all(output["joint_gates"][~active_subbands] == 0)


def test_hard_macro_routing_is_inference_only() -> None:
    frontend = build_frontend()
    frontend.gate_mode = "macro"
    frontend.set_hard_routing_top_k(1)

    with pytest.raises(RuntimeError, match="inference-only"):
        frontend(torch.randn(1, 1, 4_000))


def test_harmonic_matrices_contain_only_non_adjacent_frequency_multiples() -> None:
    frontend = HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
        subbands_per_macro=(2, 3, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
        harmonic_context=True,
    )
    edges = frontend.subband_edges_hz
    centers = 0.5 * (edges[:-1] + edges[1:])

    for ratio, matrix in (
        (2.0, frontend.second_harmonic_matrix),
        (3.0, frontend.third_harmonic_matrix),
    ):
        pairs = torch.nonzero(matrix, as_tuple=False)
        assert pairs.numel() > 0
        for target, source in pairs.tolist():
            assert abs(target - source) > 1
            harmonic = ratio * centers[source]
            assert edges[target] <= harmonic < edges[target + 1]


def test_zero_initialized_harmonic_context_matches_baseline_and_receives_gradient() -> None:
    arguments = {
        "sample_rate": 8_000,
        "macro_edges_hz": (0.0, 666.6667, 2_666.6667, 4_000.0),
        "subbands_per_macro": (2, 3, 2),
        "n_fft": 256,
        "hop_length": 64,
        "temporal_channels": 3,
    }
    torch.manual_seed(31)
    baseline = HierarchicalSpectralFrontend(**arguments)
    torch.manual_seed(31)
    harmonic = HierarchicalSpectralFrontend(**arguments, harmonic_context=True)
    waveform = torch.randn(2, 1, 4_000)

    baseline_output = baseline(waveform)
    harmonic_output = harmonic(waveform)
    assert torch.allclose(
        harmonic_output["features"], baseline_output["features"], atol=1e-7
    )

    harmonic_output["features"].square().mean().backward()
    assert harmonic.second_harmonic_scale.grad is not None
    assert harmonic.third_harmonic_scale.grad is not None
    assert torch.any(harmonic.second_harmonic_scale.grad != 0)
    assert torch.any(harmonic.third_harmonic_scale.grad != 0)
    assert sum(parameter.numel() for parameter in harmonic.parameters()) == (
        sum(parameter.numel() for parameter in baseline.parameters())
        + 2 * harmonic.temporal_channels
    )
