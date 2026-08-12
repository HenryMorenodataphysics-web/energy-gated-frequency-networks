from __future__ import annotations

import torch

from src.anomaly import NormalProfileEstimator
from src.blocks import HierarchicalSpectralFrontend
from src.models import CompactHierarchicalEncoder, HierarchicalAnomalyDetector


def build_frontend() -> HierarchicalSpectralFrontend:
    return HierarchicalSpectralFrontend(
        sample_rate=8_000,
        macro_edges_hz=(0.0, 1_000.0, 2_500.0, 4_000.0),
        subbands_per_macro=(2, 2, 2),
        n_fft=256,
        hop_length=64,
        temporal_channels=3,
    )


def build_detector() -> HierarchicalAnomalyDetector:
    frontend = build_frontend()
    estimator = NormalProfileEstimator(num_subbands=frontend.num_subbands, minimum_std=0.1)
    normal_energy = frontend(torch.randn(2, 1, 4_000))["subband_energy"].detach()
    estimator.update(normal_energy, ["machine", "machine"], ["normal", "normal"])
    profile = estimator.finalize(
        metadata={
            "frontend": {
                "sample_rate": frontend.sample_rate,
                "n_fft": frontend.n_fft,
                "hop_length": frontend.hop_length,
                "macro_edges_hz": frontend.macro_edges_hz.tolist(),
                "subbands_per_macro": list(frontend.subbands_per_macro),
            }
        }
    )
    return HierarchicalAnomalyDetector(frontend, profile, embedding_channels=8)


def test_compact_encoder_shapes_and_parameter_budget() -> None:
    encoder = CompactHierarchicalEncoder(frontend_channels=4, embedding_channels=8)
    output = encoder(torch.randn(2, 4, 16, 20), torch.randn(2, 16, 2, 20))

    assert output["embedding_map"].shape == (2, 8, 16, 20)
    assert output["sustained_embedding"].shape == (2, 24)
    assert output["event_embedding"].shape == (2, 24)
    assert output["subband_embedding"].shape == (2, 32)
    assert output["embedding"].shape == (2, 48)
    assert sum(parameter.numel() for parameter in encoder.parameters()) < 2_000


def test_detector_keeps_score_and_embedding_as_separate_outputs() -> None:
    detector = build_detector()
    waveform = torch.randn(2, 1, 4_000)
    output = detector(waveform, ["machine", "unknown"])
    with torch.no_grad():
        for parameter in detector.encoder.parameters():
            parameter.add_(10.0)
    changed_encoder_output = detector(waveform, ["machine", "unknown"])

    assert output["recording_score"].shape == (2,)
    assert output["embedding"].shape == (2, 28)
    assert output["known_condition"].tolist() == [True, False]
    assert output["gate_regularization_loss"].ndim == 0
    assert torch.allclose(
        changed_encoder_output["recording_score"],
        output["recording_score"],
    )


def test_detector_rejects_incompatible_profile_signature() -> None:
    detector = build_detector()
    incompatible = build_frontend()
    incompatible.hop_length = 32

    try:
        HierarchicalAnomalyDetector(incompatible, detector.normal_profile)
    except ValueError as error:
        assert "hop_length" in str(error)
    else:
        raise AssertionError("an incompatible profile signature was accepted")


def test_detector_reconstructs_masked_subbands_without_exposing_profile_input() -> None:
    detector = build_detector()
    waveform = torch.randn(2, 1, 4_000)
    mask = torch.zeros(2, detector.frontend.num_subbands, dtype=torch.bool)
    mask[:, 2:4] = True
    output = detector(waveform, ["machine", "machine"], masked_subbands=mask)

    assert output["reconstructed_log_energy_z"].shape == output["z_scores"][:, :, 0].shape
    loss = torch.nn.functional.smooth_l1_loss(
        output["reconstructed_log_energy_z"][mask],
        output["z_scores"][:, :, 0][mask].detach(),
    )
    loss.backward()
    assert detector.reconstruction_head.weight.grad is not None
