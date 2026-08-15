from __future__ import annotations

import torch

from src.anomaly import (
    ConditionedSpectralBaselineEstimator,
    LogMelFrontend,
    build_mel_filterbank,
)


def test_log_mel_frontend_is_finite_and_preserves_batch() -> None:
    frontend = LogMelFrontend(
        sample_rate=8_000,
        n_fft=256,
        hop_length=64,
        n_mels=16,
    )
    waveform = torch.randn(3, 2, 4_000)

    output = frontend(waveform)

    assert output.shape[:2] == (3, 16)
    assert torch.isfinite(output).all()
    assert torch.allclose(
        build_mel_filterbank(8_000, 256, 16).sum(dim=1),
        torch.ones(16),
        atol=1e-6,
    )


def test_normal_only_spectral_baselines_detect_an_off_subspace_event() -> None:
    generator = torch.Generator().manual_seed(3)
    normal = torch.randn(8, 4, 20, generator=generator) * 0.05
    normal[:, 0] += torch.linspace(-1.0, 1.0, 20)
    estimator = ConditionedSpectralBaselineEstimator(
        n_mels=4,
        memory_size=32,
        pca_rank=1,
        top_fraction=0.25,
        seed=5,
    )
    estimator.update(normal, ["machine"] * normal.shape[0])
    baselines = estimator.finalize()
    normal_query = normal[:1]
    anomalous_query = normal_query.clone()
    anomalous_query[:, 2, 10:15] += 5.0

    normal_scores = baselines.score(normal_query, ["machine"])
    anomalous_scores = baselines.score(anomalous_query, ["machine"])

    for score_name in (
        "logmel_gaussian_score",
        "logmel_knn_score",
        "logmel_pca_reconstruction_score",
    ):
        assert anomalous_scores[score_name].item() > normal_scores[score_name].item()
