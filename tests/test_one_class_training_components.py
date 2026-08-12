from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.io import wavfile

from src.anomaly import (
    ConditionedEmbeddingEstimator,
    ConditionedEmbeddingProfile,
    anti_collapse_loss,
    deep_svdd_loss,
    deep_svdd_scores,
    stabilize_center,
    standardized_embedding_scores,
)
from src.data import AnomalyAudioRecord, AnomalyWindowDataset
from src.models import Conv1DAnomalyEncoder
from src.models import soft_event_pool
from scripts.train_mimii_one_class import is_meaningful_improvement, make_audio_view


def test_deep_svdd_scores_and_center_stabilization() -> None:
    center = stabilize_center(torch.tensor([0.0, -0.01, 2.0]), epsilon=0.1)
    embedding = torch.stack((center, center + 1.0))

    assert center.tolist() == pytest.approx([0.1, -0.1, 2.0])
    assert deep_svdd_scores(embedding, center).tolist() == pytest.approx([0.0, 1.0])
    assert deep_svdd_loss(embedding, center).item() == pytest.approx(0.5)


def test_early_stopping_requires_meaningful_relative_improvement() -> None:
    assert is_meaningful_improvement(0.99, 1.0, 0.005)
    assert not is_meaningful_improvement(0.995, 1.0, 0.01)
    assert is_meaningful_improvement(0.989, 1.0, 0.01)


def test_anti_collapse_objective_penalizes_constant_embeddings() -> None:
    collapsed = torch.zeros(8, 4, requires_grad=True)
    result = anti_collapse_loss(collapsed, collapsed, variance_target=0.05)
    result["representation_loss"].backward()

    assert result["variance_loss"].item() > 0
    assert result["embedding_std"].item() < 0.05
    assert collapsed.grad is not None


def test_standardized_score_uses_fitted_normal_scale() -> None:
    embedding = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    scores = standardized_embedding_scores(
        embedding,
        normal_mean=torch.tensor([1.0, 2.0]),
        normal_std=torch.tensor([2.0, 1.0]),
    )

    assert scores.tolist() == pytest.approx([2.0, 0.5])


def test_conditioned_embedding_profile_uses_condition_and_global_fallback() -> None:
    estimator = ConditionedEmbeddingEstimator(minimum_std=0.1)
    estimator.update(
        torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0]]),
        ["id_00", "id_00", "id_02", "id_02"],
    )
    profile = estimator.finalize()
    scores, _, known = profile.scores(
        torch.tensor([[1.0, 1.0], [11.0, 11.0], [6.0, 6.0]]),
        ["id_00", "id_02", "unknown"],
    )
    restored = ConditionedEmbeddingProfile.from_dict(profile.to_dict())

    assert scores.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert known.tolist() == [True, True, False]
    assert restored.condition_ids == profile.condition_ids
    assert torch.allclose(restored.fallback_mean, profile.fallback_mean)


def test_soft_event_pool_retains_a_brief_peak() -> None:
    features = torch.tensor([[[0.0, 0.0, 4.0, 0.0]]])
    pooled = soft_event_pool(features, temperature=0.1)

    assert pooled.item() > features.mean().item() + 2.0


def test_audio_views_preserve_shape_and_approximately_preserve_energy() -> None:
    waveform = torch.randn(4, 2, 1_000)
    view = make_audio_view(waveform, noise_fraction=0.001, max_shift_fraction=0.05)

    assert view.shape == waveform.shape
    assert view.square().mean().item() == pytest.approx(
        waveform.square().mean().item(),
        rel=0.01,
    )


def test_conv1d_encoder_shares_weights_across_audio_channels() -> None:
    model = Conv1DAnomalyEncoder(embedding_channels=8)
    mono = torch.randn(2, 1, 4_000)
    duplicated = mono.repeat(1, 3, 1)

    model.eval()
    mono_output = model(mono)
    duplicated_output = model(duplicated)

    assert duplicated_output["channel_embeddings"].shape == (2, 3, 8)
    assert duplicated_output["embedding"].shape == (2, 16)
    assert torch.allclose(mono_output["embedding"], duplicated_output["embedding"], atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000


def test_conv1d_embedding_does_not_change_between_train_and_eval_modes() -> None:
    model = Conv1DAnomalyEncoder(embedding_channels=8)
    waveform = torch.randn(4, 2, 4_000)

    model.train()
    training_embedding = model(waveform)["embedding"]
    model.eval()
    evaluation_embedding = model(waveform)["embedding"]

    assert torch.allclose(training_embedding, evaluation_embedding, atol=1e-6)


def test_window_dataset_preserves_channels_and_reads_only_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(".tmp") / "one_class_window_test.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(20, dtype=np.int16)
    audio = np.stack((samples, -samples), axis=1)
    wavfile.write(path, 10, audio)
    original_read = sf.read
    requested_frames: list[int] = []

    def tracked_read(*args, **kwargs):
        requested_frames.append(int(kwargs["frames"]))
        return original_read(*args, **kwargs)

    monkeypatch.setattr("src.data.anomaly_window_dataset.sf.read", tracked_read)
    record = AnomalyAudioRecord(
        path=path,
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00/condition",
        group_id="recording",
        label="normal",
    )
    try:
        dataset = AnomalyWindowDataset(
            [record],
            target_sample_rate=10,
            duration_seconds=1.0,
            crop_mode="grid",
            evaluation_windows=2,
        )
        first = dataset[0]
        last = dataset[1]
    finally:
        path.unlink(missing_ok=True)

    assert first["waveform"].shape == (2, 10)
    assert requested_frames == [10, 10]
    assert first["waveform"][0, 0].item() == pytest.approx(0.0)
    assert last["waveform"][0, 0].item() == pytest.approx(10 / 32768)
    assert first["condition_id"] == "test/machine/id_00/condition"
