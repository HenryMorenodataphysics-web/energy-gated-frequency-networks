from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader

from src.anomaly import (
    ConditionedEmbeddingEstimator,
    ConditionedEmbeddingProfile,
    anti_collapse_loss,
    deep_svdd_loss,
    deep_svdd_scores,
    stabilize_center,
    standardized_embedding_scores,
)
from src.data import (
    AnomalyAudioRecord,
    AnomalyWindowDataset,
    ConditionBatchSampler,
    HybridConditionBatchSampler,
)
from src.models import Conv1DAnomalyEncoder
from src.models import soft_event_pool
from scripts.train_mimii_one_class import (
    apply_condition_thresholds,
    apply_hybrid_calibration,
    best_f1_threshold,
    bootstrap_normal_profile,
    build_model,
    configure_training_stage,
    complementary_subband_masks,
    evaluate_representation_epoch,
    fit_condition_thresholds,
    fit_hybrid_calibration,
    is_meaningful_improvement,
    load_pretrained_one_class_backbone,
    make_audio_view,
    objective_embedding,
    pairwise_anomaly_ranking_loss,
    representation_geometry,
    supervised_anomaly_loss,
    supervised_anomaly_objective,
    threshold_at_max_fpr,
    training_stage,
)
from src.blocks import HierarchicalSpectralFrontend


def test_capacity_matched_conv1d_has_exactly_304_parameters(tmp_path: Path) -> None:
    record = AnomalyAudioRecord(
        path=tmp_path / "normal.wav",
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00",
        group_id="normal.wav",
        label="normal",
    )
    profile = bootstrap_normal_profile(
        (record,),
        sample_rate=8_000,
        n_fft=256,
        hop_length=64,
        macro_edges_hz=None,
        subbands_per_macro=(2, 3, 2),
    )

    model = build_model("conv1d", profile, conv1d_channels=(2, 2, 7))

    assert sum(parameter.numel() for parameter in model.parameters()) == 304


def test_wide_egfn_has_about_one_thousand_parameters(tmp_path: Path) -> None:
    record = AnomalyAudioRecord(
        path=tmp_path / "normal.wav",
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00",
        group_id="normal.wav",
        label="normal",
    )
    profile = bootstrap_normal_profile(
        (record,),
        sample_rate=8_000,
        n_fft=256,
        hop_length=64,
        macro_edges_hz=None,
        subbands_per_macro=(2, 3, 2),
    )

    model = build_model("egfn", profile, egfn_embedding_channels=22)

    assert sum(parameter.numel() for parameter in model.parameters()) == 1_032


def test_supervised_anomaly_loss_updates_optional_head(tmp_path: Path) -> None:
    record = AnomalyAudioRecord(
        path=tmp_path / "normal.wav",
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00",
        group_id="normal.wav",
        label="normal",
    )
    profile = bootstrap_normal_profile(
        (record,), 8_000, 256, 64, None, (2, 3, 2)
    )
    model = build_model("egfn", profile, supervised_anomaly_head=True)
    output = model(torch.randn(4, 1, 4_000), ["test/machine/id_00"] * 4)

    loss = supervised_anomaly_loss(output, torch.tensor([0, 0, 1, 1]), 1.0)
    loss.backward()

    assert loss.item() > 0
    assert model.anomaly_head.weight.grad is not None


def test_pairwise_ranking_rewards_anomalies_above_normals() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    ordered = torch.tensor([-1.0, -0.5, 0.5, 1.0], requires_grad=True)
    reversed_logits = -ordered.detach()

    ordered_loss = pairwise_anomaly_ranking_loss(ordered, labels, margin=0.5)
    reversed_loss = pairwise_anomaly_ranking_loss(
        reversed_logits, labels, margin=0.5
    )
    objective = supervised_anomaly_objective(
        {"anomaly_logit": ordered},
        labels,
        positive_weight=1.0,
        ranking_weight=0.5,
        ranking_margin=0.5,
    )
    objective["supervised_loss"].backward()

    assert ordered_loss < reversed_loss
    assert objective["supervised_loss"].item() == pytest.approx(
        (objective["bce_loss"] + 0.5 * objective["ranking_loss"]).item()
    )
    assert ordered.grad is not None


def test_hybrid_calibration_uses_validation_labels_without_test_data() -> None:
    validation = {
        "recording_ids": [f"r{index}" for index in range(6)],
        "condition_ids": ["machine"] * 6,
        "labels": np.asarray([0, 0, 0, 1, 1, 1]),
        "scores": {
            "memory_score": np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            "supervised_score": np.asarray([0.2, 0.1, 0.3, 0.8, 0.7, 0.9]),
        },
    }

    calibration = fit_hybrid_calibration(validation, "memory_score", 0.95)
    scores = apply_hybrid_calibration(validation, calibration)
    threshold = best_f1_threshold(validation["labels"], scores)

    assert 0 <= calibration["alpha"] <= 1
    assert threshold == pytest.approx(calibration["threshold"])
    assert np.all(scores[:3] < scores[3:])


def test_hybrid_calibration_can_use_anomaly_reference_evidence() -> None:
    validation = {
        "recording_ids": [f"r{index}" for index in range(6)],
        "condition_ids": ["machine"] * 6,
        "labels": np.asarray([0, 0, 0, 1, 1, 1]),
        "scores": {
            "memory_score": np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            "supervised_score": np.asarray([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
            "reference_score": np.asarray([0.1, 0.2, 0.1, 0.8, 0.9, 0.85]),
        },
    }

    calibration = fit_hybrid_calibration(validation, "memory_score", 0.95)
    scores = apply_hybrid_calibration(validation, calibration)

    assert set(calibration["weights"]) == {
        "memory_score",
        "reference_score",
        "supervised_score",
    }
    assert sum(calibration["weights"].values()) == pytest.approx(1.0)
    assert calibration["weights"]["reference_score"] > 0
    assert np.max(scores[:3]) < np.min(scores[3:])


def test_fpr_constrained_threshold_respects_validation_cap() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.7, 0.9])

    threshold, metrics = threshold_at_max_fpr(labels, scores, 0.25)

    assert metrics["fp"] / (metrics["fp"] + metrics["tn"]) <= 0.25
    assert threshold == pytest.approx(0.4)
    assert metrics["recall"] == pytest.approx(1.0)


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


def test_learnable_filter_schedule_has_three_non_overlapping_stages() -> None:
    stages = [training_stage(epoch, True, 2, 3) for epoch in range(1, 8)]

    assert stages == [
        "filter_warmup",
        "filter_warmup",
        "filter_adaptation",
        "filter_adaptation",
        "filter_adaptation",
        "representation_finetune",
        "representation_finetune",
    ]
    assert training_stage(1, False, 0, 0) == "representation"
    assert training_stage(1, False, 0, 0, head_warmup_epochs=2) == "head_warmup"
    assert training_stage(3, False, 0, 0, head_warmup_epochs=2) == "representation"


def test_head_warmup_freezes_every_parameter_except_anomaly_head(tmp_path: Path) -> None:
    record = AnomalyAudioRecord(
        path=tmp_path / "normal.wav",
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00",
        group_id="normal.wav",
        label="normal",
    )
    profile = bootstrap_normal_profile((record,), 8_000, 256, 64, None, (2, 3, 2))
    model = build_model("egfn", profile, supervised_anomaly_head=True)

    trainable = configure_training_stage(model, "egfn", "head_warmup")

    assert trainable == sum(p.numel() for p in model.anomaly_head.parameters())
    assert all(p.requires_grad for p in model.anomaly_head.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("anomaly_head.")
    )


def test_one_class_checkpoint_load_leaves_new_head_untouched(tmp_path: Path) -> None:
    record = AnomalyAudioRecord(
        path=tmp_path / "normal.wav",
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00",
        group_id="normal.wav",
        label="normal",
    )
    source_profile = bootstrap_normal_profile(
        (record,), 8_000, 256, 64, None, (2, 3, 2)
    )
    source = build_model("egfn", source_profile)
    checkpoint = tmp_path / "one_class.pt"
    torch.save(
        {
            "args": {"model": "egfn", "training_mode": "one_class"},
            "model_state": source.state_dict(),
        },
        checkpoint,
    )
    target_profile = bootstrap_normal_profile(
        (record,), 8_000, 256, 64, None, (2, 3, 2)
    )
    target = build_model("egfn", target_profile, supervised_anomaly_head=True)
    original_head = target.anomaly_head.weight.detach().clone()

    loaded = load_pretrained_one_class_backbone(target, checkpoint)

    assert loaded == sum(parameter.numel() for parameter in source.parameters())
    assert torch.equal(target.encoder.input_projection.weight, source.encoder.input_projection.weight)
    assert torch.equal(target.anomaly_head.weight, original_head)


def test_hybrid_sampler_balances_labels_within_each_condition(tmp_path: Path) -> None:
    records = []
    for condition in ("id_00", "id_02"):
        for index in range(5):
            records.append(
                AnomalyAudioRecord(
                    path=tmp_path / f"{condition}_normal_{index}.wav",
                    dataset_name="test",
                    machine_type="machine",
                    machine_id=condition,
                    condition_id=condition,
                    group_id=f"{condition}_normal_{index}",
                    label="normal",
                )
            )
        for index in range(3):
            records.append(
                AnomalyAudioRecord(
                    path=tmp_path / f"{condition}_anomaly_{index}.wav",
                    dataset_name="test",
                    machine_type="machine",
                    machine_id=condition,
                    condition_id=condition,
                    group_id=f"{condition}_anomaly_{index}",
                    label="anomalous",
                )
            )
    dataset = AnomalyWindowDataset(records, 8_000, 0.5)
    sampler = HybridConditionBatchSampler(dataset, batch_size=4, shuffle=False)

    for batch in sampler:
        selected = [dataset.records[dataset._index[index][0]] for index in batch]
        assert sum(record.is_normal for record in selected) == 2
        assert len({record.condition_id for record in selected}) == 1


def test_filter_adaptation_freezes_every_parameter_except_filter_weights() -> None:
    class StagedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frontend = HierarchicalSpectralFrontend(
                sample_rate=8_000,
                macro_edges_hz=(0.0, 666.6667, 2_666.6667, 4_000.0),
                subbands_per_macro=(2, 3, 2),
                n_fft=256,
                hop_length=64,
                temporal_channels=3,
                learnable_subband_weights=True,
            )
            self.encoder = torch.nn.Linear(3, 2)

    model = StagedModel()
    configure_training_stage(model, "egfn", "filter_warmup")
    assert not model.frontend.subband_weight_logits.requires_grad
    assert model.encoder.weight.requires_grad

    trainable_count = configure_training_stage(
        model, "egfn", "filter_adaptation"
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["frontend.subband_weight_logits"]
    assert trainable_count == model.frontend.subband_weight_logits.numel()

    configure_training_stage(model, "egfn", "representation_finetune")
    assert not model.frontend.subband_weight_logits.requires_grad
    assert model.encoder.weight.requires_grad


def test_anti_collapse_objective_penalizes_constant_embeddings() -> None:
    collapsed = torch.zeros(8, 4, requires_grad=True)
    result = anti_collapse_loss(collapsed, collapsed, variance_target=0.05)
    result["representation_loss"].backward()

    assert result["variance_loss"].item() > 0
    assert result["embedding_std"].item() < 0.05
    assert collapsed.grad is not None


def test_covariance_loss_is_normalized_by_off_diagonal_terms() -> None:
    base = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    four_dimensions = base.repeat(1, 4).requires_grad_()
    eight_dimensions = base.repeat(1, 8).requires_grad_()

    four = anti_collapse_loss(
        four_dimensions,
        four_dimensions,
        invariance_weight=0.0,
        variance_weight=0.0,
        covariance_weight=1.0,
    )
    eight = anti_collapse_loss(
        eight_dimensions,
        eight_dimensions,
        invariance_weight=0.0,
        variance_weight=0.0,
        covariance_weight=1.0,
    )

    assert four["covariance_loss"].item() == pytest.approx(
        eight["covariance_loss"].item(), rel=1e-6
    )


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


def test_audio_view_is_repeatable_with_a_fixed_generator() -> None:
    waveform = torch.randn(3, 1, 1_000)
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)

    first = make_audio_view(waveform, 0.01, 0.05, generator=first_generator)
    second = make_audio_view(waveform, 0.01, 0.05, generator=second_generator)

    assert torch.equal(first, second)


def test_audio_view_shift_uses_padding_instead_of_circular_wrap() -> None:
    waveform = torch.zeros(1, 1, 100)
    waveform[..., -1] = 1.0
    generator = torch.Generator().manual_seed(0)
    view = make_audio_view(waveform, 0.0, 0.5, generator=generator)

    assert view[..., 0].item() == 0.0


def test_memory_objective_uses_mean_and_peak_of_exact_feature_map() -> None:
    feature_map = torch.tensor([[[[0.0, 1.0, 4.0]]]])
    embedding = objective_embedding(
        {"gated_profile_features": feature_map},
        "memory",
        "gated_profile",
    )

    assert embedding.squeeze(0).tolist() == pytest.approx([5.0 / 3.0, 4.0])


def test_representation_geometry_detects_redundant_dimensions() -> None:
    values = torch.linspace(-1.0, 1.0, 100)
    geometry = representation_geometry(torch.stack((values, values), dim=1))

    assert geometry["active_dimensions"] == 2
    assert geometry["effective_rank"] == pytest.approx(1.0, abs=1e-5)
    assert geometry["mean_absolute_off_diagonal_correlation"] == pytest.approx(1.0)


def test_complementary_masks_cover_every_subband_once() -> None:
    masks = complementary_subband_masks(3, 16, 0.25, torch.device("cpu"))
    coverage = torch.stack(masks).sum(dim=0)

    assert len(masks) == 4
    assert torch.equal(coverage, torch.ones_like(coverage))


def test_condition_thresholds_use_only_normal_validation_scores() -> None:
    validation = {
        "condition_ids": ["id_00", "id_00", "id_02", "id_02"],
        "labels": np.zeros(4, dtype=np.int64),
        "scores": {"score": np.asarray([1.0, 3.0, 10.0, 14.0])},
    }
    thresholds, fallback = fit_condition_thresholds(validation, "score", 0.5)
    test = {
        "condition_ids": ["id_00", "id_02", "unknown"],
        "scores": {"score": np.asarray([2.0, 12.0, fallback])},
    }

    calibrated = apply_condition_thresholds(
        test, "score", thresholds, fallback
    )

    assert thresholds == pytest.approx({"id_00": 2.0, "id_02": 12.0})
    assert calibrated.tolist() == pytest.approx([1.0, 1.0, 1.0])
    validation["labels"][0] = 1
    with pytest.raises(ValueError, match="only normal"):
        fit_condition_thresholds(validation, "score", 0.5)


def test_validation_objective_is_repeatable_for_the_same_seed() -> None:
    model = Conv1DAnomalyEncoder(embedding_channels=4)
    examples = [
        {"waveform": torch.randn(1, 1_000), "condition_id": "machine"}
        for _ in range(4)
    ]
    loader = DataLoader(examples, batch_size=4)
    arguments = (
        model,
        "conv1d",
        loader,
        torch.device("cpu"),
        0.0,
        0.05,
        25.0,
        25.0,
        0.0,
        0.0,
        0.25,
        0.01,
        0.05,
        123,
    )

    first = evaluate_representation_epoch(*arguments)
    second = evaluate_representation_epoch(*arguments)

    assert first == pytest.approx(second)


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


def test_condition_batch_sampler_never_mixes_operating_conditions() -> None:
    records = [
        AnomalyAudioRecord(
            path=Path(f"{condition}_{index}.wav"),
            dataset_name="test",
            machine_type="machine",
            machine_id=condition,
            condition_id=condition,
            group_id=f"{condition}-{index}",
            label="normal",
        )
        for condition in ("id_00", "id_02")
        for index in range(5)
    ]
    dataset = AnomalyWindowDataset(
        records,
        target_sample_rate=10,
        duration_seconds=1.0,
        crop_mode="center",
    )
    sampler = ConditionBatchSampler(
        dataset,
        batch_size=3,
        shuffle=True,
        drop_last=False,
        seed=7,
    )

    batches = list(sampler)

    assert len(batches) == 4
    assert all(
        len({dataset.condition_id_at(index) for index in batch}) == 1
        for batch in batches
    )
