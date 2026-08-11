from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.anomaly import ConditionedNormalProfile, NormalProfileEstimator


def constant_energy(log_value: float, frames: int, subbands: int = 2) -> torch.Tensor:
    value = math.exp(log_value)
    return torch.full((1, subbands, frames), value)


def fit_example_profile() -> ConditionedNormalProfile:
    estimator = NormalProfileEstimator(num_subbands=2, minimum_std=0.1)
    estimator.update(constant_energy(0.0, frames=5), ["machine_a"], ["normal"])
    estimator.update(constant_energy(2.0, frames=20), ["machine_a"], ["normal"])
    estimator.update(constant_energy(4.0, frames=7), ["machine_b"], ["normal"])
    return estimator.finalize()


def test_profile_weights_each_record_equally_despite_duration() -> None:
    profile = fit_example_profile()
    machine_a = profile.condition_ids.index("machine_a")

    assert profile.mean[machine_a, :, 0].tolist() == pytest.approx([1.0, 1.0])
    assert profile.record_counts[machine_a].item() == 2


def test_conditions_keep_independent_statistics() -> None:
    profile = fit_example_profile()
    machine_a = profile.condition_ids.index("machine_a")
    machine_b = profile.condition_ids.index("machine_b")

    assert profile.mean[machine_a, 0, 0].item() == pytest.approx(1.0)
    assert profile.mean[machine_b, 0, 0].item() == pytest.approx(4.0)


def test_known_condition_is_standardized_with_its_profile() -> None:
    profile = fit_example_profile()
    result = profile.standardize(constant_energy(1.0, frames=4), ["machine_a"])

    assert result["known_condition"].tolist() == [True]
    assert torch.allclose(result["z_scores"][:, :, 0], torch.zeros(1, 2, 4), atol=1e-5)


def test_unknown_condition_uses_finite_fallback() -> None:
    profile = fit_example_profile()
    result = profile.standardize(constant_energy(2.5, frames=4), ["new_machine"])

    assert result["known_condition"].tolist() == [False]
    assert torch.isfinite(result["z_scores"]).all()


def test_profile_rejects_leakage_and_non_normal_labels() -> None:
    with pytest.raises(ValueError, match="training split"):
        NormalProfileEstimator(num_subbands=2, split_name="test")

    estimator = NormalProfileEstimator(num_subbands=2)
    with pytest.raises(ValueError, match="non-normal"):
        estimator.update(constant_energy(1.0, frames=4), ["machine"], ["anomalous"])


def test_minimum_condition_count_is_enforced() -> None:
    estimator = NormalProfileEstimator(
        num_subbands=2,
        minimum_records_per_condition=2,
    )
    estimator.update(constant_energy(1.0, frames=4), ["machine"], ["normal"])

    with pytest.raises(ValueError, match="requires 2"):
        estimator.finalize()


def test_json_round_trip() -> None:
    profile = fit_example_profile()
    path = Path(".tmp") / "normal_profile_test.json"
    try:
        profile.save_json(path)
        restored = ConditionedNormalProfile.load_json(path)
    finally:
        path.unlink(missing_ok=True)

    assert restored.condition_ids == profile.condition_ids
    assert torch.equal(restored.record_counts, profile.record_counts)
    assert torch.allclose(restored.mean, profile.mean)
    assert torch.allclose(restored.std, profile.std)
    assert torch.allclose(restored.fallback_mean, profile.fallback_mean)
    assert torch.allclose(restored.fallback_std, profile.fallback_std)
