from __future__ import annotations

import numpy as np
import pytest

from src.utils.macro_band_selection import (
    condition_balanced_power_profile,
    select_macro_band_candidates,
    validate_training_profile_source,
)


def test_condition_balancing_prevents_large_group_from_dominating() -> None:
    large_group = np.tile([9.0, 1.0], (100, 1))
    small_group = np.asarray([[1.0, 9.0]])
    powers = np.vstack([large_group, small_group])
    conditions = ["large"] * len(large_group) + ["small"]

    profile = condition_balanced_power_profile(powers, conditions)

    assert np.allclose(profile, [0.5, 0.5])


def test_equal_energy_candidate_differs_from_uniform_when_power_is_skewed() -> None:
    fine_edges = np.arange(7, dtype=np.float64) * 100.0
    powers = np.asarray(
        [
            [0.30, 0.03, 0.03, 0.03, 0.30, 0.31],
            [0.30, 0.03, 0.03, 0.03, 0.30, 0.31],
        ]
    )

    uniform, balanced = select_macro_band_candidates(
        fine_edges,
        powers,
        ["machine_a", "machine_b"],
        num_macro_bands=3,
    )

    assert uniform.edges_hz == (0.0, 200.0, 400.0, 600.0)
    assert balanced.edges_hz == (0.0, 200.0, 500.0, 600.0)
    assert all(left < right for left, right in zip(balanced.edges_hz, balanced.edges_hz[1:]))
    assert sum(balanced.energy_fraction) == pytest.approx(1.0)


def test_minimum_width_is_respected() -> None:
    fine_edges = np.arange(10, dtype=np.float64)
    powers = np.asarray([[0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03]])

    _, balanced = select_macro_band_candidates(
        fine_edges,
        powers,
        ["machine"],
        num_macro_bands=3,
        min_fine_bands=2,
    )

    widths = np.diff(balanced.edges_hz)
    assert np.all(widths >= 2.0)


@pytest.mark.parametrize(
    ("split_name", "labels"),
    [("test", ["normal"]), ("train", ["normal", "anomalous"])],
)
def test_profile_source_rejects_leakage(split_name: str, labels: list[str]) -> None:
    with pytest.raises(ValueError):
        validate_training_profile_source(split_name, labels)
