from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MacroBandCandidate:
    policy: str
    edges_hz: tuple[float, ...]
    edges_fraction: tuple[float, ...]
    energy_fraction: tuple[float, ...]


def validate_training_profile_source(
    split_name: str,
    labels: Sequence[str],
) -> None:
    if split_name != "train":
        raise ValueError("macro bands must be selected from the training split.")
    unexpected = sorted(set(labels) - {"normal"})
    if unexpected:
        raise ValueError(
            "macro bands must use only normal training records; "
            f"found labels: {unexpected}."
        )


def condition_balanced_power_profile(
    powers: np.ndarray,
    condition_ids: Sequence[str],
) -> np.ndarray:
    """Average per-record power fractions, weighting every condition equally."""
    powers = np.asarray(powers, dtype=np.float64)
    if powers.ndim != 2 or powers.shape[0] == 0 or powers.shape[1] == 0:
        raise ValueError("powers must have shape [records, fine_bands].")
    if len(condition_ids) != powers.shape[0]:
        raise ValueError("condition_ids must contain one value per record.")
    if not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("powers must be finite and non-negative.")

    totals = powers.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every record must contain positive spectral power.")
    fractions = powers / totals

    conditions = np.asarray(condition_ids, dtype=object)
    condition_profiles = []
    for condition in sorted(set(condition_ids)):
        condition_profiles.append(fractions[conditions == condition].mean(axis=0))
    profile = np.mean(np.stack(condition_profiles, axis=0), axis=0)
    return profile / profile.sum()


def _validate_edges(fine_edges_hz: Sequence[float], fine_band_count: int) -> np.ndarray:
    edges = np.asarray(fine_edges_hz, dtype=np.float64)
    if edges.ndim != 1 or edges.size != fine_band_count + 1:
        raise ValueError("fine_edges_hz must contain one more edge than fine bands.")
    if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("fine_edges_hz must be finite and strictly increasing.")
    return edges


def _macro_energy(profile: np.ndarray, cut_indices: Sequence[int]) -> tuple[float, ...]:
    boundaries = (0, *cut_indices, profile.size)
    return tuple(
        float(profile[start:end].sum())
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def _candidate(
    policy: str,
    fine_edges_hz: np.ndarray,
    profile: np.ndarray,
    cut_indices: Sequence[int],
) -> MacroBandCandidate:
    selected_edges = (fine_edges_hz[0], *(fine_edges_hz[index] for index in cut_indices), fine_edges_hz[-1])
    span = fine_edges_hz[-1] - fine_edges_hz[0]
    fractions = tuple(float((edge - fine_edges_hz[0]) / span) for edge in selected_edges)
    return MacroBandCandidate(
        policy=policy,
        edges_hz=tuple(float(edge) for edge in selected_edges),
        edges_fraction=fractions,
        energy_fraction=_macro_energy(profile, cut_indices),
    )


def select_macro_band_candidates(
    fine_edges_hz: Sequence[float],
    powers: np.ndarray,
    condition_ids: Sequence[str],
    num_macro_bands: int = 3,
    min_fine_bands: int = 1,
) -> tuple[MacroBandCandidate, MacroBandCandidate]:
    if num_macro_bands < 2:
        raise ValueError("num_macro_bands must be at least 2.")
    if min_fine_bands < 1:
        raise ValueError("min_fine_bands must be positive.")

    profile = condition_balanced_power_profile(powers, condition_ids)
    fine_band_count = profile.size
    if fine_band_count < num_macro_bands * min_fine_bands:
        raise ValueError("not enough fine bands for the requested minimum widths.")
    edges = _validate_edges(fine_edges_hz, fine_band_count)

    uniform_cuts = []
    for macro_index in range(1, num_macro_bands):
        target = macro_index * fine_band_count / num_macro_bands
        cut = int(round(target))
        minimum = uniform_cuts[-1] + min_fine_bands if uniform_cuts else min_fine_bands
        maximum = fine_band_count - (num_macro_bands - macro_index) * min_fine_bands
        uniform_cuts.append(min(max(cut, minimum), maximum))

    cumulative = np.cumsum(profile)
    energy_cuts = []
    for macro_index in range(1, num_macro_bands):
        minimum = energy_cuts[-1] + min_fine_bands if energy_cuts else min_fine_bands
        maximum = fine_band_count - (num_macro_bands - macro_index) * min_fine_bands
        candidates = np.arange(minimum, maximum + 1)
        target = macro_index / num_macro_bands
        distances = np.abs(cumulative[candidates - 1] - target)
        energy_cuts.append(int(candidates[int(np.argmin(distances))]))

    return (
        _candidate("uniform", edges, profile, uniform_cuts),
        _candidate("condition_balanced_energy", edges, profile, energy_cuts),
    )
