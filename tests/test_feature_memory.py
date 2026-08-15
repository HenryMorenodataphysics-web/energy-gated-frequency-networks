from __future__ import annotations

import pytest
import torch

from src.anomaly import ConditionedFeatureMemory, ConditionedFeatureMemoryEstimator


def test_conditioned_memory_scores_local_deviation_and_preserves_location() -> None:
    band_memory = torch.stack(
        (torch.zeros(4, 1), torch.full((4, 1), 10.0))
    )
    memory = ConditionedFeatureMemory(
        condition_ids=("id_00",),
        memories=(band_memory,),
        mean=torch.tensor([[[0.0], [10.0]]]),
        std=torch.ones(1, 2, 1),
        fallback_memory=band_memory,
        fallback_mean=torch.tensor([[0.0], [10.0]]),
        fallback_std=torch.ones(2, 1),
        temporal_pool=1,
        top_fraction=1.0,
        query_chunk_size=2,
    )
    feature_map = torch.tensor(
        [
            [[[0.0], [10.0]]],
            [[[10.0], [0.0]]],
        ]
    )
    result = memory.score(feature_map, ["id_00", "id_00"])

    assert result["local_memory_score"].shape == (2, 2, 1)
    assert result["subband_memory_score"].shape == (2, 2)
    assert result["recording_memory_score"][0].item() == 0.0
    assert result["recording_memory_score"][1].item() > 0.0
    assert torch.all(result["local_memory_score"][1] > 0)
    assert result["known_memory_condition"].tolist() == [True, True]


def test_memory_estimator_is_bounded_conditioned_and_serializable() -> None:
    estimator = ConditionedFeatureMemoryEstimator(
        max_vectors_per_condition=5,
        temporal_pool=1,
        top_fraction=0.5,
        seed=7,
    )
    feature_map = torch.cat(
        (torch.zeros(2, 2, 1, 6), torch.full((2, 2, 1, 6), 10.0)),
        dim=0,
    )
    estimator.update(feature_map, ["id_00", "id_00", "id_02", "id_02"])
    memory = estimator.finalize()
    restored = ConditionedFeatureMemory.from_dict(memory.to_dict())
    query = torch.stack(
        (torch.zeros(2, 1, 2), torch.full((2, 1, 2), 10.0), torch.zeros(2, 1, 2))
    )
    result = restored.score(query, ["id_00", "id_02", "unknown"])

    assert memory.summary()["memory_sizes"] == {"id_00": [5], "id_02": [5]}
    assert memory.summary()["fallback_memory_sizes"] == [5]
    assert result["recording_memory_score"][:2].tolist() == [0.0, 0.0]
    assert result["known_memory_condition"].tolist() == [True, True, False]


def test_anomaly_bank_reuses_normal_standardization_for_comparable_distances() -> None:
    normal = ConditionedFeatureMemory(
        condition_ids=("machine",),
        memories=(torch.zeros(1, 2, 1),),
        mean=torch.zeros(1, 1, 1),
        std=torch.ones(1, 1, 1),
        fallback_memory=torch.zeros(1, 2, 1),
        fallback_mean=torch.zeros(1, 1),
        fallback_std=torch.ones(1, 1),
        temporal_pool=1,
        top_fraction=1.0,
    )
    anomaly = ConditionedFeatureMemory(
        condition_ids=("machine",),
        memories=(torch.full((1, 2, 1), 10.0),),
        mean=torch.full((1, 1, 1), 10.0),
        std=torch.ones(1, 1, 1),
        fallback_memory=torch.full((1, 2, 1), 10.0),
        fallback_mean=torch.full((1, 1), 10.0),
        fallback_std=torch.ones(1, 1),
        temporal_pool=1,
        top_fraction=1.0,
    ).with_statistics_from(normal)
    query = torch.tensor([[[[0.0]]], [[[10.0]]]])

    normal_distance = normal.score(query, ["machine", "machine"])[
        "recording_memory_score"
    ]
    anomaly_distance = anomaly.score(query, ["machine", "machine"])[
        "recording_memory_score"
    ]
    reference_score = normal_distance / (
        normal_distance + anomaly_distance + 1e-8
    )

    assert anomaly.mean.item() == 0.0
    assert normal_distance.tolist() == [0.0, 100.0]
    assert anomaly_distance.tolist() == [100.0, 0.0]
    assert reference_score.tolist() == pytest.approx([0.0, 1.0])
