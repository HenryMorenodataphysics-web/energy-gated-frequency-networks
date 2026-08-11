from __future__ import annotations

import pytest
import torch

from src.anomaly import ProfileAnomalyScorer


def test_score_shapes_and_condition_status() -> None:
    scorer = ProfileAnomalyScorer(top_fraction=0.5)
    result = scorer(torch.zeros(2, 4, 2, 6), torch.tensor([True, False]))

    assert result["local_score"].shape == (2, 4, 6)
    assert result["subband_score"].shape == (2, 4)
    assert result["frame_score"].shape == (2, 6)
    assert result["recording_score"].shape == (2,)
    assert result["known_condition"].tolist() == [True, False]


def test_localized_deviation_remains_localizable() -> None:
    z_scores = torch.zeros(1, 4, 2, 8)
    z_scores[0, 2, 0, 5] = 6.0
    result = ProfileAnomalyScorer(top_fraction=0.25)(z_scores)

    assert result["local_score"][0].argmax().item() == 2 * 8 + 5
    assert result["subband_score"][0].argmax().item() == 2
    assert result["frame_score"][0].argmax().item() == 5


def test_descriptor_weights_and_clipping_are_deterministic() -> None:
    z_scores = torch.tensor([[[[3.0], [100.0]]]])
    scorer = ProfileAnomalyScorer(
        descriptor_weights=(1.0, 0.0),
        z_clip=5.0,
        recording_quantile=1.0,
    )

    assert scorer(z_scores)["recording_score"].item() == pytest.approx(3.0)
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 0


@pytest.mark.parametrize(
    ("argument", "value"),
    (("top_fraction", 0.0), ("recording_quantile", 1.1), ("z_clip", 0.0)),
)
def test_invalid_configuration_is_rejected(argument: str, value: float) -> None:
    with pytest.raises(ValueError):
        ProfileAnomalyScorer(**{argument: value})
