from __future__ import annotations

import pytest

from scripts.run_mimii_hierarchical_gating_ablation import (
    parse_experiments,
    validate_payload,
)


def result_payload(**overrides: object) -> dict[str, object]:
    args = {
        "gate_mode": "none",
        "conditional_subgates": False,
        "memory_representation": "encoder",
        "objective_representation": "encoder",
        "gate_regularization_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    args.update(overrides)
    return {
        "protocol": "normal_only_aligned_representation_v7",
        "fit_labels": ["normal"],
        "args": args,
        "metrics": {
            "auc": 0.7,
            "accuracy": 0.6,
            "precision": 0.5,
            "recall": 0.4,
            "f1": 0.45,
            "tn": 2,
            "fp": 1,
            "fn": 3,
            "tp": 4,
        },
    }


def test_payload_accepts_only_aligned_causal_configuration() -> None:
    validate_payload(result_payload(), "test", "none")


def test_load_row_rejects_a_memory_representation_not_trained_by_objective(
) -> None:
    with pytest.raises(ValueError, match="causal ablation"):
        validate_payload(
            result_payload(memory_representation="gated_profile"),
            "test",
            "none",
        )


def test_conditional_subgates_are_a_separate_experiment() -> None:
    validate_payload(
        result_payload(gate_mode="hierarchical", conditional_subgates=True),
        "test",
        "hierarchical_conditional",
    )


def test_experiment_selection_allows_a_focused_causal_comparison() -> None:
    assert parse_experiments("none,macro") == ["none", "macro"]
    with pytest.raises(ValueError, match="only"):
        parse_experiments("none,unknown")
