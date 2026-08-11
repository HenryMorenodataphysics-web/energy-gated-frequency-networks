from __future__ import annotations

import numpy as np

from src.utils import metrics_at_threshold, select_f1_threshold


def test_metrics_at_threshold_counts_confusion_matrix() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])

    metrics = metrics_at_threshold(targets, probabilities, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["accuracy"] == 0.5


def test_select_f1_threshold_uses_validation_probabilities() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.9])

    threshold, rows = select_f1_threshold(
        targets,
        probabilities,
        thresholds=np.array([0.25, 0.5]),
    )

    assert threshold == 0.25
    assert len(rows) == 2
