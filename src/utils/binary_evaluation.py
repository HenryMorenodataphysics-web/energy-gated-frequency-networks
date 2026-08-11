from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def metrics_at_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    auc = float("nan")
    if np.unique(targets).size == 2:
        auc = float(roc_auc_score(targets, probabilities))
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == targets)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_sweep(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    return [metrics_at_threshold(targets, probabilities, float(value)) for value in thresholds]


def select_f1_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, list[dict[str, float | int]]]:
    rows = threshold_sweep(targets, probabilities, thresholds)
    # Prefer precision when F1 ties, then the threshold closest to 0.5.
    best = max(rows, key=lambda row: (row["f1"], row["precision"], -abs(row["threshold"] - 0.5)))
    return float(best["threshold"]), rows
