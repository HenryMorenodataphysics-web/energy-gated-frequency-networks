from __future__ import annotations

from scripts.run_mimii_multiseed import aggregate


def test_aggregate_reports_sample_mean_and_std() -> None:
    rows = []
    for seed, accuracy in ((1, 0.8), (2, 0.9), (3, 1.0)):
        for operating_point in ("default", "calibrated"):
            rows.append(
                {
                    "model": "conv1d",
                    "seed": seed,
                    "operating_point": operating_point,
                    "threshold": 0.5,
                    "accuracy": accuracy,
                    "precision": accuracy,
                    "recall": accuracy,
                    "f1": accuracy,
                    "auc": accuracy,
                }
            )

    summary = aggregate(rows)

    assert len(summary) == 2
    assert summary[0]["n_seeds"] == 3
    assert abs(summary[0]["accuracy_mean"] - 0.9) < 1e-12
    assert abs(summary[0]["accuracy_std"] - 0.1) < 1e-12
