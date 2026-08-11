from __future__ import annotations

import pytest
import torch

from src.anomaly import GateRegularizer


def test_target_utilization_avoids_penalizing_balanced_routing() -> None:
    regularizer = GateRegularizer(
        macro_target_utilization=0.5,
        subband_target_utilization=0.5,
    )
    macro = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    subband = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    result = regularizer(macro, subband, progress=0.0)

    assert result["gate_utilization_loss"].item() == pytest.approx(0.0)
    assert result["macro_dead_fraction"].item() == 0.0
    assert result["subband_always_open_fraction"].item() == 0.0


def test_collapsed_gates_are_detected_and_penalized() -> None:
    regularizer = GateRegularizer()
    result = regularizer(torch.ones(2, 3, 4), torch.zeros(2, 6, 4))

    assert result["gate_utilization_loss"].item() > 0
    assert result["macro_always_open_fraction"].item() == 1.0
    assert result["subband_dead_fraction"].item() == 1.0


def test_hardening_starts_after_warmup_and_remains_differentiable() -> None:
    regularizer = GateRegularizer(
        macro_target_utilization=0.5,
        subband_target_utilization=0.5,
        hardening_warmup=0.25,
    )
    logits = torch.zeros(1, 5, 8, requires_grad=True)
    gates = torch.sigmoid(logits)
    warmup = regularizer(gates[:, :2], gates[:, 2:], progress=0.25)
    hardened = regularizer(gates[:, :2], gates[:, 2:], progress=1.0)
    hardened["gate_regularization_loss"].backward()

    assert warmup["gate_hardening_schedule"].item() == 0.0
    assert hardened["gate_hardening_schedule"].item() == 1.0
    assert hardened["gate_regularization_loss"].item() > warmup["gate_regularization_loss"].item()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_gate_probabilities_are_rejected() -> None:
    regularizer = GateRegularizer()
    with pytest.raises(ValueError, match="probabilities"):
        regularizer(torch.tensor([[[1.2]]]), torch.tensor([[[0.5]]]))
