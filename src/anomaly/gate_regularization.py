from __future__ import annotations

import torch
import torch.nn as nn


class GateRegularizer(nn.Module):
    """Prevent gate collapse, then gradually encourage discrete routing."""

    def __init__(
        self,
        macro_target_utilization: float = 2.0 / 3.0,
        subband_target_utilization: float = 0.35,
        utilization_weight: float = 1.0,
        hardening_weight: float = 0.1,
        hardening_warmup: float = 0.25,
        collapse_threshold: float = 0.02,
    ) -> None:
        super().__init__()
        for name, value in (
            ("macro_target_utilization", macro_target_utilization),
            ("subband_target_utilization", subband_target_utilization),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1).")
        if utilization_weight < 0 or hardening_weight < 0:
            raise ValueError("regularization weights must be non-negative.")
        if not 0 <= hardening_warmup < 1:
            raise ValueError("hardening_warmup must be in [0, 1).")
        if not 0 < collapse_threshold < 0.5:
            raise ValueError("collapse_threshold must be in (0, 0.5).")

        self.macro_target_utilization = float(macro_target_utilization)
        self.subband_target_utilization = float(subband_target_utilization)
        self.utilization_weight = float(utilization_weight)
        self.hardening_weight = float(hardening_weight)
        self.hardening_warmup = float(hardening_warmup)
        self.collapse_threshold = float(collapse_threshold)

    @staticmethod
    def _validate_gates(name: str, gates: torch.Tensor) -> None:
        if gates.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, gate, time].")
        if not torch.isfinite(gates).all() or torch.any((gates < 0) | (gates > 1)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")

    def forward(
        self,
        macro_gates: torch.Tensor,
        subband_gates: torch.Tensor,
        progress: float = 0.0,
        regularize_macro: bool = True,
        regularize_subband: bool = True,
    ) -> dict[str, torch.Tensor]:
        self._validate_gates("macro_gates", macro_gates)
        self._validate_gates("subband_gates", subband_gates)
        if macro_gates.shape[0] != subband_gates.shape[0] or macro_gates.shape[2] != subband_gates.shape[2]:
            raise ValueError("macro and subband gates must align in batch and time.")
        if not 0 <= progress <= 1:
            raise ValueError("progress must be in [0, 1].")

        macro_utilization = macro_gates.mean(dim=(0, 2))
        subband_utilization = subband_gates.mean(dim=(0, 2))
        utilization_terms = []
        binary_terms = []
        if regularize_macro:
            utilization_terms.append(
                (macro_utilization - self.macro_target_utilization).square().mean()
            )
            binary_terms.append((macro_gates * (1.0 - macro_gates)).mean())
        if regularize_subband:
            utilization_terms.append(
                (subband_utilization - self.subband_target_utilization).square().mean()
            )
            binary_terms.append((subband_gates * (1.0 - subband_gates)).mean())
        zero = macro_gates.new_zeros(())
        utilization_loss = (
            torch.stack(utilization_terms).mean() if utilization_terms else zero
        )
        binary_loss = torch.stack(binary_terms).mean() if binary_terms else zero
        hardening_schedule = max(
            0.0,
            (float(progress) - self.hardening_warmup) / (1.0 - self.hardening_warmup),
        )
        total_loss = (
            self.utilization_weight * utilization_loss
            + self.hardening_weight * hardening_schedule * binary_loss
        )

        threshold = self.collapse_threshold
        return {
            "gate_regularization_loss": total_loss,
            "gate_utilization_loss": utilization_loss,
            "gate_binary_loss": binary_loss,
            "gate_hardening_schedule": total_loss.new_tensor(hardening_schedule),
            "macro_mean_utilization": macro_utilization.mean(),
            "subband_mean_utilization": subband_utilization.mean(),
            "macro_dead_fraction": (macro_utilization <= threshold).float().mean(),
            "macro_always_open_fraction": (macro_utilization >= 1.0 - threshold).float().mean(),
            "subband_dead_fraction": (subband_utilization <= threshold).float().mean(),
            "subband_always_open_fraction": (
                subband_utilization >= 1.0 - threshold
            ).float().mean(),
        }
