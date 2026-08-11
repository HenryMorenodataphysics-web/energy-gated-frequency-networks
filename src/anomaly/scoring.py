from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class ProfileAnomalyScorer(nn.Module):
    """Aggregate normal-profile deviations without using learned gates."""

    def __init__(
        self,
        descriptor_weights: Sequence[float] = (1.0, 1.0),
        z_clip: float = 10.0,
        top_fraction: float = 0.25,
        recording_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(tuple(descriptor_weights), dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0 or torch.any(weights < 0):
            raise ValueError("descriptor_weights must be a non-empty non-negative sequence.")
        if float(weights.sum()) <= 0:
            raise ValueError("at least one descriptor weight must be positive.")
        if z_clip <= 0:
            raise ValueError("z_clip must be positive.")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")
        if not 0 <= recording_quantile <= 1:
            raise ValueError("recording_quantile must be in [0, 1].")

        self.register_buffer("descriptor_weights", weights / weights.sum())
        self.z_clip = float(z_clip)
        self.top_fraction = float(top_fraction)
        self.recording_quantile = float(recording_quantile)

    def forward(
        self,
        z_scores: torch.Tensor,
        known_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if z_scores.ndim != 4:
            raise ValueError("z_scores must have shape [batch, subband, descriptor, time].")
        if z_scores.shape[2] != self.descriptor_weights.numel():
            raise ValueError("z_scores descriptor count does not match descriptor_weights.")
        if not torch.isfinite(z_scores).all():
            raise ValueError("z_scores must contain only finite values.")

        clipped = z_scores.clamp(-self.z_clip, self.z_clip)
        local_score = torch.sqrt(
            (clipped.square() * self.descriptor_weights[None, None, :, None]).sum(dim=2)
        )
        top_count = max(1, math.ceil(local_score.shape[1] * self.top_fraction))
        frame_score = local_score.topk(top_count, dim=1).values.mean(dim=1)
        recording_score = torch.quantile(
            frame_score,
            self.recording_quantile,
            dim=-1,
        )
        result = {
            "local_score": local_score,
            "subband_score": local_score.mean(dim=-1),
            "frame_score": frame_score,
            "recording_score": recording_score,
        }
        if known_condition is not None:
            if known_condition.shape != (z_scores.shape[0],):
                raise ValueError("known_condition must contain one value per batch item.")
            result["known_condition"] = known_condition.to(
                device=z_scores.device,
                dtype=torch.bool,
            )
        return result
