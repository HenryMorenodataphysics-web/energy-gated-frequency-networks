from __future__ import annotations

import torch


def stabilize_center(center: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Keep fixed Deep SVDD center components away from trivial zero values."""
    if center.ndim != 1:
        raise ValueError("center must have shape [embedding].")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    stabilized = center.detach().clone()
    small = stabilized.abs() < epsilon
    stabilized[small & (stabilized < 0)] = -epsilon
    stabilized[small & (stabilized >= 0)] = epsilon
    return stabilized


def deep_svdd_scores(embedding: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2 or center.shape != (embedding.shape[1],):
        raise ValueError("embedding and center shapes do not match.")
    if embedding.device != center.device:
        raise ValueError("embedding and center must be on the same device.")
    return (embedding - center).square().mean(dim=1)


def deep_svdd_loss(embedding: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return deep_svdd_scores(embedding, center).mean()
