from __future__ import annotations

import torch


def soft_event_pool(
    features: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Differentiable maximum-like pooling over the final temporal dimension."""
    if features.ndim < 2:
        raise ValueError("features must include a temporal dimension.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    weights = torch.softmax(features / temperature, dim=-1)
    return (features * weights).sum(dim=-1)
