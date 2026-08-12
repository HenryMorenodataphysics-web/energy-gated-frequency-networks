from __future__ import annotations

import torch
import torch.nn.functional as F


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def anti_collapse_loss(
    first_embedding: torch.Tensor,
    second_embedding: torch.Tensor,
    variance_target: float = 0.05,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Preserve information while learning invariance between normal audio views."""
    if first_embedding.shape != second_embedding.shape or first_embedding.ndim != 2:
        raise ValueError("embeddings must share shape [batch, embedding].")
    if first_embedding.shape[0] < 2:
        raise ValueError("anti-collapse loss requires at least two examples.")
    if variance_target <= 0 or epsilon <= 0:
        raise ValueError("variance_target and epsilon must be positive.")
    if min(invariance_weight, variance_weight, covariance_weight) < 0:
        raise ValueError("loss weights must be non-negative.")

    invariance_loss = F.mse_loss(first_embedding, second_embedding)
    first_centered = first_embedding - first_embedding.mean(dim=0)
    second_centered = second_embedding - second_embedding.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=True) + epsilon)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=True) + epsilon)
    variance_loss = 0.5 * (
        F.relu(variance_target - first_std).mean()
        + F.relu(variance_target - second_std).mean()
    )

    batch_denominator = first_embedding.shape[0] - 1
    first_covariance = first_centered.T @ first_centered / batch_denominator
    second_covariance = second_centered.T @ second_centered / batch_denominator
    embedding_dimensions = first_embedding.shape[1]
    covariance_loss = (
        off_diagonal(first_covariance).square().sum()
        + off_diagonal(second_covariance).square().sum()
    ) / (2.0 * embedding_dimensions)
    total_loss = (
        invariance_weight * invariance_loss
        + variance_weight * variance_loss
        + covariance_weight * covariance_loss
    )
    return {
        "representation_loss": total_loss,
        "invariance_loss": invariance_loss,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "embedding_std": 0.5 * (first_std.mean() + second_std.mean()),
    }


def standardized_embedding_scores(
    embedding: torch.Tensor,
    normal_mean: torch.Tensor,
    normal_std: torch.Tensor,
    minimum_std: float = 1e-3,
) -> torch.Tensor:
    if embedding.ndim != 2 or normal_mean.shape != (embedding.shape[1],):
        raise ValueError("embedding and normal_mean shapes do not match.")
    if normal_std.shape != normal_mean.shape:
        raise ValueError("normal_std and normal_mean shapes do not match.")
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive.")
    if embedding.device != normal_mean.device or embedding.device != normal_std.device:
        raise ValueError("embedding statistics must be on the embedding device.")
    z_scores = (embedding - normal_mean) / normal_std.clamp_min(minimum_std)
    return z_scores.square().mean(dim=1)


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
