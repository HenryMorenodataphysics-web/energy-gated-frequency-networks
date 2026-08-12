from __future__ import annotations

from collections.abc import Mapping, Sequence

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


class ConditionedEmbeddingProfile:
    """Normal embedding statistics indexed by operating condition."""

    def __init__(
        self,
        condition_ids: Sequence[str],
        mean: torch.Tensor,
        std: torch.Tensor,
        sample_counts: torch.Tensor,
        fallback_mean: torch.Tensor,
        fallback_std: torch.Tensor,
        minimum_std: float = 1e-3,
    ) -> None:
        self.condition_ids = tuple(condition_ids)
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("condition_ids must be non-empty and unique.")
        if mean.ndim != 2 or mean.shape != std.shape:
            raise ValueError("mean and std must have shape [condition, embedding].")
        if mean.shape[0] != len(self.condition_ids):
            raise ValueError("condition_ids must match the statistics.")
        if sample_counts.shape != (len(self.condition_ids),):
            raise ValueError("sample_counts must contain one count per condition.")
        if fallback_mean.shape != mean.shape[1:] or fallback_std.shape != mean.shape[1:]:
            raise ValueError("fallback statistics must match the embedding dimension.")
        if minimum_std <= 0 or torch.any(std <= 0) or torch.any(fallback_std <= 0):
            raise ValueError("embedding standard deviations must be positive.")
        self._condition_to_index = {
            condition_id: index for index, condition_id in enumerate(self.condition_ids)
        }
        self.mean = mean.detach().float()
        self.std = std.detach().float()
        self.sample_counts = sample_counts.detach().long()
        self.fallback_mean = fallback_mean.detach().float()
        self.fallback_std = fallback_std.detach().float()
        self.minimum_std = float(minimum_std)

    def to(self, device: torch.device | str) -> ConditionedEmbeddingProfile:
        return ConditionedEmbeddingProfile(
            self.condition_ids,
            self.mean.to(device),
            self.std.to(device),
            self.sample_counts.to(device),
            self.fallback_mean.to(device),
            self.fallback_std.to(device),
            self.minimum_std,
        )

    def standardize(
        self,
        embedding: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[1] != self.mean.shape[1]:
            raise ValueError("embedding dimension does not match the fitted profile.")
        if len(condition_ids) != embedding.shape[0]:
            raise ValueError("condition_ids must contain one value per embedding.")
        if embedding.device != self.mean.device:
            raise ValueError("move the embedding profile to the embedding device.")
        means = []
        standard_deviations = []
        known = []
        for condition_id in condition_ids:
            index = self._condition_to_index.get(condition_id)
            if index is None:
                means.append(self.fallback_mean)
                standard_deviations.append(self.fallback_std)
                known.append(False)
            else:
                means.append(self.mean[index])
                standard_deviations.append(self.std[index])
                known.append(True)
        selected_mean = torch.stack(means)
        selected_std = torch.stack(standard_deviations).clamp_min(self.minimum_std)
        return (
            (embedding - selected_mean) / selected_std,
            torch.tensor(known, dtype=torch.bool, device=embedding.device),
        )

    def scores(
        self,
        embedding: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_scores, known = self.standardize(embedding, condition_ids)
        return z_scores.square().mean(dim=1), z_scores, known

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "condition_ids": list(self.condition_ids),
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
            "sample_counts": self.sample_counts.cpu().tolist(),
            "fallback_mean": self.fallback_mean.cpu().tolist(),
            "fallback_std": self.fallback_std.cpu().tolist(),
            "minimum_std": self.minimum_std,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConditionedEmbeddingProfile:
        if payload.get("version") != 1:
            raise ValueError("unsupported embedding profile version.")
        return cls(
            condition_ids=payload["condition_ids"],
            mean=torch.tensor(payload["mean"]),
            std=torch.tensor(payload["std"]),
            sample_counts=torch.tensor(payload["sample_counts"]),
            fallback_mean=torch.tensor(payload["fallback_mean"]),
            fallback_std=torch.tensor(payload["fallback_std"]),
            minimum_std=float(payload["minimum_std"]),
        )


class ConditionedEmbeddingEstimator:
    """Fit condition-specific moments and a pooled fallback from normal embeddings."""

    def __init__(self, minimum_std: float = 1e-3) -> None:
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive.")
        self.minimum_std = float(minimum_std)
        self._moments: dict[str, dict[str, torch.Tensor | int]] = {}

    def update(self, embedding: torch.Tensor, condition_ids: Sequence[str]) -> None:
        if embedding.ndim != 2 or len(condition_ids) != embedding.shape[0]:
            raise ValueError("embeddings and condition_ids must share a batch dimension.")
        values = embedding.detach().double().cpu()
        for value, condition_id in zip(values, condition_ids, strict=True):
            if not condition_id:
                raise ValueError("condition_ids must be non-empty strings.")
            moments = self._moments.setdefault(
                condition_id,
                {
                    "count": 0,
                    "sum": torch.zeros_like(value),
                    "square_sum": torch.zeros_like(value),
                },
            )
            moments["count"] = int(moments["count"]) + 1
            moments["sum"] = moments["sum"] + value
            moments["square_sum"] = moments["square_sum"] + value.square()

    def finalize(self) -> ConditionedEmbeddingProfile:
        if not self._moments:
            raise RuntimeError("no embeddings were provided.")
        condition_ids = tuple(sorted(self._moments))
        means = []
        second_moments = []
        counts = []
        total_sum = None
        total_square_sum = None
        total_count = 0
        for condition_id in condition_ids:
            moments = self._moments[condition_id]
            count = int(moments["count"])
            value_sum = moments["sum"]
            square_sum = moments["square_sum"]
            means.append(value_sum / count)
            second_moments.append(square_sum / count)
            counts.append(count)
            total_sum = value_sum.clone() if total_sum is None else total_sum + value_sum
            total_square_sum = (
                square_sum.clone()
                if total_square_sum is None
                else total_square_sum + square_sum
            )
            total_count += count
        mean = torch.stack(means)
        second = torch.stack(second_moments)
        std = (second - mean.square()).clamp_min(self.minimum_std**2).sqrt()
        fallback_mean = total_sum / total_count
        fallback_second = total_square_sum / total_count
        fallback_std = (
            fallback_second - fallback_mean.square()
        ).clamp_min(self.minimum_std**2).sqrt()
        return ConditionedEmbeddingProfile(
            condition_ids,
            mean,
            std,
            torch.tensor(counts),
            fallback_mean,
            fallback_std,
            self.minimum_std,
        )


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
