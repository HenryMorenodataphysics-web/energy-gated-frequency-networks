from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def local_feature_descriptors(
    feature_map: torch.Tensor,
    temporal_pool: int,
) -> tuple[torch.Tensor, int, int]:
    """Return local descriptors as [batch, subband*time, channel]."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [batch, channel, subband, time].")
    if temporal_pool <= 0:
        raise ValueError("temporal_pool must be positive.")
    pool_size = min(int(temporal_pool), feature_map.shape[-1])
    pooled = F.avg_pool2d(
        feature_map,
        kernel_size=(1, pool_size),
        stride=(1, pool_size),
        ceil_mode=True,
    )
    batch, channels, subbands, frames = pooled.shape
    descriptors = pooled.permute(0, 2, 3, 1).reshape(
        batch, subbands * frames, channels
    )
    return descriptors, subbands, frames


class ConditionedFeatureMemory:
    """Bounded normal-feature memory with condition-specific kNN scoring."""

    def __init__(
        self,
        condition_ids: Sequence[str],
        memories: Sequence[torch.Tensor],
        mean: torch.Tensor,
        std: torch.Tensor,
        fallback_memory: torch.Tensor,
        fallback_mean: torch.Tensor,
        fallback_std: torch.Tensor,
        temporal_pool: int = 4,
        top_fraction: float = 0.05,
        minimum_std: float = 1e-3,
        query_chunk_size: int = 2_048,
    ) -> None:
        self.condition_ids = tuple(condition_ids)
        self.memories = tuple(memory.detach().float() for memory in memories)
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("condition_ids must be non-empty and unique.")
        if len(self.memories) != len(self.condition_ids):
            raise ValueError("memories must contain one tensor per condition.")
        if mean.ndim != 2 or mean.shape != std.shape:
            raise ValueError("mean and std must have shape [condition, channel].")
        if mean.shape[0] != len(self.condition_ids):
            raise ValueError("condition_ids must match the feature statistics.")
        channels = mean.shape[1]
        if any(memory.ndim != 2 or memory.shape[1] != channels for memory in self.memories):
            raise ValueError("each memory must have shape [example, channel].")
        if any(memory.shape[0] == 0 for memory in self.memories):
            raise ValueError("condition memories must be non-empty.")
        if fallback_memory.ndim != 2 or fallback_memory.shape[1] != channels:
            raise ValueError("fallback_memory has an unexpected shape.")
        if fallback_memory.shape[0] == 0:
            raise ValueError("fallback_memory must be non-empty.")
        if fallback_mean.shape != (channels,) or fallback_std.shape != (channels,):
            raise ValueError("fallback statistics must match the channel count.")
        if temporal_pool <= 0 or query_chunk_size <= 0 or minimum_std <= 0:
            raise ValueError("memory configuration values must be positive.")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")
        if torch.any(std <= 0) or torch.any(fallback_std <= 0):
            raise ValueError("feature standard deviations must be positive.")
        self._condition_to_index = {
            condition_id: index for index, condition_id in enumerate(self.condition_ids)
        }
        self.mean = mean.detach().float()
        self.std = std.detach().float()
        self.fallback_memory = fallback_memory.detach().float()
        self.fallback_mean = fallback_mean.detach().float()
        self.fallback_std = fallback_std.detach().float()
        self.temporal_pool = int(temporal_pool)
        self.top_fraction = float(top_fraction)
        self.minimum_std = float(minimum_std)
        self.query_chunk_size = int(query_chunk_size)

    def to(self, device: torch.device | str) -> ConditionedFeatureMemory:
        return ConditionedFeatureMemory(
            self.condition_ids,
            [memory.to(device) for memory in self.memories],
            self.mean.to(device),
            self.std.to(device),
            self.fallback_memory.to(device),
            self.fallback_mean.to(device),
            self.fallback_std.to(device),
            self.temporal_pool,
            self.top_fraction,
            self.minimum_std,
            self.query_chunk_size,
        )

    @staticmethod
    def _nearest_squared_distance(
        query: torch.Tensor,
        memory: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        results = []
        memory_squared = memory.square().sum(dim=1).unsqueeze(0)
        for start in range(0, query.shape[0], chunk_size):
            chunk = query[start : start + chunk_size]
            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                + memory_squared
                - 2.0 * chunk @ memory.T
            ).clamp_min(0.0)
            results.append(distances.amin(dim=1) / query.shape[1])
        return torch.cat(results)

    def score(
        self,
        feature_map: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(condition_ids) != feature_map.shape[0]:
            raise ValueError("condition_ids must contain one value per feature map.")
        if feature_map.device != self.mean.device:
            raise ValueError("move the feature memory to the feature_map device.")
        descriptors, subbands, frames = local_feature_descriptors(
            feature_map, self.temporal_pool
        )
        local_scores = []
        known = []
        for sample, condition_id in zip(descriptors, condition_ids, strict=True):
            index = self._condition_to_index.get(condition_id)
            if index is None:
                memory = self.fallback_memory
                mean = self.fallback_mean
                std = self.fallback_std
                known.append(False)
            else:
                memory = self.memories[index]
                mean = self.mean[index]
                std = self.std[index]
                known.append(True)
            scale = std.clamp_min(self.minimum_std)
            standardized_query = (sample - mean) / scale
            standardized_memory = (memory - mean) / scale
            local_scores.append(
                self._nearest_squared_distance(
                    standardized_query,
                    standardized_memory,
                    self.query_chunk_size,
                ).reshape(subbands, frames)
            )
        local_score = torch.stack(local_scores)
        time_top_count = max(1, math.ceil(frames * self.top_fraction))
        subband_score = local_score.topk(time_top_count, dim=-1).values.mean(dim=-1)
        recording_top_count = max(
            1, math.ceil(subbands * frames * self.top_fraction)
        )
        recording_score = local_score.flatten(1).topk(
            recording_top_count, dim=1
        ).values.mean(dim=1)
        return {
            "local_memory_score": local_score,
            "subband_memory_score": subband_score,
            "recording_memory_score": recording_score,
            "known_memory_condition": torch.tensor(
                known, dtype=torch.bool, device=feature_map.device
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "condition_ids": list(self.condition_ids),
            "memories": [memory.cpu().tolist() for memory in self.memories],
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
            "fallback_memory": self.fallback_memory.cpu().tolist(),
            "fallback_mean": self.fallback_mean.cpu().tolist(),
            "fallback_std": self.fallback_std.cpu().tolist(),
            "temporal_pool": self.temporal_pool,
            "top_fraction": self.top_fraction,
            "minimum_std": self.minimum_std,
            "query_chunk_size": self.query_chunk_size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConditionedFeatureMemory:
        if payload.get("version") != 1:
            raise ValueError("unsupported feature memory version.")
        return cls(
            condition_ids=payload["condition_ids"],
            memories=[torch.tensor(value) for value in payload["memories"]],
            mean=torch.tensor(payload["mean"]),
            std=torch.tensor(payload["std"]),
            fallback_memory=torch.tensor(payload["fallback_memory"]),
            fallback_mean=torch.tensor(payload["fallback_mean"]),
            fallback_std=torch.tensor(payload["fallback_std"]),
            temporal_pool=int(payload["temporal_pool"]),
            top_fraction=float(payload["top_fraction"]),
            minimum_std=float(payload["minimum_std"]),
            query_chunk_size=int(payload["query_chunk_size"]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "condition_ids": list(self.condition_ids),
            "memory_sizes": {
                condition_id: int(memory.shape[0])
                for condition_id, memory in zip(
                    self.condition_ids, self.memories, strict=True
                )
            },
            "fallback_memory_size": int(self.fallback_memory.shape[0]),
            "feature_channels": int(self.mean.shape[1]),
            "temporal_pool": self.temporal_pool,
            "top_fraction": self.top_fraction,
        }


class ConditionedFeatureMemoryEstimator:
    """Fit bounded condition memories using deterministic random priorities."""

    def __init__(
        self,
        max_vectors_per_condition: int = 512,
        temporal_pool: int = 4,
        top_fraction: float = 0.05,
        minimum_std: float = 1e-3,
        query_chunk_size: int = 2_048,
        seed: int = 42,
    ) -> None:
        if max_vectors_per_condition <= 0:
            raise ValueError("max_vectors_per_condition must be positive.")
        self.max_vectors_per_condition = int(max_vectors_per_condition)
        self.temporal_pool = int(temporal_pool)
        self.top_fraction = float(top_fraction)
        self.minimum_std = float(minimum_std)
        self.query_chunk_size = int(query_chunk_size)
        self._generator = torch.Generator().manual_seed(seed)
        self._moments: dict[str, dict[str, torch.Tensor | int]] = {}
        self._memories: dict[str, torch.Tensor] = {}
        self._priorities: dict[str, torch.Tensor] = {}

    def _update_key(self, key: str, descriptors: torch.Tensor) -> None:
        values = descriptors.detach().float().cpu()
        double_values = values.double()
        moments = self._moments.setdefault(
            key,
            {
                "count": 0,
                "sum": torch.zeros(values.shape[1], dtype=torch.float64),
                "square_sum": torch.zeros(values.shape[1], dtype=torch.float64),
            },
        )
        moments["count"] = int(moments["count"]) + values.shape[0]
        moments["sum"] = moments["sum"] + double_values.sum(dim=0)
        moments["square_sum"] = (
            moments["square_sum"] + double_values.square().sum(dim=0)
        )
        priorities = torch.rand(values.shape[0], generator=self._generator)
        if key in self._memories:
            values = torch.cat((self._memories[key], values))
            priorities = torch.cat((self._priorities[key], priorities))
        keep = min(self.max_vectors_per_condition, values.shape[0])
        indices = priorities.topk(keep).indices
        self._memories[key] = values[indices]
        self._priorities[key] = priorities[indices]

    def update(
        self,
        feature_map: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> None:
        descriptors, _, _ = local_feature_descriptors(
            feature_map.detach(), self.temporal_pool
        )
        if len(condition_ids) != descriptors.shape[0]:
            raise ValueError("condition_ids must contain one value per feature map.")
        for sample, condition_id in zip(descriptors, condition_ids, strict=True):
            if not condition_id:
                raise ValueError("condition_ids must be non-empty strings.")
            self._update_key(condition_id, sample)
            self._update_key("__fallback__", sample)

    def _statistics(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self._moments[key]
        count = int(moments["count"])
        mean = moments["sum"] / count
        second = moments["square_sum"] / count
        std = (second - mean.square()).clamp_min(self.minimum_std**2).sqrt()
        return mean.float(), std.float()

    def finalize(self) -> ConditionedFeatureMemory:
        condition_ids = tuple(sorted(set(self._moments) - {"__fallback__"}))
        if not condition_ids or "__fallback__" not in self._moments:
            raise RuntimeError("no feature maps were provided.")
        means = []
        standard_deviations = []
        for condition_id in condition_ids:
            mean, std = self._statistics(condition_id)
            means.append(mean)
            standard_deviations.append(std)
        fallback_mean, fallback_std = self._statistics("__fallback__")
        return ConditionedFeatureMemory(
            condition_ids=condition_ids,
            memories=[self._memories[item] for item in condition_ids],
            mean=torch.stack(means),
            std=torch.stack(standard_deviations),
            fallback_memory=self._memories["__fallback__"],
            fallback_mean=fallback_mean,
            fallback_std=fallback_std,
            temporal_pool=self.temporal_pool,
            top_fraction=self.top_fraction,
            minimum_std=self.minimum_std,
            query_chunk_size=self.query_chunk_size,
        )
