from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DESCRIPTOR_NAMES = ("log_energy", "absolute_log_energy_delta")


def energy_descriptors(
    subband_energy: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Convert [batch, subband, time] energy into stable profile descriptors."""
    if subband_energy.ndim != 3:
        raise ValueError("subband_energy must have shape [batch, subband, time].")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if torch.any(subband_energy < 0):
        raise ValueError("subband_energy must be non-negative.")

    log_energy = torch.log(subband_energy.clamp_min(epsilon))
    absolute_delta = F.pad(torch.diff(log_energy, dim=-1).abs(), (1, 0))
    return torch.stack((log_energy, absolute_delta), dim=2)


class ConditionedNormalProfile(nn.Module):
    """Static normal statistics indexed by dataset and operating condition."""

    def __init__(
        self,
        condition_ids: Sequence[str],
        mean: torch.Tensor,
        std: torch.Tensor,
        record_counts: torch.Tensor,
        fallback_mean: torch.Tensor,
        fallback_std: torch.Tensor,
        epsilon: float = 1e-8,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        condition_ids = tuple(condition_ids)
        if len(condition_ids) == 0 or len(set(condition_ids)) != len(condition_ids):
            raise ValueError("condition_ids must be non-empty and unique.")
        if mean.ndim != 3 or mean.shape != std.shape:
            raise ValueError("mean and std must have shape [condition, subband, descriptor].")
        if mean.shape[0] != len(condition_ids):
            raise ValueError("condition_ids must match the first statistics dimension.")
        if mean.shape[2] != len(DESCRIPTOR_NAMES):
            raise ValueError("statistics contain an unexpected descriptor count.")
        if record_counts.shape != (len(condition_ids),):
            raise ValueError("record_counts must contain one count per condition.")
        if fallback_mean.shape != mean.shape[1:] or fallback_std.shape != mean.shape[1:]:
            raise ValueError("fallback statistics must have shape [subband, descriptor].")
        if torch.any(std <= 0) or torch.any(fallback_std <= 0):
            raise ValueError("all standard deviations must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.condition_ids = condition_ids
        self._condition_to_index = {
            condition_id: index for index, condition_id in enumerate(condition_ids)
        }
        self.epsilon = float(epsilon)
        self.metadata = dict(metadata or {})
        self.register_buffer("mean", mean.detach().float())
        self.register_buffer("std", std.detach().float())
        self.register_buffer("record_counts", record_counts.detach().long())
        self.register_buffer("fallback_mean", fallback_mean.detach().float())
        self.register_buffer("fallback_std", fallback_std.detach().float())

    @property
    def num_subbands(self) -> int:
        return self.mean.shape[1]

    def standardize(
        self,
        subband_energy: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(condition_ids) != subband_energy.shape[0]:
            raise ValueError("condition_ids must contain one value per batch item.")
        if subband_energy.shape[1] != self.num_subbands:
            raise ValueError("subband count does not match the fitted profile.")
        if self.mean.device != subband_energy.device:
            raise ValueError("move the normal profile to the same device as subband_energy.")

        descriptors = energy_descriptors(subband_energy, self.epsilon)
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

        selected_mean = torch.stack(means).unsqueeze(-1)
        selected_std = torch.stack(standard_deviations).unsqueeze(-1)
        z_scores = (descriptors - selected_mean) / selected_std
        return {
            "descriptors": descriptors,
            "z_scores": z_scores,
            "known_condition": torch.tensor(
                known,
                dtype=torch.bool,
                device=subband_energy.device,
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "descriptor_names": DESCRIPTOR_NAMES,
            "condition_ids": self.condition_ids,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "record_counts": self.record_counts.detach().cpu().tolist(),
            "fallback_mean": self.fallback_mean.detach().cpu().tolist(),
            "fallback_std": self.fallback_std.detach().cpu().tolist(),
            "epsilon": self.epsilon,
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ConditionedNormalProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported normal profile version.")
        if tuple(payload.get("descriptor_names", ())) != DESCRIPTOR_NAMES:
            raise ValueError("normal profile descriptor definition does not match.")
        return cls(
            condition_ids=payload["condition_ids"],
            mean=torch.tensor(payload["mean"]),
            std=torch.tensor(payload["std"]),
            record_counts=torch.tensor(payload["record_counts"]),
            fallback_mean=torch.tensor(payload["fallback_mean"]),
            fallback_std=torch.tensor(payload["fallback_std"]),
            epsilon=float(payload["epsilon"]),
            metadata=payload.get("metadata", {}),
        )


class NormalProfileEstimator:
    """Incrementally estimate normal moments without mixing conditions."""

    def __init__(
        self,
        num_subbands: int,
        split_name: str = "train",
        minimum_records_per_condition: int = 1,
        minimum_std: float = 1e-3,
        epsilon: float = 1e-8,
    ) -> None:
        if split_name != "train":
            raise ValueError("normal profiles must be fitted from the training split.")
        if num_subbands <= 0:
            raise ValueError("num_subbands must be positive.")
        if minimum_records_per_condition <= 0:
            raise ValueError("minimum_records_per_condition must be positive.")
        if minimum_std <= 0 or epsilon <= 0:
            raise ValueError("minimum_std and epsilon must be positive.")

        self.num_subbands = int(num_subbands)
        self.minimum_records_per_condition = int(minimum_records_per_condition)
        self.minimum_std = float(minimum_std)
        self.epsilon = float(epsilon)
        self._moments: dict[str, dict[str, torch.Tensor | int]] = {}

    def update(
        self,
        subband_energy: torch.Tensor,
        condition_ids: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        batch_size = subband_energy.shape[0]
        if subband_energy.ndim != 3 or subband_energy.shape[1] != self.num_subbands:
            raise ValueError("subband_energy has an unexpected shape.")
        if len(condition_ids) != batch_size or len(labels) != batch_size:
            raise ValueError("condition_ids and labels must match the batch size.")
        unexpected = sorted(set(labels) - {"normal"})
        if unexpected:
            raise ValueError(f"normal profile received non-normal labels: {unexpected}.")

        descriptors = energy_descriptors(subband_energy.detach(), self.epsilon).double().cpu()
        record_means = descriptors.mean(dim=-1)
        record_second_moments = descriptors.square().mean(dim=-1)
        for index, condition_id in enumerate(condition_ids):
            if not condition_id:
                raise ValueError("condition_ids must be non-empty strings.")
            if condition_id not in self._moments:
                self._moments[condition_id] = {
                    "count": 0,
                    "mean_sum": torch.zeros_like(record_means[index]),
                    "second_sum": torch.zeros_like(record_second_moments[index]),
                }
            moments = self._moments[condition_id]
            moments["count"] = int(moments["count"]) + 1
            moments["mean_sum"] = moments["mean_sum"] + record_means[index]
            moments["second_sum"] = (
                moments["second_sum"] + record_second_moments[index]
            )

    def finalize(
        self,
        metadata: Mapping[str, object] | None = None,
    ) -> ConditionedNormalProfile:
        if not self._moments:
            raise RuntimeError("no normal records were provided.")

        condition_ids = tuple(sorted(self._moments))
        means = []
        second_moments = []
        counts = []
        for condition_id in condition_ids:
            moments = self._moments[condition_id]
            count = int(moments["count"])
            if count < self.minimum_records_per_condition:
                raise ValueError(
                    f"condition {condition_id!r} has {count} records; "
                    f"requires {self.minimum_records_per_condition}."
                )
            means.append(moments["mean_sum"] / count)
            second_moments.append(moments["second_sum"] / count)
            counts.append(count)

        mean = torch.stack(means)
        second = torch.stack(second_moments)
        variance = (second - mean.square()).clamp_min(self.minimum_std**2)
        fallback_mean = mean.mean(dim=0)
        fallback_second = second.mean(dim=0)
        fallback_variance = (
            fallback_second - fallback_mean.square()
        ).clamp_min(self.minimum_std**2)
        return ConditionedNormalProfile(
            condition_ids=condition_ids,
            mean=mean,
            std=variance.sqrt(),
            record_counts=torch.tensor(counts),
            fallback_mean=fallback_mean,
            fallback_std=fallback_variance.sqrt(),
            epsilon=self.epsilon,
            metadata=metadata,
        )
