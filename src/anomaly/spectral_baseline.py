from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def _hz_to_mel(frequency_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency_hz / 700.0)


def _mel_to_hz(frequency_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, frequency_mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> torch.Tensor:
    if sample_rate <= 0 or n_fft <= 1 or n_mels <= 1:
        raise ValueError("mel filterbank configuration values must be positive.")
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_edges = torch.linspace(
        _hz_to_mel(frequencies[:1])[0],
        _hz_to_mel(frequencies[-1:])[0],
        n_mels + 2,
    )
    edges_hz = _mel_to_hz(mel_edges)
    filters = []
    for low, center, high in zip(
        edges_hz[:-2], edges_hz[1:-1], edges_hz[2:], strict=True
    ):
        rising = (frequencies - low) / (center - low).clamp_min(1e-12)
        falling = (high - frequencies) / (high - center).clamp_min(1e-12)
        triangular = torch.minimum(rising, falling).clamp(0.0, 1.0)
        filters.append(triangular / triangular.sum().clamp_min(1e-12))
    return torch.stack(filters)


class LogMelFrontend(nn.Module):
    """Dataset-independent, level-preserving multichannel log-mel frontend."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if hop_length <= 0 or hop_length > n_fft or epsilon <= 0:
            raise ValueError("invalid log-mel frontend configuration.")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.epsilon = float(epsilon)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(sample_rate, n_fft, n_mels),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] < 1:
            raise ValueError("waveform must have shape [batch, channel, time].")
        batch, channels, samples = waveform.shape
        spectrum = torch.stft(
            waveform.reshape(batch * channels, samples),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square().reshape(
            batch, channels, spectrum.shape[-2], spectrum.shape[-1]
        ).mean(dim=1)
        mel_power = torch.einsum("mf,bft->bmt", self.mel_filterbank, power)
        return torch.log(mel_power.clamp_min(self.epsilon))


class ConditionedSpectralBaselines:
    """Normal-only Gaussian, frame-kNN, and PCA reconstruction baselines."""

    def __init__(
        self,
        condition_ids: Sequence[str],
        mean: torch.Tensor,
        std: torch.Tensor,
        components: torch.Tensor,
        memories: Sequence[torch.Tensor],
        fallback_mean: torch.Tensor,
        fallback_std: torch.Tensor,
        fallback_components: torch.Tensor,
        fallback_memory: torch.Tensor,
        top_fraction: float,
        minimum_std: float,
        query_chunk_size: int,
    ) -> None:
        self.condition_ids = tuple(condition_ids)
        self.mean = mean.float()
        self.std = std.float()
        self.components = components.float()
        self.memories = tuple(memory.float() for memory in memories)
        self.fallback_mean = fallback_mean.float()
        self.fallback_std = fallback_std.float()
        self.fallback_components = fallback_components.float()
        self.fallback_memory = fallback_memory.float()
        self.top_fraction = float(top_fraction)
        self.minimum_std = float(minimum_std)
        self.query_chunk_size = int(query_chunk_size)
        self._condition_to_index = {
            condition_id: index for index, condition_id in enumerate(self.condition_ids)
        }

    def to(self, device: torch.device | str) -> ConditionedSpectralBaselines:
        return ConditionedSpectralBaselines(
            self.condition_ids,
            self.mean.to(device),
            self.std.to(device),
            self.components.to(device),
            [memory.to(device) for memory in self.memories],
            self.fallback_mean.to(device),
            self.fallback_std.to(device),
            self.fallback_components.to(device),
            self.fallback_memory.to(device),
            self.top_fraction,
            self.minimum_std,
            self.query_chunk_size,
        )

    @staticmethod
    def _nearest_distance(
        query: torch.Tensor,
        memory: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        memory_squared = memory.square().sum(dim=1).unsqueeze(0)
        distances = []
        for start in range(0, query.shape[0], chunk_size):
            chunk = query[start : start + chunk_size]
            pairwise = (
                chunk.square().sum(dim=1, keepdim=True)
                + memory_squared
                - 2.0 * chunk @ memory.T
            ).clamp_min(0.0)
            distances.append(pairwise.amin(dim=1) / query.shape[1])
        return torch.cat(distances)

    def _top_pool(self, values: torch.Tensor) -> torch.Tensor:
        count = max(1, math.ceil(values.shape[-1] * self.top_fraction))
        return values.topk(count, dim=-1).values.mean(dim=-1)

    def score(
        self,
        log_mel: torch.Tensor,
        condition_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if log_mel.ndim != 3 or len(condition_ids) != log_mel.shape[0]:
            raise ValueError("log_mel and condition_ids must share the batch dimension.")
        frames = log_mel.transpose(1, 2)
        gaussian_scores = []
        knn_scores = []
        reconstruction_scores = []
        known = []
        for sample, condition_id in zip(frames, condition_ids, strict=True):
            index = self._condition_to_index.get(condition_id)
            if index is None:
                mean = self.fallback_mean
                std = self.fallback_std
                components = self.fallback_components
                memory = self.fallback_memory
                known.append(False)
            else:
                mean = self.mean[index]
                std = self.std[index]
                components = self.components[index]
                memory = self.memories[index]
                known.append(True)
            scale = std.clamp_min(self.minimum_std)
            standardized = (sample - mean) / scale
            standardized_memory = (memory - mean) / scale
            gaussian_local = standardized.square().mean(dim=1)
            knn_local = self._nearest_distance(
                standardized,
                standardized_memory,
                self.query_chunk_size,
            )
            centered = sample - mean
            reconstruction = (centered @ components.T) @ components
            reconstruction_local = (centered - reconstruction).square().mean(dim=1)
            gaussian_scores.append(self._top_pool(gaussian_local))
            knn_scores.append(self._top_pool(knn_local))
            reconstruction_scores.append(self._top_pool(reconstruction_local))
        return {
            "logmel_gaussian_score": torch.stack(gaussian_scores),
            "logmel_knn_score": torch.stack(knn_scores),
            "logmel_pca_reconstruction_score": torch.stack(reconstruction_scores),
            "known_condition": torch.tensor(
                known, dtype=torch.bool, device=log_mel.device
            ),
        }


class ConditionedSpectralBaselineEstimator:
    def __init__(
        self,
        n_mels: int,
        memory_size: int = 2_048,
        pca_rank: int = 16,
        top_fraction: float = 0.05,
        minimum_std: float = 1e-3,
        query_chunk_size: int = 2_048,
        seed: int = 42,
    ) -> None:
        if n_mels <= 1 or memory_size <= 0 or not 0 < pca_rank < n_mels:
            raise ValueError("invalid spectral baseline dimensions.")
        if not 0 < top_fraction <= 1 or minimum_std <= 0 or query_chunk_size <= 0:
            raise ValueError("invalid spectral baseline scoring configuration.")
        self.n_mels = int(n_mels)
        self.memory_size = int(memory_size)
        self.pca_rank = int(pca_rank)
        self.top_fraction = float(top_fraction)
        self.minimum_std = float(minimum_std)
        self.query_chunk_size = int(query_chunk_size)
        self._generator = torch.Generator().manual_seed(seed)
        self._moments: dict[str, dict[str, torch.Tensor | int]] = {}
        self._memories: dict[str, torch.Tensor] = {}
        self._priorities: dict[str, torch.Tensor] = {}

    def _update_key(self, key: str, frames: torch.Tensor) -> None:
        values = frames.detach().double().cpu()
        moments = self._moments.setdefault(
            key,
            {
                "count": 0,
                "sum": torch.zeros(self.n_mels, dtype=torch.float64),
                "cross": torch.zeros(self.n_mels, self.n_mels, dtype=torch.float64),
            },
        )
        moments["count"] = int(moments["count"]) + values.shape[0]
        moments["sum"] = moments["sum"] + values.sum(dim=0)
        moments["cross"] = moments["cross"] + values.T @ values
        priorities = torch.rand(values.shape[0], generator=self._generator)
        stored_values = values.float()
        if key in self._memories:
            stored_values = torch.cat((self._memories[key], stored_values))
            priorities = torch.cat((self._priorities[key], priorities))
        keep = min(self.memory_size, stored_values.shape[0])
        indices = priorities.topk(keep).indices
        self._memories[key] = stored_values[indices]
        self._priorities[key] = priorities[indices]

    def update(self, log_mel: torch.Tensor, condition_ids: Sequence[str]) -> None:
        if log_mel.ndim != 3 or log_mel.shape[1] != self.n_mels:
            raise ValueError("log_mel must have shape [batch, mel, time].")
        if len(condition_ids) != log_mel.shape[0]:
            raise ValueError("condition_ids must contain one value per example.")
        frames = log_mel.transpose(1, 2)
        grouped_examples: dict[str, list[int]] = {}
        for example_index, condition_id in enumerate(condition_ids):
            if not condition_id:
                raise ValueError("condition_ids must be non-empty strings.")
            grouped_examples.setdefault(condition_id, []).append(example_index)
        self._update_key("__fallback__", frames.reshape(-1, self.n_mels))
        for condition_id, example_indices in grouped_examples.items():
            self._update_key(
                condition_id,
                frames[example_indices].reshape(-1, self.n_mels),
            )

    def _statistics(
        self,
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moments = self._moments[key]
        count = int(moments["count"])
        mean = moments["sum"] / count
        covariance = moments["cross"] / count - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        variance = covariance.diagonal().clamp_min(self.minimum_std**2)
        _, eigenvectors = torch.linalg.eigh(covariance)
        components = eigenvectors[:, -self.pca_rank :].T
        return mean.float(), variance.sqrt().float(), components.float()

    def finalize(self) -> ConditionedSpectralBaselines:
        condition_ids = tuple(sorted(set(self._moments) - {"__fallback__"}))
        if not condition_ids or "__fallback__" not in self._moments:
            raise RuntimeError("no normal log-mel features were provided.")
        statistics = [self._statistics(condition_id) for condition_id in condition_ids]
        fallback = self._statistics("__fallback__")
        return ConditionedSpectralBaselines(
            condition_ids=condition_ids,
            mean=torch.stack([item[0] for item in statistics]),
            std=torch.stack([item[1] for item in statistics]),
            components=torch.stack([item[2] for item in statistics]),
            memories=[self._memories[item] for item in condition_ids],
            fallback_mean=fallback[0],
            fallback_std=fallback[1],
            fallback_components=fallback[2],
            fallback_memory=self._memories["__fallback__"],
            top_fraction=self.top_fraction,
            minimum_std=self.minimum_std,
            query_chunk_size=self.query_chunk_size,
        )
