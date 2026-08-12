from __future__ import annotations

import torch
import torch.nn as nn

from src.models.event_pooling import soft_event_pool


class Conv1DAnomalyEncoder(nn.Module):
    """Small shared-channel Conv1D encoder for one-class comparison."""

    def __init__(
        self,
        embedding_channels: int = 8,
        channels: tuple[int, int, int] = (4, 8, 8),
        event_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_channels <= 0 or any(channel <= 0 for channel in channels):
            raise ValueError("all channel counts must be positive.")
        if event_temperature <= 0:
            raise ValueError("event_temperature must be positive.")
        self.event_temperature = float(event_temperature)
        first, second, third = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, first, kernel_size=31, stride=4, padding=15, bias=False),
            nn.GroupNorm(1, first, affine=False),
            nn.GELU(),
            nn.Conv1d(first, second, kernel_size=15, stride=4, padding=7, bias=False),
            nn.GroupNorm(1, second, affine=False),
            nn.GELU(),
            nn.Conv1d(second, third, kernel_size=9, stride=4, padding=4, bias=False),
            nn.GroupNorm(1, third, affine=False),
            nn.GELU(),
        )
        self.projection = nn.Linear(third, embedding_channels, bias=False)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim != 3 or waveform.shape[1] < 1:
            raise ValueError("waveform must have shape [batch, channels, time].")
        batch_size, audio_channels, samples = waveform.shape
        feature_map = self.encoder(
            waveform.reshape(batch_size * audio_channels, 1, samples)
        )
        sustained = self.projection(feature_map.mean(dim=-1)).reshape(
            batch_size,
            audio_channels,
            -1,
        )
        event = self.projection(
            soft_event_pool(feature_map, self.event_temperature)
        ).reshape(batch_size, audio_channels, -1)
        sustained_embedding = sustained.mean(dim=1)
        event_embedding = event.mean(dim=1)
        return {
            "feature_map": feature_map.reshape(
                batch_size, audio_channels, feature_map.shape[1], feature_map.shape[2]
            ).mean(dim=1),
            "channel_embeddings": sustained,
            "sustained_embedding": sustained_embedding,
            "event_embedding": event_embedding,
            "embedding": torch.cat((sustained_embedding, event_embedding), dim=1),
        }
