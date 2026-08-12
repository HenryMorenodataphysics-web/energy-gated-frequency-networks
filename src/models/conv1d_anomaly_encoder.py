from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DAnomalyEncoder(nn.Module):
    """Small shared-channel Conv1D encoder for one-class comparison."""

    def __init__(
        self,
        embedding_channels: int = 8,
        channels: tuple[int, int, int] = (4, 8, 8),
    ) -> None:
        super().__init__()
        if embedding_channels <= 0 or any(channel <= 0 for channel in channels):
            raise ValueError("all channel counts must be positive.")
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(third, embedding_channels, bias=False)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim != 3 or waveform.shape[1] < 1:
            raise ValueError("waveform must have shape [batch, channels, time].")
        batch_size, audio_channels, samples = waveform.shape
        channel_embeddings = self.encoder(
            waveform.reshape(batch_size * audio_channels, 1, samples)
        ).flatten(1)
        channel_embeddings = self.projection(channel_embeddings).reshape(
            batch_size,
            audio_channels,
            -1,
        )
        return {
            "channel_embeddings": channel_embeddings,
            "embedding": channel_embeddings.mean(dim=1),
        }
