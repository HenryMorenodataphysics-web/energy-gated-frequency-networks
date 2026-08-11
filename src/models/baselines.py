from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DBaseline(nn.Module):
    """Small waveform CNN baseline for comparison with EGFN."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, int, int] = (16, 32, 64),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c1, c2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c2, c3, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return {"logits": logits}
