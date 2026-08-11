from __future__ import annotations

import torch
import torch.nn as nn

from src.blocks.energy_gated_frequency_neuron import (
    FINE_FILTER_BANDS,
    DEFAULT_FILTER_BANDS,
    EnergyGatedFrequencyNeuron,
    FilterBand,
)


class FrequencyPooling(nn.Module):
    """Summarize temporal band responses into interpretable features."""

    def __init__(self, include_energy: bool = True, include_gates: bool = True) -> None:
        super().__init__()
        self.include_energy = include_energy
        self.include_gates = include_gates

    def feature_multiplier(self) -> int:
        multiplier = 3
        if self.include_energy:
            multiplier += 1
        if self.include_gates:
            multiplier += 1
        return multiplier

    def forward(
        self,
        y: torch.Tensor,
        gates: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        mean = y.mean(dim=-1)
        max_values = y.amax(dim=-1)
        std = y.std(dim=-1, unbiased=False)

        features = [mean, max_values, std]
        if self.include_energy:
            features.append(torch.log1p(energy))
        if self.include_gates:
            features.append(gates)

        return torch.cat(features, dim=-1)


class FrequencyGatedClassifier(nn.Module):
    """Small classifier that keeps the custom frequency block as the main actor."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        activation: str = "gelu",
        learnable_filters: bool = False,
        filter_mode: str | None = None,
        filter_bands: tuple[FilterBand, ...] | None = None,
        gate_mode: str = "independent",
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.frontend = EnergyGatedFrequencyNeuron(
            sample_rate=sample_rate,
            kernel_size=kernel_size,
            filter_bands=filter_bands,
            activation=activation,
            learnable_filters=learnable_filters,
            filter_mode=filter_mode,
            gate_mode=gate_mode,
            gate_hidden_dim=gate_hidden_dim,
        )
        self.pooling = FrequencyPooling(include_energy=True, include_gates=True)

        in_features = self.frontend.num_filters * self.pooling.feature_multiplier()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y, gates, energy = self.frontend(x)
        features = self.pooling(y, gates, energy)
        logits = self.classifier(features)
        return {
            "logits": logits,
            "features": features,
            "band_outputs": y,
            "gates": gates,
            "energy": energy,
        }


class FrequencyGatedTemporalClassifier(nn.Module):
    """EGFN frontend followed by a lightweight temporal convolution head."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        temporal_channels: tuple[int, int] = (32, 64),
        frontend_channels: int | None = None,
        dropout: float = 0.15,
        activation: str = "gelu",
        learnable_filters: bool = False,
        filter_mode: str | None = None,
        filter_bands: tuple[FilterBand, ...] | None = None,
        gate_mode: str = "independent",
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.frontend = EnergyGatedFrequencyNeuron(
            sample_rate=sample_rate,
            kernel_size=kernel_size,
            filter_bands=filter_bands,
            activation=activation,
            learnable_filters=learnable_filters,
            filter_mode=filter_mode,
            gate_mode=gate_mode,
            gate_hidden_dim=gate_hidden_dim,
        )

        c1, c2 = temporal_channels
        temporal_input_channels = self.frontend.num_filters
        self.frontend_projection = nn.Identity()
        if frontend_channels is not None and frontend_channels > self.frontend.num_filters:
            self.frontend_projection = nn.Sequential(
                nn.Conv1d(self.frontend.num_filters, frontend_channels, kernel_size=1),
                nn.BatchNorm1d(frontend_channels),
                nn.GELU(),
            )
            temporal_input_channels = frontend_channels

        self.temporal_head = nn.Sequential(
            nn.Conv1d(temporal_input_channels, c1, kernel_size=15, padding=7),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c1, c2, kernel_size=9, padding=4),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y, gates, energy = self.frontend(x)
        projected = self.frontend_projection(y)
        temporal_features = self.temporal_head(projected)
        logits = self.classifier(temporal_features)
        return {
            "logits": logits,
            "band_outputs": y,
            "projected_band_outputs": projected,
            "gates": gates,
            "energy": energy,
            "temporal_features": temporal_features,
        }


class MatchedConvTemporalClassifier(nn.Module):
    """Unconstrained waveform frontend with the same temporal head as EGFN."""

    def __init__(
        self,
        num_classes: int,
        num_frontend_filters: int = 8,
        kernel_size: int = 101,
        temporal_channels: tuple[int, int] = (32, 64),
        frontend_channels: int | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal alignment.")

        self.waveform_frontend = nn.Sequential(
            nn.Conv1d(
                1,
                num_frontend_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
        )

        c1, c2 = temporal_channels
        temporal_input_channels = num_frontend_filters
        self.frontend_projection = nn.Identity()
        if frontend_channels is not None and frontend_channels > num_frontend_filters:
            self.frontend_projection = nn.Sequential(
                nn.Conv1d(num_frontend_filters, frontend_channels, kernel_size=1),
                nn.BatchNorm1d(frontend_channels),
                nn.GELU(),
            )
            temporal_input_channels = frontend_channels

        self.temporal_head = nn.Sequential(
            nn.Conv1d(temporal_input_channels, c1, kernel_size=15, padding=7),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c1, c2, kernel_size=9, padding=4),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        band_outputs = self.waveform_frontend(x)
        projected = self.frontend_projection(band_outputs)
        temporal_features = self.temporal_head(projected)
        logits = self.classifier(temporal_features)
        return {
            "logits": logits,
            "band_outputs": band_outputs,
            "projected_band_outputs": projected,
            "temporal_features": temporal_features,
        }


def get_filter_bands(name: str) -> tuple[FilterBand, ...]:
    if name == "default":
        return DEFAULT_FILTER_BANDS
    if name == "fine":
        return FINE_FILTER_BANDS
    raise ValueError("filter bank must be 'default' or 'fine'.")
