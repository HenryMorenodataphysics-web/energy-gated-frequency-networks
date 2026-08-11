"""Standalone Energy-Gated Frequency Neuron implementation.

Dependency: PyTorch only.
Input: mono waveform tensor [batch, 1, time].
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


FilterBand = tuple[str, float | None, float | None]

DEFAULT_FILTER_BANDS: tuple[FilterBand, ...] = (
    ("lowpass", None, 300.0),
    ("bandpass", 300.0, 800.0),
    ("bandpass", 800.0, 2_000.0),
    ("bandpass", 2_000.0, 4_000.0),
    ("highpass", 4_000.0, None),
)

FINE_FILTER_BANDS: tuple[FilterBand, ...] = (
    ("lowpass", None, 250.0),
    ("bandpass", 250.0, 500.0),
    ("bandpass", 500.0, 750.0),
    ("bandpass", 750.0, 1_000.0),
    ("bandpass", 1_000.0, 1_500.0),
    ("bandpass", 1_500.0, 2_500.0),
    ("bandpass", 2_500.0, 4_000.0),
    ("highpass", 4_000.0, None),
)


class EnergyGatedFrequencyNeuron(nn.Module):
    """Frequency filter bank with energy-conditioned gates.

    filter_mode:
        fixed: fixed windowed-sinc FIR kernels (EGFN V1).
        free: FIR coefficients initialized as filters and then learned freely
              (EGFN V1 used in the reported experiments).
        sinc: learnable low/high cutoffs with physically constrained kernels
              (EGFN V2).

    Returns:
        band_outputs: [batch, num_filters, time]
        gates: [batch, num_filters]
        energy: [batch, num_filters]
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        filter_bands: Iterable[FilterBand] | None = None,
        activation: str = "gelu",
        filter_mode: str = "free",
        use_log_energy: bool = True,
        gate_mode: str = "independent",
        gate_hidden_dim: int | None = None,
        min_frequency_hz: float = 20.0,
        min_bandwidth_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if filter_mode not in {"fixed", "free", "sinc"}:
            raise ValueError("filter_mode must be fixed, free, or sinc.")
        if gate_mode not in {"independent", "contextual"}:
            raise ValueError("gate_mode must be independent or contextual.")

        self.sample_rate = int(sample_rate)
        self.kernel_size = int(kernel_size)
        self.activation_name = activation
        self.filter_mode = filter_mode
        self.use_log_energy = use_log_energy
        self.gate_mode = gate_mode
        self.min_frequency_hz = float(min_frequency_hz)
        self.min_bandwidth_hz = float(min_bandwidth_hz)
        self.max_frequency_hz = sample_rate / 2 - min_frequency_hz

        if self.min_frequency_hz <= 0 or self.min_bandwidth_hz <= 0:
            raise ValueError("Frequency constraints must be positive.")
        if self.max_frequency_hz <= self.min_frequency_hz + self.min_bandwidth_hz:
            raise ValueError("Sample rate is too low for the frequency constraints.")

        self.filter_bands = tuple(filter_bands or DEFAULT_FILTER_BANDS)
        self.num_filters = len(self.filter_bands)

        initial_filters = self._create_filter_bank(self.filter_bands).unsqueeze(1)
        if filter_mode == "free":
            self.filters = nn.Parameter(initial_filters)
        elif filter_mode == "fixed":
            self.register_buffer("filters", initial_filters)
        else:
            self._initialize_sinc_parameters()

        self.bias = nn.Parameter(torch.zeros(self.num_filters))
        self.gate_alpha = nn.Parameter(torch.ones(self.num_filters))
        self.gate_beta = nn.Parameter(torch.zeros(self.num_filters))

        if gate_mode == "contextual":
            hidden = gate_hidden_dim or max(8, self.num_filters * 2)
            self.context_gate = nn.Sequential(
                nn.Linear(self.num_filters, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_filters),
            )
        else:
            self.context_gate = None

    def _sinc_lowpass(self, cutoff_hz: float) -> torch.Tensor:
        nyquist = self.sample_rate / 2
        if not 0 < cutoff_hz < nyquist:
            raise ValueError(f"cutoff_hz must be between 0 and {nyquist}.")
        half = self.kernel_size // 2
        n = torch.arange(-half, half + 1, dtype=torch.float32)
        cutoff = cutoff_hz / self.sample_rate
        kernel = 2 * cutoff * torch.sinc(2 * cutoff * n)
        kernel = kernel * torch.hamming_window(self.kernel_size, periodic=False)
        return kernel / kernel.sum().clamp_min(1e-8)

    def _create_filter_bank(self, bands: Iterable[FilterBand]) -> torch.Tensor:
        kernels: list[torch.Tensor] = []
        for kind, low, high in bands:
            if kind == "lowpass":
                if high is None:
                    raise ValueError("lowpass requires a high cutoff.")
                kernel = self._sinc_lowpass(high)
            elif kind == "highpass":
                if low is None:
                    raise ValueError("highpass requires a low cutoff.")
                lowpass = self._sinc_lowpass(low)
                delta = torch.zeros_like(lowpass)
                delta[self.kernel_size // 2] = 1.0
                kernel = delta - lowpass
            elif kind == "bandpass":
                if low is None or high is None:
                    raise ValueError("bandpass requires low and high cutoffs.")
                kernel = self._sinc_lowpass(high) - self._sinc_lowpass(low)
            else:
                raise ValueError(f"Unknown filter kind: {kind}")
            kernels.append(kernel)
        return torch.stack(kernels)

    def _initial_cutoffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        lows, highs = [], []
        for kind, low, high in self.filter_bands:
            if kind == "lowpass":
                low_value = self.min_frequency_hz
                high_value = float(high) if high is not None else 250.0
            elif kind == "highpass":
                low_value = float(low) if low is not None else 4_000.0
                high_value = self.max_frequency_hz
            else:
                if low is None or high is None:
                    raise ValueError("bandpass requires low and high cutoffs.")
                low_value, high_value = float(low), float(high)

            low_value = min(
                max(low_value, self.min_frequency_hz),
                self.max_frequency_hz - self.min_bandwidth_hz,
            )
            high_value = min(
                max(high_value, low_value + self.min_bandwidth_hz),
                self.max_frequency_hz,
            )
            lows.append(low_value)
            highs.append(high_value)
        return torch.tensor(lows), torch.tensor(highs)

    def _low_from_fraction(self, fraction: torch.Tensor) -> torch.Tensor:
        span = self.max_frequency_hz - self.min_bandwidth_hz - self.min_frequency_hz
        return self.min_frequency_hz + fraction * span

    def _initialize_sinc_parameters(self) -> None:
        initial_low, initial_high = self._initial_cutoffs()
        low_fraction = (
            (initial_low - self.min_frequency_hz)
            / (self.max_frequency_hz - self.min_bandwidth_hz - self.min_frequency_hz)
        ).clamp(1e-4, 1 - 1e-4)
        adjusted_low = self._low_from_fraction(low_fraction)
        bandwidth_fraction = (
            (initial_high - adjusted_low - self.min_bandwidth_hz)
            / (self.max_frequency_hz - adjusted_low - self.min_bandwidth_hz)
        ).clamp(1e-4, 1 - 1e-4)

        self.low_cutoff_raw = nn.Parameter(torch.logit(low_fraction))
        self.bandwidth_raw = nn.Parameter(torch.logit(bandwidth_fraction))

        half = self.kernel_size // 2
        self.register_buffer(
            "sinc_n",
            torch.arange(-half, half + 1, dtype=torch.float32),
        )
        self.register_buffer(
            "sinc_window",
            torch.hann_window(self.kernel_size, periodic=False),
        )

    def cutoff_frequencies_hz(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.filter_mode != "sinc":
            raise RuntimeError("Cutoffs are only available in sinc mode.")
        low = self._low_from_fraction(torch.sigmoid(self.low_cutoff_raw))
        available = self.max_frequency_hz - low - self.min_bandwidth_hz
        bandwidth = self.min_bandwidth_hz + torch.sigmoid(self.bandwidth_raw) * available
        return low, low + bandwidth

    def get_filters(self) -> torch.Tensor:
        if self.filter_mode != "sinc":
            return self.filters

        low_hz, high_hz = self.cutoff_frequencies_hz()
        low = (low_hz / self.sample_rate).unsqueeze(1)
        high = (high_hz / self.sample_rate).unsqueeze(1)
        n = self.sinc_n.unsqueeze(0)

        lowpass_high = 2 * high * torch.sinc(2 * high * n)
        lowpass_low = 2 * low * torch.sinc(2 * low * n)
        kernels = (lowpass_high - lowpass_low) * self.sinc_window.unsqueeze(0)
        kernels = kernels / kernels.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return kernels.unsqueeze(1)

    def _activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(tensor)
        if self.activation_name == "tanh":
            return torch.tanh(tensor)
        if self.activation_name == "gelu":
            return F.gelu(tensor)
        if self.activation_name == "silu":
            return F.silu(tensor)
        raise ValueError("activation must be relu, tanh, gelu, or silu.")

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Input must have shape [batch, 1, time].")

        filtered = F.conv1d(
            x,
            self.get_filters(),
            padding=self.kernel_size // 2,
        )
        activated = self._activation(
            filtered + self.bias.view(1, -1, 1)
        )
        energy = filtered.square().mean(dim=-1)
        gate_input = torch.log1p(energy) if self.use_log_energy else energy

        if self.gate_mode == "contextual":
            if self.context_gate is None:
                raise RuntimeError("Missing contextual gate network.")
            gates = torch.sigmoid(self.context_gate(gate_input))
        else:
            gates = torch.sigmoid(
                self.gate_alpha.view(1, -1) * gate_input
                + self.gate_beta.view(1, -1)
            )

        band_outputs = gates.unsqueeze(-1) * activated
        return band_outputs, gates, energy


class FrequencyPooling(nn.Module):
    """Global interpretable pooling used by the small EGFN classifier."""

    def forward(
        self,
        band_outputs: torch.Tensor,
        gates: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                band_outputs.mean(dim=-1),
                band_outputs.amax(dim=-1),
                band_outputs.std(dim=-1, unbiased=False),
                torch.log1p(energy),
                gates,
            ],
            dim=-1,
        )


class FrequencyGatedClassifier(nn.Module):
    """Compact EGFN with global statistical pooling and an MLP."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        filter_bands: tuple[FilterBand, ...] = FINE_FILTER_BANDS,
        filter_mode: str = "free",
        gate_mode: str = "independent",
    ) -> None:
        super().__init__()
        self.frontend = EnergyGatedFrequencyNeuron(
            sample_rate=sample_rate,
            kernel_size=kernel_size,
            filter_bands=filter_bands,
            filter_mode=filter_mode,
            gate_mode=gate_mode,
        )
        self.pooling = FrequencyPooling()
        features = self.frontend.num_filters * 5
        self.classifier = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bands, gates, energy = self.frontend(x)
        features = self.pooling(bands, gates, energy)
        logits = self.classifier(features)
        return {
            "logits": logits,
            "features": features,
            "band_outputs": bands,
            "gates": gates,
            "energy": energy,
        }


class FrequencyGatedTemporalClassifier(nn.Module):
    """EGFN frontend followed by the temporal-wide classification head."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        filter_bands: tuple[FilterBand, ...] = FINE_FILTER_BANDS,
        filter_mode: str = "free",
        gate_mode: str = "independent",
        frontend_channels: int = 32,
        temporal_channels: tuple[int, int] = (64, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.frontend = EnergyGatedFrequencyNeuron(
            sample_rate=sample_rate,
            kernel_size=kernel_size,
            filter_bands=filter_bands,
            filter_mode=filter_mode,
            gate_mode=gate_mode,
        )

        num_filters = self.frontend.num_filters
        c1, c2 = temporal_channels
        if frontend_channels > num_filters:
            self.frontend_projection = nn.Sequential(
                nn.Conv1d(num_filters, frontend_channels, kernel_size=1),
                nn.BatchNorm1d(frontend_channels),
                nn.GELU(),
            )
            temporal_input = frontend_channels
        else:
            self.frontend_projection = nn.Identity()
            temporal_input = num_filters

        self.temporal_head = nn.Sequential(
            nn.Conv1d(temporal_input, c1, kernel_size=15, padding=7),
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
        bands, gates, energy = self.frontend(x)
        projected = self.frontend_projection(bands)
        temporal_features = self.temporal_head(projected)
        logits = self.classifier(temporal_features)
        return {
            "logits": logits,
            "band_outputs": bands,
            "projected_band_outputs": projected,
            "gates": gates,
            "energy": energy,
            "temporal_features": temporal_features,
        }


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
) -> torch.optim.Optimizer:
    """AdamW that excludes physical Sinc cutoffs from weight decay."""
    frequency_parameters = []
    regular_parameters = []
    for name, parameter in model.named_parameters():
        if name.endswith(("low_cutoff_raw", "bandwidth_raw")):
            frequency_parameters.append(parameter)
        else:
            regular_parameters.append(parameter)

    groups = [{"params": regular_parameters, "weight_decay": weight_decay}]
    if frequency_parameters:
        groups.append({"params": frequency_parameters, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=learning_rate)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use filter_mode="free" to reproduce EGFN V1 temporal wide.
    # Use filter_mode="sinc" for the constrained EGFN V2 frontend.
    model = FrequencyGatedTemporalClassifier(
        num_classes=2,
        sample_rate=16_000,
        filter_mode="free",
        gate_mode="independent",
        frontend_channels=32,
        temporal_channels=(64, 128),
        dropout=0.25,
    ).to(device)

    waveform = torch.randn(4, 1, 32_000, device=device)
    targets = torch.randint(0, 2, (4,), device=device)
    outputs = model(waveform)
    loss = F.cross_entropy(outputs["logits"], targets)
    loss.backward()

    print(f"device={device}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print(f"logits={tuple(outputs['logits'].shape)}")
    print(f"bands={tuple(outputs['band_outputs'].shape)}")
    print(f"gates={tuple(outputs['gates'].shape)}")
    print(f"energy={tuple(outputs['energy'].shape)}")
    print(f"loss={loss.item():.4f}")
