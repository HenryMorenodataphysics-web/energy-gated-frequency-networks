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
    """Frequency filter bank with learned energy gates.

    Input shape:
        x: [batch, 1, time]

    Output:
        y: [batch, num_bands, time]
        gates: [batch, num_bands]
        energy: [batch, num_bands]
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        kernel_size: int = 101,
        filter_bands: Iterable[FilterBand] | None = None,
        activation: str = "gelu",
        learnable_filters: bool = False,
        filter_mode: str | None = None,
        use_log_energy: bool = True,
        gate_mode: str = "independent",
        gate_hidden_dim: int | None = None,
        min_frequency_hz: float = 20.0,
        min_bandwidth_hz: float = 50.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal alignment.")
        if gate_mode not in {"none", "independent", "contextual"}:
            raise ValueError("gate_mode must be 'none', 'independent', or 'contextual'.")
        if filter_mode is None:
            filter_mode = "free" if learnable_filters else "fixed"
        if filter_mode not in {"fixed", "free", "sinc"}:
            raise ValueError("filter_mode must be 'fixed', 'free', or 'sinc'.")
        if learnable_filters and filter_mode != "free":
            raise ValueError("learnable_filters=True is only compatible with filter_mode='free'.")

        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.use_log_energy = use_log_energy
        self.gate_mode = gate_mode
        self.filter_mode = filter_mode
        self.min_frequency_hz = float(min_frequency_hz)
        self.min_bandwidth_hz = float(min_bandwidth_hz)
        self.max_frequency_hz = sample_rate / 2 - self.min_frequency_hz

        if self.min_frequency_hz <= 0:
            raise ValueError("min_frequency_hz must be positive.")
        if self.min_bandwidth_hz <= 0:
            raise ValueError("min_bandwidth_hz must be positive.")
        if self.max_frequency_hz <= self.min_frequency_hz + self.min_bandwidth_hz:
            raise ValueError("The sample rate is too low for the requested frequency constraints.")

        if filter_bands is None:
            filter_bands = DEFAULT_FILTER_BANDS

        self.filter_bands = tuple(filter_bands)
        self.num_filters = len(self.filter_bands)

        filters = self._create_filter_bank(self.filter_bands).unsqueeze(1)
        if filter_mode == "free":
            self.filters = nn.Parameter(filters)
        elif filter_mode == "fixed":
            self.register_buffer("filters", filters)
        else:
            initial_low, initial_high = self._initial_sinc_cutoffs(self.filter_bands)
            low_fraction = (
                (initial_low - self.min_frequency_hz)
                / (self.max_frequency_hz - self.min_bandwidth_hz - self.min_frequency_hz)
            )
            low_fraction = low_fraction.clamp(1e-4, 1.0 - 1e-4)
            initial_low = self._low_cutoffs_from_fraction(low_fraction)
            bandwidth_fraction = (
                (initial_high - initial_low - self.min_bandwidth_hz)
                / (self.max_frequency_hz - initial_low - self.min_bandwidth_hz)
            )
            bandwidth_fraction = bandwidth_fraction.clamp(1e-4, 1.0 - 1e-4)
            self.low_cutoff_raw = nn.Parameter(torch.logit(low_fraction))
            self.bandwidth_raw = nn.Parameter(torch.logit(bandwidth_fraction))

            half_width = self.kernel_size // 2
            n = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
            self.register_buffer("sinc_n", n)
            self.register_buffer(
                "sinc_window",
                torch.hann_window(self.kernel_size, periodic=False),
            )

        self.bias = nn.Parameter(torch.zeros(self.num_filters))
        if gate_mode == "none":
            self.register_parameter("gate_alpha", None)
            self.register_parameter("gate_beta", None)
        else:
            self.gate_alpha = nn.Parameter(torch.ones(self.num_filters))
            self.gate_beta = nn.Parameter(torch.zeros(self.num_filters))

        if gate_mode == "contextual":
            hidden_dim = gate_hidden_dim or max(8, self.num_filters * 2)
            self.context_gate = nn.Sequential(
                nn.Linear(self.num_filters, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_filters),
            )
        else:
            self.context_gate = None

    def _sinc_lowpass(self, cutoff_hz: float) -> torch.Tensor:
        nyquist = self.sample_rate / 2
        if cutoff_hz <= 0 or cutoff_hz >= nyquist:
            raise ValueError(f"cutoff_hz must be between 0 and {nyquist} Hz.")

        half_width = self.kernel_size // 2
        n = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
        normalized_cutoff = cutoff_hz / self.sample_rate

        h = 2 * normalized_cutoff * torch.sinc(2 * normalized_cutoff * n)
        window = torch.hamming_window(self.kernel_size, periodic=False)
        h = h * window
        return h / h.sum()

    def _create_filter_bank(self, filter_bands: Iterable[FilterBand]) -> torch.Tensor:
        filters: list[torch.Tensor] = []

        for filter_type, f_low, f_high in filter_bands:
            if filter_type == "lowpass":
                if f_high is None:
                    raise ValueError("lowpass filters require f_high.")
                h = self._sinc_lowpass(f_high)
            elif filter_type == "highpass":
                if f_low is None:
                    raise ValueError("highpass filters require f_low.")
                low = self._sinc_lowpass(f_low)
                delta = torch.zeros_like(low)
                delta[self.kernel_size // 2] = 1.0
                h = delta - low
            elif filter_type == "bandpass":
                if f_low is None or f_high is None:
                    raise ValueError("bandpass filters require f_low and f_high.")
                low_high = self._sinc_lowpass(f_high)
                low_low = self._sinc_lowpass(f_low)
                h = low_high - low_low
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

            filters.append(h)

        return torch.stack(filters, dim=0)

    def _initial_sinc_cutoffs(
        self,
        filter_bands: Iterable[FilterBand],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lows = []
        highs = []
        for filter_type, f_low, f_high in filter_bands:
            if filter_type == "lowpass":
                low = self.min_frequency_hz
                high = float(f_high) if f_high is not None else self.min_frequency_hz + 250.0
            elif filter_type == "highpass":
                low = float(f_low) if f_low is not None else self.max_frequency_hz - 1_000.0
                high = self.max_frequency_hz
            else:
                if f_low is None or f_high is None:
                    raise ValueError("bandpass filters require f_low and f_high.")
                low = float(f_low)
                high = float(f_high)

            low = min(max(low, self.min_frequency_hz), self.max_frequency_hz - self.min_bandwidth_hz)
            high = min(max(high, low + self.min_bandwidth_hz), self.max_frequency_hz)
            lows.append(low)
            highs.append(high)
        return torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32)

    def _low_cutoffs_from_fraction(self, fraction: torch.Tensor) -> torch.Tensor:
        low_range = self.max_frequency_hz - self.min_bandwidth_hz - self.min_frequency_hz
        return self.min_frequency_hz + fraction * low_range

    def cutoff_frequencies_hz(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.filter_mode != "sinc":
            raise RuntimeError("Cutoff parameters are only available in sinc filter mode.")
        low = self._low_cutoffs_from_fraction(torch.sigmoid(self.low_cutoff_raw))
        available_bandwidth = self.max_frequency_hz - low - self.min_bandwidth_hz
        bandwidth = self.min_bandwidth_hz + torch.sigmoid(self.bandwidth_raw) * available_bandwidth
        return low, low + bandwidth

    def get_filters(self) -> torch.Tensor:
        if self.filter_mode != "sinc":
            return self.filters

        low_hz, high_hz = self.cutoff_frequencies_hz()
        low = (low_hz / self.sample_rate).unsqueeze(1)
        high = (high_hz / self.sample_rate).unsqueeze(1)
        n = self.sinc_n.unsqueeze(0)
        lowpass_high = 2.0 * high * torch.sinc(2.0 * high * n)
        lowpass_low = 2.0 * low * torch.sinc(2.0 * low * n)
        filters = (lowpass_high - lowpass_low) * self.sinc_window.unsqueeze(0)
        filters = filters / filters.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return filters.unsqueeze(1)

    def _activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(z)
        if self.activation_name == "tanh":
            return torch.tanh(z)
        if self.activation_name == "gelu":
            return F.gelu(z)
        if self.activation_name == "silu":
            return F.silu(z)
        raise ValueError("activation must be one of: relu, tanh, gelu, silu.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Input must have shape [batch, 1, time].")

        padding = self.kernel_size // 2
        u = F.conv1d(x, self.get_filters(), padding=padding)
        z = u + self.bias.view(1, -1, 1)
        a = self._activation(z)

        energy = torch.mean(u.square(), dim=-1)
        gate_input = torch.log1p(energy) if self.use_log_energy else energy
        if self.gate_mode == "none":
            gates = torch.ones_like(energy)
        elif self.gate_mode == "contextual":
            if self.context_gate is None:
                raise RuntimeError("context_gate is missing for contextual gate mode.")
            gates = torch.sigmoid(self.context_gate(gate_input))
        else:
            if self.gate_alpha is None or self.gate_beta is None:
                raise RuntimeError("Independent gate parameters are missing.")
            gates = torch.sigmoid(
                self.gate_alpha.view(1, -1) * gate_input + self.gate_beta.view(1, -1)
            )

        y = gates.unsqueeze(-1) * a
        return y, gates, energy
