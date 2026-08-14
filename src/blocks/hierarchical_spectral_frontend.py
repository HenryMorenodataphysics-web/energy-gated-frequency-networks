from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalSpectralFrontend(nn.Module):
    """Three-level spectral router with real temporal subband transforms.

    The block selects and transforms time-frequency evidence. It intentionally
    does not produce an anomaly score.
    """

    def __init__(
        self,
        sample_rate: int,
        macro_edges_hz: Sequence[float],
        subbands_per_macro: Sequence[int] = (4, 8, 4),
        n_fft: int = 512,
        hop_length: int = 128,
        temporal_channels: int = 4,
        temporal_kernel_size: int = 5,
        learnable_subband_weights: bool = False,
        gate_mode: str = "hierarchical",
        normalize_gate_inputs: bool = False,
        conditional_subgates: bool = False,
        harmonic_context: bool = False,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if n_fft <= 1 or n_fft % 2 != 0:
            raise ValueError("n_fft must be an even integer greater than 1.")
        if hop_length <= 0 or hop_length > n_fft:
            raise ValueError("hop_length must be between 1 and n_fft.")
        if temporal_channels <= 0:
            raise ValueError("temporal_channels must be positive.")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if gate_mode not in {"none", "macro", "subband", "hierarchical"}:
            raise ValueError(
                "gate_mode must be none, macro, subband, or hierarchical."
            )

        macro_edges = torch.as_tensor(tuple(macro_edges_hz), dtype=torch.float32)
        if macro_edges.numel() != 4:
            raise ValueError("macro_edges_hz must define exactly three macro bands.")
        if not torch.all(macro_edges[1:] > macro_edges[:-1]):
            raise ValueError("macro_edges_hz must be strictly increasing.")
        nyquist = sample_rate / 2.0
        if macro_edges[0] < 0 or macro_edges[-1] > nyquist:
            raise ValueError(f"macro edges must remain within 0-{nyquist} Hz.")

        subband_counts = tuple(int(count) for count in subbands_per_macro)
        if len(subband_counts) != 3 or any(count <= 0 for count in subband_counts):
            raise ValueError("subbands_per_macro must contain three positive counts.")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.temporal_channels = int(temporal_channels)
        self.epsilon = float(epsilon)
        self.num_macro_bands = 3
        self.subbands_per_macro = subband_counts
        self.num_subbands = sum(subband_counts)
        self.learnable_subband_weights = bool(learnable_subband_weights)
        self.gate_mode = gate_mode
        self.normalize_gate_inputs = bool(normalize_gate_inputs)
        self.conditional_subgates = bool(conditional_subgates)
        self.harmonic_context_enabled = bool(harmonic_context)
        self.hard_routing_top_k: int | None = None

        frequency_bins = torch.linspace(0.0, nyquist, n_fft // 2 + 1)
        subband_edges: list[float] = []
        subband_macro_indices: list[int] = []
        for macro_index, count in enumerate(subband_counts):
            local_edges = torch.linspace(
                macro_edges[macro_index],
                macro_edges[macro_index + 1],
                count + 1,
            )
            if not subband_edges:
                subband_edges.append(float(local_edges[0]))
            subband_edges.extend(float(edge) for edge in local_edges[1:])
            subband_macro_indices.extend([macro_index] * count)

        subband_edges_tensor = torch.tensor(subband_edges, dtype=torch.float32)
        subband_mask = self._build_frequency_mask(frequency_bins, subband_edges_tensor)
        macro_mask = self._build_frequency_mask(frequency_bins, macro_edges)

        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        self.register_buffer("frequency_bins_hz", frequency_bins)
        self.register_buffer("macro_edges_hz", macro_edges)
        self.register_buffer("subband_edges_hz", subband_edges_tensor)
        self.register_buffer("subband_macro_index", torch.tensor(subband_macro_indices))
        self.register_buffer("subband_mask", subband_mask)
        self.register_buffer("subband_support", subband_mask > 0)
        self.register_buffer("macro_mask", macro_mask)
        self.register_buffer(
            "second_harmonic_matrix",
            self._build_harmonic_matrix(subband_edges_tensor, ratio=2.0),
        )
        self.register_buffer(
            "third_harmonic_matrix",
            self._build_harmonic_matrix(subband_edges_tensor, ratio=3.0),
        )
        if self.learnable_subband_weights:
            initial_logits = torch.zeros_like(subband_mask)
            self.subband_weight_logits = nn.Parameter(initial_logits)
        else:
            self.register_parameter("subband_weight_logits", None)

        temporal_padding = temporal_kernel_size // 2
        self.shared_temporal_transform = nn.Sequential(
            nn.Conv1d(
                1,
                temporal_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
            ),
            nn.GELU(),
            nn.Conv1d(
                temporal_channels,
                temporal_channels,
                kernel_size=3,
                padding=1,
                groups=temporal_channels,
            ),
            nn.GELU(),
        )
        self.neighbor_context = nn.Conv2d(
            temporal_channels,
            temporal_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=temporal_channels,
            bias=False,
        )
        if self.harmonic_context_enabled:
            self.second_harmonic_scale = nn.Parameter(torch.zeros(temporal_channels))
            self.third_harmonic_scale = nn.Parameter(torch.zeros(temporal_channels))
        else:
            self.register_parameter("second_harmonic_scale", None)
            self.register_parameter("third_harmonic_scale", None)
        self.parent_gate = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        child_channels = 4 if self.conditional_subgates else 3
        self.child_gate = nn.Conv1d(child_channels, 1, kernel_size=1)

    def set_hard_routing_top_k(self, top_k: int | None) -> None:
        """Enable inference-only top-k macro routing without changing parameters."""
        if top_k is not None and not 1 <= top_k <= self.num_macro_bands:
            raise ValueError(
                f"hard routing top_k must be between 1 and {self.num_macro_bands}."
            )
        if top_k is not None and self.gate_mode != "macro":
            raise ValueError("hard routing currently requires gate_mode='macro'.")
        self.hard_routing_top_k = top_k

    @staticmethod
    def _build_frequency_mask(
        frequency_bins: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        masks = []
        for index, (low_hz, high_hz) in enumerate(
            zip(edges[:-1], edges[1:], strict=True)
        ):
            is_last = index == edges.numel() - 2
            mask = (frequency_bins >= low_hz) & (
                frequency_bins <= high_hz if is_last else frequency_bins < high_hz
            )
            if not torch.any(mask):
                raise ValueError(
                    f"frequency band {float(low_hz)}-{float(high_hz)} Hz "
                    "contains no STFT bins; increase n_fft or widen the band."
                )
            masks.append(mask.float() / mask.sum())
        return torch.stack(masks)

    @staticmethod
    def _build_harmonic_matrix(
        subband_edges: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Map non-adjacent fundamentals to bands containing ratio * center."""
        if subband_edges.ndim != 1 or subband_edges.numel() < 3:
            raise ValueError("subband_edges must define at least two subbands.")
        if ratio <= 1:
            raise ValueError("harmonic ratio must be greater than one.")
        centers = 0.5 * (subband_edges[:-1] + subband_edges[1:])
        num_subbands = centers.numel()
        matrix = torch.zeros(num_subbands, num_subbands, dtype=torch.float32)
        for source, center in enumerate(centers):
            harmonic_frequency = ratio * center
            target_matches = torch.nonzero(
                (harmonic_frequency >= subband_edges[:-1])
                & (harmonic_frequency < subband_edges[1:]),
                as_tuple=False,
            ).flatten()
            if target_matches.numel() != 1:
                continue
            target = int(target_matches.item())
            if abs(target - source) <= 1:
                continue
            matrix[target, source] = 1.0
        row_mass = matrix.sum(dim=1, keepdim=True)
        return torch.where(row_mass > 0, matrix / row_mass.clamp_min(1.0), matrix)

    @staticmethod
    def _absolute_delta(values: torch.Tensor) -> torch.Tensor:
        return F.pad(torch.diff(values, dim=-1).abs(), (1, 0))

    def _normalize_over_time(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
            self.epsilon**0.5
        )
        return (values - mean) / std

    def _gate_inputs(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self._absolute_delta(values)
        if not self.normalize_gate_inputs:
            return values, delta
        return self._normalize_over_time(values), self._normalize_over_time(delta)

    @staticmethod
    def activation_signature(
        joint_gates: torch.Tensor,
        subband_log_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Return [mean gate, peak gate, gated energy, duration, gated delta]."""
        if joint_gates.shape != subband_log_energy.shape or joint_gates.ndim != 3:
            raise ValueError(
                "joint_gates and subband_log_energy must share [batch, subband, time]."
            )
        delta = HierarchicalSpectralFrontend._absolute_delta(subband_log_energy)
        gate_mass = joint_gates.sum(dim=-1).clamp_min(1e-8)
        return torch.stack(
            (
                joint_gates.mean(dim=-1),
                joint_gates.amax(dim=-1),
                (joint_gates * subband_log_energy).sum(dim=-1) / gate_mass,
                (joint_gates >= 0.5).float().mean(dim=-1),
                (joint_gates * delta).sum(dim=-1) / gate_mass,
            ),
            dim=-1,
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        batch_size, channels, samples = waveform.shape
        spectrum = torch.stft(
            waveform.reshape(batch_size * channels, samples),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        return power.reshape(batch_size, channels, power.shape[-2], power.shape[-1]).mean(
            dim=1
        )

    def effective_subband_weights(self) -> torch.Tensor:
        if self.subband_weight_logits is None:
            return self.subband_mask
        masked_logits = self.subband_weight_logits.masked_fill(
            ~self.subband_support, torch.finfo(self.subband_weight_logits.dtype).min
        )
        return torch.softmax(masked_logits, dim=1)

    def forward(
        self,
        waveform: torch.Tensor,
        masked_subbands: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if waveform.ndim != 3 or waveform.shape[1] < 1:
            raise ValueError("waveform must have shape [batch, channels, time].")
        if waveform.shape[-1] <= self.n_fft // 2:
            raise ValueError("waveform is too short for the configured STFT.")

        power = self._spectrogram(waveform)
        log_power = torch.log(power.clamp_min(self.epsilon))
        subband_weights = self.effective_subband_weights()
        subband_energy = torch.einsum("sf,bft->bst", subband_weights, power)
        macro_energy = torch.einsum("mf,bft->bmt", self.macro_mask, power)
        subband_log_energy = torch.log(subband_energy.clamp_min(self.epsilon))
        macro_log_energy = torch.log(macro_energy.clamp_min(self.epsilon))
        subband_gate_energy, subband_delta = self._gate_inputs(subband_log_energy)
        macro_gate_energy, macro_delta = self._gate_inputs(macro_log_energy)

        batch_size, _, frames = subband_log_energy.shape
        parent_descriptors = torch.stack((macro_gate_energy, macro_delta), dim=2)
        parent_descriptors = parent_descriptors.reshape(
            batch_size * self.num_macro_bands,
            2,
            frames,
        )
        macro_gates = torch.sigmoid(self.parent_gate(parent_descriptors)).reshape(
            batch_size,
            self.num_macro_bands,
            frames,
        )
        if self.hard_routing_top_k is not None:
            if self.training:
                raise RuntimeError("hard routing is inference-only; call eval() first.")
            selected_macro_indices = macro_gates.mean(dim=-1).topk(
                self.hard_routing_top_k, dim=1
            ).indices
            active_macro_mask = torch.zeros(
                batch_size,
                self.num_macro_bands,
                dtype=torch.bool,
                device=macro_gates.device,
            )
            active_macro_mask.scatter_(1, selected_macro_indices, True)
        else:
            active_macro_mask = torch.ones(
                batch_size,
                self.num_macro_bands,
                dtype=torch.bool,
                device=macro_gates.device,
            )
        active_subband_mask = active_macro_mask[:, self.subband_macro_index]

        if masked_subbands is not None:
            if masked_subbands.shape != (batch_size, self.num_subbands):
                raise ValueError("masked_subbands must have shape [batch, subband].")
            masked_subbands = masked_subbands.to(
                device=subband_log_energy.device, dtype=torch.bool
            )
            visible = ~masked_subbands
            visible_count = visible.sum(dim=1, keepdim=True).clamp_min(1)
            replacement = (
                subband_log_energy * visible.unsqueeze(-1)
            ).sum(dim=1, keepdim=True) / visible_count.unsqueeze(-1)
            temporal_log_energy = torch.where(
                masked_subbands.unsqueeze(-1), replacement, subband_log_energy
            )
        else:
            temporal_log_energy = subband_log_energy

        temporal_input = temporal_log_energy.reshape(
            batch_size * self.num_subbands, 1, frames
        )
        if self.hard_routing_top_k is None:
            transformed = self.shared_temporal_transform(temporal_input)
        else:
            flat_active = active_subband_mask.reshape(-1)
            active_transformed = self.shared_temporal_transform(
                temporal_input[flat_active]
            )
            transformed = active_transformed.new_zeros(
                batch_size * self.num_subbands,
                self.temporal_channels,
                frames,
            )
            transformed[flat_active] = active_transformed
        transformed = transformed.reshape(
            batch_size, self.num_subbands, self.temporal_channels, frames
        )
        transformed = transformed.permute(0, 2, 1, 3)
        context = self.neighbor_context(transformed)
        if self.harmonic_context_enabled:
            second_harmonic = torch.einsum(
                "ij,bcjt->bcit", self.second_harmonic_matrix, transformed
            )
            third_harmonic = torch.einsum(
                "ij,bcjt->bcit", self.third_harmonic_matrix, transformed
            )
            harmonic_features = (
                self.second_harmonic_scale[None, :, None, None] * second_harmonic
                + self.third_harmonic_scale[None, :, None, None] * third_harmonic
            )
        else:
            harmonic_features = torch.zeros_like(transformed)
        contextual_features = (transformed + context + harmonic_features) * active_subband_mask[
            :, None, :, None
        ]

        context_summary = contextual_features.mean(dim=1)
        if self.normalize_gate_inputs:
            context_summary = self._normalize_over_time(context_summary)
        parent_for_subband = macro_gates[:, self.subband_macro_index, :]
        if self.hard_routing_top_k is not None and self.gate_mode == "macro":
            subband_gates = torch.ones_like(parent_for_subband)
        else:
            child_energy, child_delta = self._gate_inputs(temporal_log_energy)
            child_parts = [child_energy, child_delta, context_summary]
            if self.conditional_subgates:
                child_parts.append(parent_for_subband)
            child_descriptors = torch.stack(child_parts, dim=2).reshape(
                batch_size * self.num_subbands,
                len(child_parts),
                frames,
            )
            subband_gates = torch.sigmoid(self.child_gate(child_descriptors)).reshape(
                batch_size,
                self.num_subbands,
                frames,
            )
        if self.gate_mode == "none":
            joint_gates = torch.ones_like(subband_gates)
        elif self.gate_mode == "macro":
            joint_gates = parent_for_subband
        elif self.gate_mode == "subband":
            joint_gates = subband_gates
        else:
            joint_gates = parent_for_subband * subband_gates
        if self.hard_routing_top_k is not None:
            joint_gates = joint_gates * active_subband_mask.unsqueeze(-1)
        features = contextual_features * joint_gates.unsqueeze(1)
        signature = self.activation_signature(joint_gates, subband_log_energy)

        return {
            "features": features,
            "spectrogram": log_power,
            "macro_energy": macro_energy,
            "subband_energy": subband_energy,
            "subband_log_energy": subband_log_energy,
            "subband_weights": subband_weights,
            "macro_gates": macro_gates,
            "subband_gates": subband_gates,
            "joint_gates": joint_gates,
            "active_macro_mask": active_macro_mask,
            "active_subband_mask": active_subband_mask,
            "active_macro_fraction": active_macro_mask.float().mean(),
            "active_subband_fraction": active_subband_mask.float().mean(),
            "harmonic_features": harmonic_features,
            "activation_signature": signature,
            "frequency_bins_hz": self.frequency_bins_hz,
            "macro_edges_hz": self.macro_edges_hz,
            "subband_edges_hz": self.subband_edges_hz,
            "subband_macro_index": self.subband_macro_index,
        }
