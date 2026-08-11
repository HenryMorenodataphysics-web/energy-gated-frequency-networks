from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anomaly import ConditionedNormalProfile, GateRegularizer, ProfileAnomalyScorer
from src.blocks import HierarchicalSpectralFrontend


class CompactHierarchicalEncoder(nn.Module):
    """Encode gated spectral features and profile deviations with few parameters."""

    def __init__(
        self,
        frontend_channels: int,
        descriptor_channels: int = 2,
        embedding_channels: int = 8,
    ) -> None:
        super().__init__()
        if frontend_channels <= 0 or descriptor_channels <= 0 or embedding_channels <= 0:
            raise ValueError("all encoder channel counts must be positive.")
        input_channels = frontend_channels + descriptor_channels
        self.input_projection = nn.Conv2d(input_channels, embedding_channels, 1)
        self.local_context = nn.Conv2d(
            embedding_channels,
            embedding_channels,
            kernel_size=(3, 5),
            padding=(1, 2),
            groups=embedding_channels,
        )
        self.channel_mixing = nn.Conv2d(embedding_channels, embedding_channels, 1)

    def forward(
        self,
        frontend_features: torch.Tensor,
        profile_z_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frontend_features.ndim != 4 or profile_z_scores.ndim != 4:
            raise ValueError("encoder inputs must have shape [batch, channel, subband, time].")
        z_features = profile_z_scores.permute(0, 2, 1, 3)
        if frontend_features.shape[0] != z_features.shape[0] or frontend_features.shape[2:] != z_features.shape[2:]:
            raise ValueError("frontend features and profile deviations must align.")

        projected = self.input_projection(torch.cat((frontend_features, z_features), dim=1))
        contextual = self.channel_mixing(F.gelu(self.local_context(projected)))
        embedding_map = F.gelu(projected + contextual)
        return {
            "embedding_map": embedding_map,
            "embedding": embedding_map.mean(dim=(-2, -1)),
        }


class HierarchicalAnomalyDetector(nn.Module):
    """Connect the spectral frontend, fitted normal profile, score, and encoder."""

    def __init__(
        self,
        frontend: HierarchicalSpectralFrontend,
        normal_profile: ConditionedNormalProfile,
        embedding_channels: int = 8,
        scorer: ProfileAnomalyScorer | None = None,
        gate_regularizer: GateRegularizer | None = None,
    ) -> None:
        super().__init__()
        self._validate_profile_signature(frontend, normal_profile)
        self.frontend = frontend
        self.normal_profile = normal_profile
        self.scorer = scorer or ProfileAnomalyScorer()
        self.gate_regularizer = gate_regularizer or GateRegularizer()
        self.encoder = CompactHierarchicalEncoder(
            frontend_channels=frontend.temporal_channels,
            descriptor_channels=normal_profile.mean.shape[2],
            embedding_channels=embedding_channels,
        )

    @staticmethod
    def _validate_profile_signature(
        frontend: HierarchicalSpectralFrontend,
        profile: ConditionedNormalProfile,
    ) -> None:
        signature = profile.metadata.get("frontend")
        if not isinstance(signature, dict):
            raise ValueError("normal profile is missing its spectral frontend signature.")
        expected_scalars = {
            "sample_rate": frontend.sample_rate,
            "n_fft": frontend.n_fft,
            "hop_length": frontend.hop_length,
        }
        for name, expected in expected_scalars.items():
            if signature.get(name) != expected:
                raise ValueError(f"normal profile frontend mismatch: {name}.")
        if tuple(signature.get("subbands_per_macro", ())) != frontend.subbands_per_macro:
            raise ValueError("normal profile frontend mismatch: subbands_per_macro.")
        stored_edges = torch.as_tensor(signature.get("macro_edges_hz", ()))
        if stored_edges.shape != frontend.macro_edges_hz.shape or not torch.allclose(
            stored_edges.float(), frontend.macro_edges_hz.detach().cpu(), atol=1e-4, rtol=0
        ):
            raise ValueError("normal profile frontend mismatch: macro_edges_hz.")
        if profile.num_subbands != frontend.num_subbands:
            raise ValueError("normal profile frontend mismatch: num_subbands.")

    def forward(
        self,
        waveform: torch.Tensor,
        condition_ids: Sequence[str],
        regularization_progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        frontend_outputs = self.frontend(waveform)
        profile_outputs = self.normal_profile.standardize(
            frontend_outputs["subband_energy"],
            condition_ids,
        )
        score_outputs = self.scorer(
            profile_outputs["z_scores"],
            profile_outputs["known_condition"],
        )
        encoder_outputs = self.encoder(
            frontend_outputs["features"],
            profile_outputs["z_scores"],
        )
        regularization_outputs = self.gate_regularizer(
            frontend_outputs["macro_gates"],
            frontend_outputs["subband_gates"],
            progress=regularization_progress,
        )
        return {
            **frontend_outputs,
            **profile_outputs,
            **score_outputs,
            **encoder_outputs,
            **regularization_outputs,
        }
