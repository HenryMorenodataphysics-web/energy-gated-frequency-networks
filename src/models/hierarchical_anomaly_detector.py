from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anomaly import ConditionedNormalProfile, GateRegularizer, ProfileAnomalyScorer
from src.blocks import HierarchicalSpectralFrontend
from src.models.event_pooling import soft_event_pool


class CompactHierarchicalEncoder(nn.Module):
    """Encode gated spectral features and profile deviations with few parameters."""

    def __init__(
        self,
        frontend_channels: int,
        descriptor_channels: int = 2,
        embedding_channels: int = 8,
        event_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if frontend_channels <= 0 or descriptor_channels <= 0 or embedding_channels <= 0:
            raise ValueError("all encoder channel counts must be positive.")
        if event_temperature <= 0:
            raise ValueError("event_temperature must be positive.")
        self.event_temperature = float(event_temperature)
        input_channels = frontend_channels + descriptor_channels
        self.input_projection = nn.Conv2d(
            input_channels, embedding_channels, 1, bias=False
        )
        self.local_context = nn.Conv2d(
            embedding_channels,
            embedding_channels,
            kernel_size=(3, 5),
            padding=(1, 2),
            groups=embedding_channels,
            bias=False,
        )
        self.channel_mixing = nn.Conv2d(
            embedding_channels, embedding_channels, 1, bias=False
        )

    def forward(
        self,
        frontend_features: torch.Tensor,
        profile_z_scores: torch.Tensor,
        masked_subbands: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frontend_features.ndim != 4 or profile_z_scores.ndim != 4:
            raise ValueError("encoder inputs must have shape [batch, channel, subband, time].")
        z_features = profile_z_scores.permute(0, 2, 1, 3)
        if masked_subbands is not None:
            if masked_subbands.shape != (
                frontend_features.shape[0],
                frontend_features.shape[2],
            ):
                raise ValueError("masked_subbands must have shape [batch, subband].")
            z_features = z_features.masked_fill(
                masked_subbands[:, None, :, None], 0.0
            )
        if frontend_features.shape[0] != z_features.shape[0] or frontend_features.shape[2:] != z_features.shape[2:]:
            raise ValueError("frontend features and profile deviations must align.")

        projected = self.input_projection(torch.cat((frontend_features, z_features), dim=1))
        contextual = self.channel_mixing(F.gelu(self.local_context(projected)))
        embedding_map = F.gelu(projected + contextual)
        temporal_mean = embedding_map.mean(dim=-1)
        temporal_event = soft_event_pool(embedding_map, self.event_temperature)
        sustained_embedding = torch.cat(
            (temporal_mean.mean(dim=-1), temporal_mean.mean(dim=1)),
            dim=1,
        )
        event_embedding = torch.cat(
            (temporal_event.mean(dim=-1), temporal_event.mean(dim=1)),
            dim=1,
        )
        return {
            "embedding_map": embedding_map,
            "sustained_embedding": sustained_embedding,
            "event_embedding": event_embedding,
            "subband_embedding": torch.cat(
                (temporal_mean.mean(dim=1), temporal_event.mean(dim=1)),
                dim=1,
            ),
            "embedding": torch.cat((sustained_embedding, event_embedding), dim=1),
        }


class HierarchicalAnomalyDetector(nn.Module):
    """Connect the spectral frontend, fitted normal profile, score, and encoder."""

    def __init__(
        self,
        frontend: HierarchicalSpectralFrontend,
        normal_profile: ConditionedNormalProfile,
        embedding_channels: int = 8,
        supervised_anomaly_head: bool = False,
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
        self.reconstruction_head = nn.Conv2d(
            embedding_channels, 1, kernel_size=1
        )
        embedding_dimensions = 2 * (embedding_channels + frontend.num_subbands)
        self.anomaly_head = (
            nn.Linear(embedding_dimensions, 1)
            if supervised_anomaly_head
            else None
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
        masked_subbands: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        frontend_outputs = self.frontend(waveform, masked_subbands=masked_subbands)
        profile_outputs = self.normal_profile.standardize(
            frontend_outputs["subband_energy"],
            condition_ids,
        )
        score_outputs = self.scorer(
            profile_outputs["z_scores"],
            profile_outputs["known_condition"],
        )
        encoder_mask = masked_subbands
        active_subbands = frontend_outputs.get("active_subband_mask")
        if (
            active_subbands is not None
            and self.frontend.hard_routing_top_k is not None
        ):
            routing_mask = ~active_subbands
            encoder_mask = (
                routing_mask
                if encoder_mask is None
                else encoder_mask.to(dtype=torch.bool, device=routing_mask.device)
                | routing_mask
            )
        encoder_outputs = self.encoder(
            frontend_outputs["features"],
            profile_outputs["z_scores"],
            masked_subbands=encoder_mask,
        )
        reconstruction = self.reconstruction_head(
            encoder_outputs["embedding_map"]
        ).squeeze(1)
        z_features = profile_outputs["z_scores"].permute(0, 2, 1, 3)
        gate_features = frontend_outputs["joint_gates"].unsqueeze(1)
        gated_profile_features = torch.cat(
            (z_features, gate_features, z_features * gate_features),
            dim=1,
        )
        activation_signature_features = frontend_outputs[
            "activation_signature"
        ].permute(0, 2, 1).unsqueeze(-1)
        active_macro_gates = self.frontend.gate_mode in {"macro", "hierarchical"}
        active_subband_gates = self.frontend.gate_mode in {
            "subband",
            "hierarchical",
        }
        regularization_outputs = self.gate_regularizer(
            frontend_outputs["macro_gates"],
            frontend_outputs["subband_gates"],
            progress=regularization_progress,
            regularize_macro=active_macro_gates,
            regularize_subband=active_subband_gates,
        )
        outputs = {
            **frontend_outputs,
            **profile_outputs,
            **score_outputs,
            **encoder_outputs,
            "reconstructed_log_energy_z": reconstruction,
            "gated_profile_features": gated_profile_features,
            "activation_signature_features": activation_signature_features,
            **regularization_outputs,
        }
        if self.anomaly_head is not None:
            outputs["anomaly_logit"] = self.anomaly_head(
                encoder_outputs["embedding"]
            ).squeeze(1)
        return outputs
