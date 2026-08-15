from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _load_audio(content: bytes, target_rate: int, duration_seconds: float) -> np.ndarray:
    import io

    sample_rate, audio = wavfile.read(io.BytesIO(content))
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    if int(sample_rate) != target_rate:
        audio = resample_poly(audio, target_rate, int(sample_rate)).astype(np.float32)
    target_samples = int(target_rate * duration_seconds)
    if audio.size >= target_samples:
        start = (audio.size - target_samples) // 2
        audio = audio[start : start + target_samples]
    else:
        padded = np.zeros(target_samples, dtype=np.float32)
        start = (target_samples - audio.size) // 2
        padded[start : start + audio.size] = audio
        audio = padded
    return audio


def diagnose_wav(
    content: bytes,
    checkpoint_path: Path,
    condition_id: str,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    from scripts.train_mimii_one_class import build_model, local_feature_map
    from src.anomaly.feature_memory import ConditionedFeatureMemory
    from src.anomaly.normal_profile import ConditionedNormalProfile

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    profile_payload = checkpoint["spectral_profile"]
    profile = ConditionedNormalProfile(
        condition_ids=profile_payload["condition_ids"],
        mean=torch.tensor(profile_payload["mean"]),
        std=torch.tensor(profile_payload["std"]),
        record_counts=torch.tensor(profile_payload["record_counts"]),
        fallback_mean=torch.tensor(profile_payload["fallback_mean"]),
        fallback_std=torch.tensor(profile_payload["fallback_std"]),
        epsilon=float(profile_payload["epsilon"]),
        metadata=profile_payload.get("metadata", {}),
    )
    model = build_model(
        "egfn",
        profile,
        learnable_subband_weights=bool(args.get("learnable_subband_weights", False)),
        gate_mode=args.get("gate_mode", "hierarchical"),
        normalize_gate_inputs=bool(args.get("normalize_gate_inputs", False)),
        conditional_subgates=bool(args.get("conditional_subgates", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    memory = ConditionedFeatureMemory.from_dict(checkpoint["feature_memory"]).to(device)

    frontend_signature = profile.metadata["frontend"]
    waveform = _load_audio(
        content,
        int(frontend_signature["sample_rate"]),
        float(args.get("duration", 2.0)),
    )
    tensor = torch.from_numpy(waveform).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        output = model(tensor, [condition_id], regularization_progress=1.0)
        memory_score = memory.score(
            local_feature_map(output, args.get("memory_representation", "encoder")),
            [condition_id],
        )["recording_memory_score"][0]
        gates = output["joint_gates"][0]
    score = float(memory_score.item())
    threshold = float(checkpoint["condition_thresholds"].get(condition_id, checkpoint["fallback_threshold"]))
    return {
        "status": "possible_failure" if score >= threshold else "no_failure_detected",
        "anomaly_score": score,
        "threshold": threshold,
        "condition_id": condition_id,
        "checkpoint": str(checkpoint_path),
        "evidence": {
            "primary_score": checkpoint.get("primary_score", "memory_score"),
            "known_condition": condition_id in memory.condition_ids,
            "mean_gate": float(gates.mean().item()),
            "active_gate_fraction": float((gates >= 0.5).float().mean().item()),
        },
    }


def diagnosis_as_text(diagnosis: dict[str, Any]) -> str:
    return json.dumps(diagnosis, ensure_ascii=False, indent=2)
