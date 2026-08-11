from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.io import wavfile

from src.anomaly import deep_svdd_loss, deep_svdd_scores, stabilize_center
from src.data import AnomalyAudioRecord, AnomalyWindowDataset
from src.models import Conv1DAnomalyEncoder


def test_deep_svdd_scores_and_center_stabilization() -> None:
    center = stabilize_center(torch.tensor([0.0, -0.01, 2.0]), epsilon=0.1)
    embedding = torch.stack((center, center + 1.0))

    assert center.tolist() == pytest.approx([0.1, -0.1, 2.0])
    assert deep_svdd_scores(embedding, center).tolist() == pytest.approx([0.0, 1.0])
    assert deep_svdd_loss(embedding, center).item() == pytest.approx(0.5)


def test_conv1d_encoder_shares_weights_across_audio_channels() -> None:
    model = Conv1DAnomalyEncoder(embedding_channels=8)
    mono = torch.randn(2, 1, 4_000)
    duplicated = mono.repeat(1, 3, 1)

    model.eval()
    mono_output = model(mono)
    duplicated_output = model(duplicated)

    assert duplicated_output["channel_embeddings"].shape == (2, 3, 8)
    assert torch.allclose(mono_output["embedding"], duplicated_output["embedding"], atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000


def test_window_dataset_preserves_channels_and_reads_only_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(".tmp") / "one_class_window_test.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(20, dtype=np.int16)
    audio = np.stack((samples, -samples), axis=1)
    wavfile.write(path, 10, audio)
    original_read = sf.read
    requested_frames: list[int] = []

    def tracked_read(*args, **kwargs):
        requested_frames.append(int(kwargs["frames"]))
        return original_read(*args, **kwargs)

    monkeypatch.setattr("src.data.anomaly_window_dataset.sf.read", tracked_read)
    record = AnomalyAudioRecord(
        path=path,
        dataset_name="test",
        machine_type="machine",
        machine_id="id_00",
        condition_id="machine/id_00/condition",
        group_id="recording",
        label="normal",
    )
    try:
        dataset = AnomalyWindowDataset(
            [record],
            target_sample_rate=10,
            duration_seconds=1.0,
            crop_mode="grid",
            evaluation_windows=2,
        )
        first = dataset[0]
        last = dataset[1]
    finally:
        path.unlink(missing_ok=True)

    assert first["waveform"].shape == (2, 10)
    assert requested_frames == [10, 10]
    assert first["waveform"][0, 0].item() == pytest.approx(0.0)
    assert last["waveform"][0, 0].item() == pytest.approx(10 / 32768)
    assert first["condition_id"] == "test/machine/id_00/condition"
