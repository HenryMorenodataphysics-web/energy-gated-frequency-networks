from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.signal_utils import numpy_fft_magnitude


def plot_waveform_and_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """Plot a waveform and its magnitude spectrum."""

    time = np.arange(signal.shape[-1]) / sample_rate
    freqs, magnitude = numpy_fft_magnitude(signal, sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(time, signal)
    axes[0].set_title(f"{title} - waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(freqs, magnitude)
    axes[1].set_title(f"{title} - spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
