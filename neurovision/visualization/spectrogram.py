from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(signal: np.ndarray, sample_rate: float, output: str | Path, title: str = "AI-Predicted EEG Spectrogram") -> None:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.specgram(signal, Fs=sample_rate, NFFT=128, noverlap=96, cmap="magma")
    ax.set_ylim(0, min(100, sample_rate / 2))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
