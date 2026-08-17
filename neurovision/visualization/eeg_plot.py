from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, sample_rate: float, output: str | Path) -> None:
    t = np.arange(len(actual)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, actual, label="Actual EEG", linewidth=1.2)
    ax.plot(t, predicted, label="AI-predicted EEG", linewidth=1.2)
    ax.set_title("Actual EEG vs AI-Predicted EEG")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
