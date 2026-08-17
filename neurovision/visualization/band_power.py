from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_band_power(powers: dict[str, float], output: str | Path, title: str = "Predicted Band Power") -> None:
    names = ["delta", "theta", "alpha", "beta", "gamma"]
    values = [powers.get(name, 0.0) for name in names]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar([name.title() for name in names], values, color="#4da3ff")
    ax.set_title(title)
    ax.set_ylabel("Power")
    fig.savefig(output, dpi=180)
    plt.close(fig)
