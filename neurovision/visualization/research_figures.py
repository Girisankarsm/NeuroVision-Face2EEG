from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(results: pd.DataFrame, output: str | Path) -> None:
    required = {"model", "rmse", "mae", "pearson_correlation"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results table missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ["rmse", "mae", "pearson_correlation"]):
        ax.bar(results["model"], results[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
    fig.savefig(output, dpi=180)
    plt.close(fig)
