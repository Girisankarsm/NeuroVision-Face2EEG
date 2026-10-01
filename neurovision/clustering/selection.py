"""Save reproducible model-selection curves for GMM and K-Means."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_selection_curves(folds: list[dict], output_path: str | Path, synthetic: bool = False) -> Path:
    gmm_bic: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    gmm_silhouette: dict[int, list[float]] = defaultdict(list)
    kmeans_silhouette: dict[int, list[float]] = defaultdict(list)
    for fold in folds:
        for key, value in fold["gmm_bic_curve"].items():
            k, covariance = key.split("_", 1)
            gmm_bic[covariance][int(k)].append(float(value))
        for key, value in fold["gmm_silhouette_curve"].items():
            gmm_silhouette[int(key)].append(float(value))
        for key, value in fold["kmeans_silhouette_curve"].items():
            kmeans_silhouette[int(key)].append(float(value))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for covariance, curve in sorted(gmm_bic.items()):
        ks = sorted(curve)
        axes[0].plot(ks, [np.mean(curve[k]) for k in ks], marker="o", label=f"GMM {covariance}")
    axes[0].set(title="GMM BIC by component count", xlabel="Components (k)", ylabel="Mean train BIC")
    axes[0].legend()

    for curve, label in ((gmm_silhouette, "GMM (BIC-selected covariance)"), (kmeans_silhouette, "K-Means")):
        ks = sorted(curve)
        if ks:
            axes[1].plot(ks, [np.mean(curve[k]) for k in ks], marker="o", label=label)
    axes[1].set(title="Training silhouette by component count", xlabel="Components (k)", ylabel="Mean train silhouette")
    axes[1].legend()
    title_prefix = "SYNTHETIC SANITY CHECK · " if synthetic else ""
    fig.suptitle(f"{title_prefix}Model selection (training folds only)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path