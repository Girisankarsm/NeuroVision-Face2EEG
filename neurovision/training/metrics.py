from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neurovision.preprocessing.eeg import band_power


def regression_metrics(pred: np.ndarray, target: np.ndarray, sample_rate: float = 256.0) -> dict[str, float]:
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred_flat.size < 2:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "r2": 0.0,
            "spectral_correlation": 0.0,
            "spectral_distance": 0.0,
        }

    # Pearson correlation
    corr_mat = np.corrcoef(pred_flat, target_flat)
    pearson_corr = float(corr_mat[0, 1]) if np.isfinite(corr_mat[0, 1]) else 0.0

    # Spearman rank correlation
    spearman_res = spearmanr(pred_flat, target_flat)
    spearman_corr = float(spearman_res.statistic) if hasattr(spearman_res, "statistic") else float(spearman_res[0])
    if not np.isfinite(spearman_corr):
        spearman_corr = 0.0

    # Spectral analysis
    nperseg = min(len(pred_flat), 512)
    freqs_p, psd_p = welch(pred_flat, fs=sample_rate, nperseg=nperseg)
    freqs_t, psd_t = welch(target_flat, fs=sample_rate, nperseg=nperseg)
    n = min(len(psd_p), len(psd_t))
    spec_mat = np.corrcoef(psd_p[:n], psd_t[:n])
    spectral_corr = float(spec_mat[0, 1]) if n > 1 and np.isfinite(spec_mat[0, 1]) else 0.0
    spectral_dist = float(np.linalg.norm(psd_p[:n] - psd_t[:n]))

    # Band power comparison
    pred_band = band_power(pred_flat, sample_rate)
    target_band = band_power(target_flat, sample_rate)

    metrics = {
        "mae": float(mean_absolute_error(target_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "r2": float(r2_score(target_flat, pred_flat)),
        "spectral_correlation": spectral_corr,
        "spectral_distance": spectral_dist,
    }
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        metrics[f"{band}_power_mae"] = float(abs(pred_band.get(band, 0.0) - target_band.get(band, 0.0)))

    return metrics
