from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neurovision.preprocessing.eeg import band_power


def regression_metrics(pred: np.ndarray, target: np.ndarray, sample_rate: float = 256.0) -> dict[str, float]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    corr = np.corrcoef(pred.reshape(-1), target.reshape(-1))[0, 1] if pred.size > 1 else np.nan
    freqs_p, psd_p = welch(pred.reshape(-1), fs=sample_rate, nperseg=min(pred.size, 512))
    freqs_t, psd_t = welch(target.reshape(-1), fs=sample_rate, nperseg=min(target.size, 512))
    n = min(len(psd_p), len(psd_t))
    spectral_corr = np.corrcoef(psd_p[:n], psd_t[:n])[0, 1] if n > 1 else np.nan
    pred_band = band_power(pred.reshape(-1), sample_rate)
    target_band = band_power(target.reshape(-1), sample_rate)
    metrics = {
        "mae": float(mean_absolute_error(target.reshape(-1), pred.reshape(-1))),
        "rmse": float(np.sqrt(mean_squared_error(target.reshape(-1), pred.reshape(-1)))),
        "pearson_correlation": float(corr),
        "r2": float(r2_score(target.reshape(-1), pred.reshape(-1))),
        "spectral_correlation": float(spectral_corr),
        "spectral_distance": float(np.linalg.norm(psd_p[:n] - psd_t[:n])),
    }
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        metrics[f"{band}_power_mae"] = abs(pred_band[band] - target_band[band])
    return metrics
