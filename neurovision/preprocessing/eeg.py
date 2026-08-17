from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, welch
from scipy.stats import entropy


@dataclass(frozen=True)
class EEGPreprocessingConfig:
    sample_rate: float = 256.0
    bandpass_low: float = 0.5
    bandpass_high: float = 100.0
    notch_hz: float = 50.0
    notch_q: float = 30.0


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

_trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))


def preprocess_eeg(signal: np.ndarray, cfg: EEGPreprocessingConfig) -> np.ndarray:
    eeg = np.asarray(signal, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = eeg[None, :]
    high = min(cfg.bandpass_high, cfg.sample_rate / 2.0 - 1.0)
    sos = butter(4, [cfg.bandpass_low, high], btype="bandpass", fs=cfg.sample_rate, output="sos")
    filtered = sosfiltfilt(sos, eeg, axis=-1)
    if cfg.notch_hz < cfg.sample_rate / 2.0:
        b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.sample_rate)
        from scipy.signal import filtfilt

        filtered = filtfilt(b, a, filtered, axis=-1)
    mean = filtered.mean(axis=-1, keepdims=True)
    std = filtered.std(axis=-1, keepdims=True) + 1e-8
    return ((filtered - mean) / std).astype(np.float32)


def band_power(signal: np.ndarray, sample_rate: float, bands: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    bands = bands or BANDS
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 512))
    powers: dict[str, float] = {}
    for name, (low, high) in bands.items():
        high = min(high, sample_rate / 2.0)
        mask = (freqs >= low) & (freqs < high)
        powers[name] = float(_trapz_fn(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    return powers


def eeg_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 512))
    psd_sum = float(psd.sum()) + 1e-12
    powers = band_power(x, sample_rate)
    return {
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "variance": float(np.var(x)),
        "dominant_frequency": float(freqs[np.argmax(psd)]),
        "spectral_entropy": float(entropy(psd / psd_sum)),
        **{f"{name}_power": value for name, value in powers.items()},
        **{f"{name}_relative_power": value / (sum(powers.values()) + 1e-12) for name, value in powers.items()},
    }
