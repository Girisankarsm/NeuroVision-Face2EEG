"""Synthetic sanity-check dataset generator.

Plants a **known** relationship between facial features and EEG band power
so the pipeline can be validated end-to-end. Every output produced from this
data MUST be labeled ``SYNTHETIC SANITY CHECK``.

Planted relationship
--------------------
- Alpha power is a linear function of EAR (eye aspect ratio) plus noise.
- Theta power weakly correlates with blink rate.
- Other bands are pure noise.
- EEG waveform is synthesized with the corresponding spectral profile.

This ensures that a well-functioning pipeline should:
1. Recover a significant Pearson r for alpha on real (unshuffled) data.
2. Fail the permutation test on shuffled data (p > 0.05).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


SYNTHETIC_LABEL = "SYNTHETIC SANITY CHECK"


def generate_synthetic_dataset(
    n_subjects: int = 8,
    windows_per_subject: int = 60,
    sequence_length: int = 64,
    feature_dim: int = 28,
    eeg_samples: int = 128,
    eeg_sample_rate: float = 256.0,
    seed: int = 42,
    output_path: str | Path | None = None,
    alpha_ear_weight: float = 2.0,
    noise_level: float = 0.3,
    subject_baselines: bool = True,
) -> dict[str, np.ndarray]:
    """Generate a synthetic dataset with a planted EAR→alpha relationship.

    Parameters
    ----------
    n_subjects : int
        Number of synthetic subjects.
    windows_per_subject : int
        Number of time windows per subject.
    sequence_length : int
        Number of facial feature frames per window.
    feature_dim : int
        Dimensionality of per-frame facial features (compact mode).
    eeg_samples : int
        Number of EEG samples per window.
    eeg_sample_rate : float
        EEG sampling rate in Hz.
    seed : int
        Random seed for reproducibility.
    output_path : Path, optional
        If given, save the dataset as .npz here.
    alpha_ear_weight : float
        Strength of the planted EAR→alpha correlation.
    noise_level : float
        Standard deviation of noise added to targets.

    Returns
    -------
    dict with keys: 'facial', 'eeg', 'subjects', 'log_alpha_true', 'metadata'
    """
    rng = np.random.RandomState(seed)

    n_total = n_subjects * windows_per_subject
    facial_all = []
    eeg_all = []
    subjects_all = []
    log_alpha_true = []

    for subj_idx in range(n_subjects):
        subj_id = f"synth_{subj_idx:03d}"

        # Per-subject baseline offset (simulates individual differences)
        if subject_baselines:
            subj_alpha_baseline = rng.uniform(0.5, 1.5)
            subj_ear_baseline = rng.uniform(0.25, 0.35)
        else:
            subj_alpha_baseline = 1.0
            subj_ear_baseline = 0.3

        for w in range(windows_per_subject):
            # Generate facial features: shape (sequence_length, feature_dim)
            # Col 0: ear_left, Col 1: ear_right, Col 2: ear_avg
            # Col 3: blink_event, Col 4: blink_rate
            facial = rng.randn(sequence_length, feature_dim).astype(np.float32) * 0.1

            # Plant EAR signal
            ear_signal = subj_ear_baseline + rng.randn(sequence_length).astype(np.float32) * 0.05
            facial[:, 0] = ear_signal + rng.randn(sequence_length).astype(np.float32) * 0.01  # ear_left
            facial[:, 1] = ear_signal + rng.randn(sequence_length).astype(np.float32) * 0.01  # ear_right
            facial[:, 2] = ear_signal  # ear_avg

            # Plant blink rate (col 4)
            blink_rate = rng.uniform(10, 25)
            facial[:, 4] = blink_rate + rng.randn(sequence_length).astype(np.float32) * 1.0

            # Compute mean EAR for this window (the planted predictor)
            mean_ear = float(ear_signal.mean())

            # Alpha power: linear function of mean_ear
            log_alpha = subj_alpha_baseline + alpha_ear_weight * mean_ear + rng.randn() * noise_level
            log_alpha_true.append(log_alpha)

            # Theta: weak blink rate correlation
            log_theta = 0.3 * blink_rate / 20.0 + rng.randn() * noise_level * 2

            # Other bands: pure noise
            log_delta = rng.randn() * noise_level
            log_beta = rng.randn() * noise_level
            log_gamma = rng.randn() * noise_level

            # Synthesize EEG waveform with appropriate spectral content
            eeg_window = _synthesize_eeg_from_band_powers(
                {
                    "delta": 10 ** log_delta,
                    "theta": 10 ** log_theta,
                    "alpha": 10 ** log_alpha,
                    "beta": 10 ** log_beta,
                    "gamma": 10 ** log_gamma,
                },
                n_samples=eeg_samples,
                sample_rate=eeg_sample_rate,
                rng=rng,
            )

            facial_all.append(facial)
            eeg_all.append(eeg_window)
            subjects_all.append(subj_id)

    result = {
        "facial": np.array(facial_all, dtype=np.float32),
        "eeg": np.array(eeg_all, dtype=np.float32),
        "subjects": np.array(subjects_all),
        "log_alpha_true": np.array(log_alpha_true, dtype=np.float32),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            facial=result["facial"],
            eeg=result["eeg"],
            subjects=result["subjects"],
        )

    return result


def _synthesize_eeg_from_band_powers(
    band_powers: dict[str, float],
    n_samples: int,
    sample_rate: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Create a synthetic EEG signal whose band powers match the targets.

    Uses band-filtered white noise scaled to achieve desired power in each band.
    """
    BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 100.0),
    }

    signal = np.zeros(n_samples, dtype=np.float64)
    nyquist = sample_rate / 2.0

    for band_name, (low, high) in BANDS.items():
        power = max(band_powers.get(band_name, 0.0), 1e-12)
        amplitude = np.sqrt(power)

        high_filt = min(high, nyquist - 1.0)
        if low >= high_filt:
            continue

        noise = rng.randn(n_samples + 100)  # extra samples for filter transient
        try:
            sos = butter(2, [low, high_filt], btype="bandpass", fs=sample_rate, output="sos")
            filtered = sosfiltfilt(sos, noise)
            filtered = filtered[:n_samples]
            # Normalize and scale
            std = filtered.std() + 1e-12
            signal += amplitude * (filtered / std)
        except Exception:
            # If filtering fails (e.g., bad frequency range), add raw noise
            signal += amplitude * rng.randn(n_samples) * 0.01

    return signal.astype(np.float32)
