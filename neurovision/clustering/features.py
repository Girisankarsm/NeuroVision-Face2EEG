"""Compact per-window facial features and train-only subject normalization."""
from __future__ import annotations

import numpy as np

from neurovision.preprocessing.facial import COMPACT_FEATURE_DIM, COMPACT_FEATURE_NAMES, compute_windowed_stats
from neurovision.preprocessing.normalization import SubjectNormalizer


def window_feature_names() -> list[str]:
    names = list(COMPACT_FEATURE_NAMES)
    return (
        [f"{name}_mean" for name in names]
        + [f"{name}_std" for name in names]
        + [f"{name}_min" for name in names]
        + [f"{name}_max" for name in names]
        + [f"{name}_energy" for name in names]
        + ["ear_avg_spectral_power", "head_pitch_spectral_power", "head_yaw_spectral_power", "head_roll_spectral_power"]
    )


def extract_window_features(facial: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Summarize each (frames, compact-features) window using the shared pipeline stats."""
    facial = np.asarray(facial, dtype=np.float32)
    if facial.ndim != 3:
        raise ValueError("facial must have shape (windows, frames, features)")
    if facial.shape[2] != COMPACT_FEATURE_DIM:
        raise ValueError(f"clustering requires {COMPACT_FEATURE_DIM} compact facial features per frame")
    if facial.shape[1] < 2:
        raise ValueError("each facial window must contain at least two frames")
    rows = [compute_windowed_stats(window) for window in facial]
    return np.asarray(rows, dtype=np.float32), window_feature_names()


class SubjectFeatureNormalizer:
    """Apply SubjectNormalizer's train-subject statistics to facial features only."""

    def __init__(self) -> None:
        self.normalizer = SubjectNormalizer()
        self.fitted_n_samples_: int | None = None
        self.fitted_subjects_: tuple[str, ...] = ()

    def fit(
        self,
        X: np.ndarray,
        subjects: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> "SubjectFeatureNormalizer":
        X = np.asarray(X, dtype=np.float32)
        subjects = np.asarray(subjects).astype(str)
        if len(X) != len(subjects) or not len(X):
            raise ValueError("X and subjects must contain the same nonzero number of rows")
        if targets is None:
            targets = np.zeros((len(X), 1), dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if len(targets) != len(X):
            raise ValueError("targets and X must contain the same number of rows")
        self.normalizer.fit(subjects, X, targets)
        self.fitted_n_samples_ = len(X)
        self.fitted_subjects_ = tuple(sorted(np.unique(subjects).tolist()))
        return self

    def transform(
        self,
        X: np.ndarray,
        subjects: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.fitted_n_samples_ is None:
            raise RuntimeError("SubjectFeatureNormalizer must be fitted before transform")
        X = np.asarray(X, dtype=np.float32)
        subjects = np.asarray(subjects).astype(str)
        if len(X) != len(subjects):
            raise ValueError("X and subjects must contain the same number of rows")
        target_width = len(self.normalizer.global_target_mean)
        target_values = np.zeros((len(X), target_width), dtype=np.float32) if targets is None else np.asarray(targets, dtype=np.float32)
        transformed, _ = self.normalizer.transform(subjects, X, target_values)
        return transformed

    def transform_targets(self, targets: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        if self.fitted_n_samples_ is None:
            raise RuntimeError("SubjectFeatureNormalizer must be fitted before transform")
        targets = np.asarray(targets, dtype=np.float32)
        subjects = np.asarray(subjects).astype(str)
        if len(targets) != len(subjects):
            raise ValueError("targets and subjects must contain the same number of rows")
        dummy_features = np.zeros((len(targets), len(self.normalizer.global_feature_mean)), dtype=np.float32)
        _, transformed = self.normalizer.transform(subjects, dummy_features, targets)
        return transformed

    def fit_transform(
        self,
        X: np.ndarray,
        subjects: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(X, subjects, targets).transform(X, subjects, targets)


INTERPRETABLE_FEATURES = {
    "blink_rate": "blink_rate_mean",
    "ear_std": "ear_avg_std",
    "head_motion_energy": "movement_energy_mean",
    "mar": "mar_mean",
}