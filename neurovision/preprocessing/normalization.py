"""Per-subject normalization utilities.

All normalizers are fit on training data only and then applied to both
train and test splits, preventing data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubjectNormalizer:
    """Per-subject z-scoring for EEG band power and facial features.

    Usage
    -----
    1. Fit on training subjects:
       >>> norm = SubjectNormalizer()
       >>> norm.fit(subjects_train, features_train, targets_train)

    2. Transform train and test data:
       >>> X_train, y_train = norm.transform(subjects_train, features_train, targets_train)
       >>> X_test, y_test = norm.transform(subjects_test, features_test, targets_test)

    For test subjects that were **not** seen during fit, the global mean/std
    computed across all training subjects is used as fallback.
    """

    feature_stats: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    target_stats: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    global_feature_mean: np.ndarray | None = None
    global_feature_std: np.ndarray | None = None
    global_target_mean: np.ndarray | None = None
    global_target_std: np.ndarray | None = None
    _eps: float = 1e-8

    def fit(
        self,
        subjects: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> "SubjectNormalizer":
        """Fit per-subject statistics from training data only."""
        subjects = np.asarray(subjects)
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        for subj in np.unique(subjects):
            mask = subjects == subj
            self.feature_stats[str(subj)] = {
                "mean": features[mask].mean(axis=0),
                "std": features[mask].std(axis=0) + self._eps,
            }
            self.target_stats[str(subj)] = {
                "mean": targets[mask].mean(axis=0),
                "std": targets[mask].std(axis=0) + self._eps,
            }

        # Global fallback for unseen subjects
        self.global_feature_mean = features.mean(axis=0)
        self.global_feature_std = features.std(axis=0) + self._eps
        self.global_target_mean = targets.mean(axis=0)
        self.global_target_std = targets.std(axis=0) + self._eps

        return self

    def transform(
        self,
        subjects: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply per-subject z-scoring. Uses global stats for unseen subjects."""
        subjects = np.asarray(subjects)
        features = np.asarray(features, dtype=np.float64).copy()
        targets = np.asarray(targets, dtype=np.float64).copy()

        for subj in np.unique(subjects):
            mask = subjects == subj
            key = str(subj)
            if key in self.feature_stats:
                fm, fs = self.feature_stats[key]["mean"], self.feature_stats[key]["std"]
                tm, ts = self.target_stats[key]["mean"], self.target_stats[key]["std"]
            else:
                # Unseen subject → use global training stats
                fm, fs = self.global_feature_mean, self.global_feature_std
                tm, ts = self.global_target_mean, self.global_target_std

            features[mask] = (features[mask] - fm) / fs
            targets[mask] = (targets[mask] - tm) / ts

        return features.astype(np.float32), targets.astype(np.float32)

    def inverse_transform_targets(
        self,
        subjects: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Reverse the z-scoring on targets (for evaluation)."""
        subjects = np.asarray(subjects)
        targets = np.asarray(targets, dtype=np.float64).copy()

        for subj in np.unique(subjects):
            mask = subjects == subj
            key = str(subj)
            if key in self.target_stats:
                tm, ts = self.target_stats[key]["mean"], self.target_stats[key]["std"]
            else:
                tm, ts = self.global_target_mean, self.global_target_std
            targets[mask] = targets[mask] * ts + tm

        return targets.astype(np.float32)


class GlobalNormalizer:
    """Simple global z-scoring, fit on training data only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._eps = 1e-8

    def fit(self, X: np.ndarray) -> "GlobalNormalizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + self._eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
