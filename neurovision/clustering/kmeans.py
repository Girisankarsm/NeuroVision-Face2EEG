"""K-Means baseline with training-only silhouette selection."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    def __init__(
        self,
        components: Sequence[int] = tuple(range(2, 9)),
        random_state: int = 42,
        n_init: int = 20,
    ) -> None:
        self.components = tuple(int(k) for k in components)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.model_: KMeans | None = None
        self.silhouette_curve_: dict[int, float] = {}
        self.silhouette_: float | None = None
        self.n_components_: int | None = None
        self.fitted_n_samples_: int | None = None

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("K-Means input must be a 2D array with at least three rows")
        if not np.isfinite(X).all():
            raise ValueError("K-Means input must contain only finite values")
        self.silhouette_curve_.clear()
        candidates = []
        for k in self.components:
            if k < 2 or k >= len(X):
                continue
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init).fit(X)
            if not 1 < len(np.unique(model.labels_)) < len(X):
                continue
            score = float(silhouette_score(X, model.labels_))
            self.silhouette_curve_[k] = score
            candidates.append((score, model))
        if not candidates:
            raise ValueError("No valid K-Means component counts for the training data")
        self.silhouette_, self.model_ = max(candidates, key=lambda item: item[0])
        self.n_components_ = int(self.model_.n_clusters)
        self.fitted_n_samples_ = len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("KMeansClusterer must be fitted before prediction")
        return self.model_.predict(np.asarray(X, dtype=np.float64))