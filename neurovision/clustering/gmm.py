"""Train-only Gaussian-mixture selection by Bayesian information criterion."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class GMMClusterer:
    """Choose component count and covariance type by BIC on fit data only."""

    def __init__(
        self,
        components: Sequence[int] = tuple(range(2, 9)),
        covariance_types: Sequence[str] = ("diag", "full"),
        random_state: int = 42,
        n_init: int = 3,
    ) -> None:
        self.components = tuple(int(k) for k in components)
        self.covariance_types = tuple(covariance_types)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.model_: GaussianMixture | None = None
        self.bic_curve_: dict[str, float] = {}
        self.silhouette_curve_: dict[int, float] = {}
        self.n_components_: int | None = None
        self.covariance_type_: str | None = None
        self.bic_: float | None = None
        self.silhouette_: float | None = None
        self.fitted_n_samples_: int | None = None

    def fit(self, X: np.ndarray) -> "GMMClusterer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("GMM input must be a 2D array with at least three rows")
        if not np.isfinite(X).all():
            raise ValueError("GMM input must contain only finite values")
        self.bic_curve_.clear()
        self.silhouette_curve_.clear()
        candidates: list[tuple[float, GaussianMixture]] = []
        by_k: dict[int, tuple[float, np.ndarray]] = {}
        for k in self.components:
            if k < 2 or k >= len(X):
                continue
            for covariance_type in self.covariance_types:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    random_state=self.random_state,
                    n_init=self.n_init,
                ).fit(X)
                bic = float(model.bic(X))
                self.bic_curve_[f"{k}_{covariance_type}"] = bic
                candidates.append((bic, model))
                if k not in by_k or bic < by_k[k][0]:
                    by_k[k] = (bic, model.predict(X))
        if not candidates:
            raise ValueError("No valid GMM component counts for the training data")

        self.bic_, self.model_ = min(candidates, key=lambda item: item[0])
        self.n_components_ = int(self.model_.n_components)
        self.covariance_type_ = str(self.model_.covariance_type)
        self.silhouette_curve_ = {
            k: float(silhouette_score(X, labels))
            for k, (_, labels) in by_k.items()
            if 1 < len(np.unique(labels)) < len(X)
        }
        labels = self.model_.predict(X)
        self.silhouette_ = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
        self.fitted_n_samples_ = len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._require_model().predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._require_model().predict_proba(np.asarray(X, dtype=np.float64))

    def bic(self, X: np.ndarray) -> float:
        return float(self._require_model().bic(np.asarray(X, dtype=np.float64)))

    def interpretable_means(
        self,
        pca: object,
        feature_names: Sequence[str],
        feature_keys: dict[str, str],
    ) -> list[dict[str, float]]:
        """Map PCA-space component means back to normalized named facial features."""
        means = pca.inverse_transform(self._require_model().means_)
        indices = {name: index for index, name in enumerate(feature_names)}
        result = []
        for component in means:
            result.append({
                key: float(component[indices[name]])
                for key, name in feature_keys.items()
                if name in indices
            })
        return result

    def _require_model(self) -> GaussianMixture:
        if self.model_ is None:
            raise RuntimeError("GMMClusterer must be fitted before prediction")
        return self.model_