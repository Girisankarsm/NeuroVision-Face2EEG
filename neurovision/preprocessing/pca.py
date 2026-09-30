"""PCA dimensionality reduction — fit on training data only.

This module wraps scikit-learn PCA with safeguards against data leakage:
- ``fit()`` and ``fit_transform()`` should ONLY be called on training data.
- ``transform()`` is used on validation / test data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


class FittedPCA:
    """Thin wrapper around sklearn PCA with save/load and leakage guard."""

    def __init__(self, n_components: int | float = 0.95, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._pca: PCA | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_components_(self) -> int:
        if not self._fitted:
            raise RuntimeError("PCA has not been fitted yet.")
        return self._pca.n_components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCA has not been fitted yet.")
        return self._pca.explained_variance_ratio_

    def fit(self, X_train: np.ndarray) -> "FittedPCA":
        """Fit PCA on training data ONLY."""
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(X_train)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into PCA space (safe for train or test)."""
        if not self._fitted:
            raise RuntimeError("PCA must be fit() on training data before transform().")
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed training data."""
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, path: str | Path) -> None:
        """Save PCA parameters to disk."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "random_state": self.random_state,
                "pca": self._pca,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "FittedPCA":
        """Load a previously saved PCA."""
        import pickle
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls(n_components=data["n_components"], random_state=data["random_state"])
        obj._pca = data["pca"]
        obj._fitted = data["fitted"]
        return obj

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging."""
        if not self._fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_components": int(self._pca.n_components_),
            "total_variance_explained": float(self._pca.explained_variance_ratio_.sum()),
            "top_5_variance": self._pca.explained_variance_ratio_[:5].tolist(),
        }
