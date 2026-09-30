from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from neurovision.preprocessing.dataset import BandPowerDataset
from neurovision.preprocessing.facial import compute_windowed_stats


def _extract_windowed_features(dataset: BandPowerDataset) -> np.ndarray:
    """Convert sequence features to windowed stats for classical models."""
    n_samples = len(dataset)
    # Using the first window to determine the size of the stats vector
    dummy_stats = compute_windowed_stats(dataset.facial[0])
    stats_dim = len(dummy_stats)

    X = np.zeros((n_samples, stats_dim), dtype=np.float32)
    for i in range(n_samples):
        X[i] = compute_windowed_stats(dataset.facial[i])
    return X


def evaluate_baselines(dataset_path: Path | str, folds: int = 5, sample_rate: float = 256.0) -> pd.DataFrame:
    """Evaluate ordered baselines on the dataset."""
    dataset = BandPowerDataset(dataset_path, sample_rate=sample_rate)
    subjects = dataset.subjects
    unique_subjects = np.unique(subjects)
    actual_folds = min(folds, len(unique_subjects))
    if actual_folds < 2:
        raise ValueError(f"Need at least 2 subjects for grouped baseline evaluation; found {len(unique_subjects)}.")

    X = _extract_windowed_features(dataset)
    y = dataset.targets # log band power

    # If X doesn't have the expected columns (e.g. blink rate is col 4 in mean),
    # we just use a fallback index for blink rate.
    blink_rate_col = min(4, X.shape[1] - 1)
    X_blink_only = X[:, blink_rate_col:blink_rate_col+1]

    models: dict[str, tuple[BaseEstimator, np.ndarray]] = {
        "Training Mean": (DummyRegressor(strategy="mean"), X),
        "Ridge (Blink Only)": (make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=10.0))), X_blink_only),
        "Ridge (Compact)": (make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=100.0))), X),
        "SVR": (make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(kernel="rbf", C=1.0))), X),
        "Gradient Boosting": (MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, max_depth=3)), X),
    }

    splitter = GroupKFold(n_splits=actual_folds)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=subjects), start=1):
        for name, (model, model_X) in models.items():
            model.fit(model_X[train_idx], y[train_idx])
            preds = model.predict(model_X[test_idx])

            # Simple metrics on log target
            mae = float(np.mean(np.abs(y[test_idx] - preds)))
            rmse = float(np.sqrt(np.mean((y[test_idx] - preds)**2)))

            # R2 overall
            from sklearn.metrics import r2_score
            r2 = float(np.mean([r2_score(y[test_idx, i], preds[:, i]) for i in range(y.shape[1])]))

            rows.append({
                "model": name,
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            })

    df = pd.DataFrame(rows)
    print("\n=== Baseline Models Evaluation (Grouped K-Fold) ===")
    print(df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("r2"))
    return df
