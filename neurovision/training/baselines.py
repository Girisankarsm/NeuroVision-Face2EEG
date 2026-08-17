from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from neurovision.training.metrics import regression_metrics


def evaluate_baselines(dataset_path: Path | str, folds: int = 5, sample_rate: float = 256.0) -> pd.DataFrame:
    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=sample_rate)
    subjects = dataset.subjects
    unique_subjects = np.unique(subjects)
    actual_folds = min(folds, len(unique_subjects))
    if actual_folds < 2:
        raise ValueError(f"Need at least 2 subjects for grouped baseline evaluation; found {len(unique_subjects)}.")

    x = dataset.facial.reshape(len(dataset), -1)
    y = dataset.eeg
    splitter = GroupKFold(n_splits=actual_folds)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=subjects), start=1):
        # 1. Ridge Linear Baseline
        ridge = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(Ridge(alpha=100.0, random_state=fold)),
        )
        ridge.fit(x[train_idx], y[train_idx])
        pred_ridge = ridge.predict(x[test_idx])
        rows.append({
            "model": "Ridge Linear",
            "fold": fold,
            **regression_metrics(pred_ridge, y[test_idx], sample_rate=sample_rate),
        })

        # 2. Random Forest Baseline
        rf = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=fold,
                    n_jobs=-1,
                )
            ),
        )
        rf.fit(x[train_idx], y[train_idx])
        pred_rf = rf.predict(x[test_idx])
        rows.append({
            "model": "Random Forest",
            "fold": fold,
            **regression_metrics(pred_rf, y[test_idx], sample_rate=sample_rate),
        })

    df = pd.DataFrame(rows)
    print("\n=== Baseline Models Evaluation (Grouped K-Fold) ===")
    print(df.groupby("model")[["mae", "rmse", "pearson_correlation", "r2"]].mean())
    return df
