from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from neurovision.training.metrics import regression_metrics


def evaluate_random_forest_baseline(dataset_path: Path, folds: int = 5, sample_rate: float = 256.0) -> pd.DataFrame:
    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=sample_rate)
    subjects = dataset.subjects
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < folds:
        raise ValueError(f"Need at least {folds} subjects for grouped baseline evaluation; found {len(unique_subjects)}.")

    x = dataset.facial.reshape(len(dataset), -1)
    y = dataset.eeg
    splitter = GroupKFold(n_splits=folds)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=subjects), start=1):
        model = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=fold,
                    n_jobs=-1,
                )
            ),
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        rows.append({"model": "Random Forest", "fold": fold, **regression_metrics(pred, y[test_idx], sample_rate=sample_rate)})
    return pd.DataFrame(rows)
