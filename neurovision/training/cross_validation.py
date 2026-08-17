from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from neurovision.preprocessing.dataset import SynchronizedWindowDataset


def grouped_fold_indices(dataset_path: Path, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset = SynchronizedWindowDataset(dataset_path)
    unique_subjects = np.unique(dataset.subjects)
    if len(unique_subjects) < folds:
        raise ValueError(f"Need at least {folds} subjects for GroupKFold; found {len(unique_subjects)}.")
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(np.arange(len(dataset)), groups=dataset.subjects))
