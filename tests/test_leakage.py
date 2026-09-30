import numpy as np
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit

from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from pathlib import Path


def test_subject_leakage_in_splits(tmp_path: Path):
    num_windows = 40
    seq_len = 8
    feat_dim = 28
    eeg_samples = 128

    facial = np.random.randn(num_windows, seq_len, feat_dim).astype(np.float32)
    eeg = np.random.randn(num_windows, eeg_samples).astype(np.float32)
    subjects = np.array(["subj_1"] * 10 + ["subj_2"] * 10 + ["subj_3"] * 10 + ["subj_4"] * 10)

    dataset_file = tmp_path / "test_dataset.npz"
    np.savez(dataset_file, facial=facial, eeg=eeg, subjects=subjects)

    dataset = SynchronizedWindowDataset(dataset_file)
    groups = dataset.subjects

    # Test GroupShuffleSplit (used in train.py)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])

    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects) == 3
    assert len(val_subjects) == 1

    # Test GroupKFold (used in cross_validation.py)
    kf = GroupKFold(n_splits=4)
    for train_idx, val_idx in kf.split(np.arange(len(dataset)), groups=groups):
        train_subjects = set(groups[train_idx])
        val_subjects = set(groups[val_idx])
        assert len(train_subjects.intersection(val_subjects)) == 0
