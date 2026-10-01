"""Label-invariant cluster agreement and subject bootstrap sampling."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def mean_pairwise_ari(assignments: list[np.ndarray]) -> float:
    if len(assignments) < 2:
        return float("nan")
    scores = [
        adjusted_rand_score(assignments[left], assignments[right])
        for left in range(len(assignments))
        for right in range(left + 1, len(assignments))
    ]
    return float(np.mean(scores))


def subject_bootstrap_indices(
    subjects: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[np.ndarray]:
    """Resample whole training subjects with replacement and return row indices."""
    subjects = np.asarray(subjects).astype(str)
    unique_subjects = np.unique(subjects)
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_bootstrap):
        selected = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        samples.append(np.concatenate([np.flatnonzero(subjects == subject) for subject in selected]))
    return samples