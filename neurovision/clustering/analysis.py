"""Held-out cluster/EEG statistics and subject-level uncertainty summaries."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest, kruskal


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


BAND_COLUMNS = {"theta": 1, "alpha": 2, "beta": 3}


def epsilon_squared(statistic: float, n_groups: int, n_samples: int) -> float:
    if n_samples <= n_groups or n_groups < 2:
        return 0.0
    return float(max(0.0, (statistic - n_groups + 1) / (n_samples - n_groups)))


def cluster_band_statistics(
    cluster_labels: np.ndarray,
    z_scored_targets: np.ndarray,
    bands: Sequence[str] = ("alpha", "theta", "beta"),
) -> dict[str, dict[str, float | int]]:
    """Compute Kruskal-Wallis tests and epsilon-squared for held-out windows."""
    labels = np.asarray(cluster_labels)
    targets = np.asarray(z_scored_targets, dtype=np.float64)
    result: dict[str, dict[str, float | int]] = {}
    raw_p = []
    for band in bands:
        groups = [targets[labels == label, BAND_COLUMNS[band]] for label in np.unique(labels)]
        groups = [group for group in groups if len(group)]
        combined = np.concatenate(groups) if groups else np.asarray([], dtype=np.float64)
        if len(groups) < 2 or sum(map(len, groups)) <= len(groups) or np.ptp(combined) == 0.0:
            result[band] = {"statistic": 0.0, "p_value": 1.0, "epsilon_squared": 0.0, "n": int(len(labels))}
            raw_p.append(1.0)
            continue
        statistic, p_value = kruskal(*groups)
        effect = epsilon_squared(float(statistic), len(groups), sum(map(len, groups)))
        result[band] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "epsilon_squared": effect,
            "n": int(sum(map(len, groups))),
        }
        raw_p.append(float(p_value))
    adjusted = _holm_adjust(raw_p) if raw_p else []
    for band, p_adjusted in zip(bands, adjusted):
        result[band]["p_adjusted_holm"] = float(p_adjusted)
    return result


def permutation_kw_test(
    cluster_labels: np.ndarray,
    values: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    strata: np.ndarray | None = None,
) -> dict[str, object]:
    """Shuffle held-out labels while preserving their counts; return the null H distribution."""
    labels = np.asarray(cluster_labels)
    values = np.asarray(values, dtype=np.float64)
    groups = np.unique(labels)
    if len(values) != len(labels):
        raise ValueError("Permutation test requires aligned values and cluster labels")
    if len(groups) < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "null_distribution": [0.0] * n_permutations,
            "n_permutations": n_permutations,
        }
    if np.ptp(values) == 0.0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "null_distribution": [0.0] * n_permutations,
            "n_permutations": n_permutations,
        }
    observed = float(kruskal(*(values[labels == group] for group in groups)).statistic)
    rng = np.random.RandomState(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    strata = np.zeros(len(labels), dtype=np.int64) if strata is None else np.asarray(strata)
    if len(strata) != len(labels):
        raise ValueError("strata and cluster_labels must have the same length")
    stratum_rows = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    for index in range(n_permutations):
        shuffled = labels.copy()
        for rows in stratum_rows:
            shuffled[rows] = rng.permutation(labels[rows])
        null[index] = kruskal(*(values[shuffled == group] for group in groups)).statistic
    p_value = float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))
    return {"statistic": observed, "p_value": p_value, "null_distribution": null.tolist(), "n_permutations": n_permutations}


def bootstrap_mean_ci(
    values: Sequence[float],
    seed: int = 42,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_subjects": 0}
    rng = np.random.RandomState(seed)
    samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "ci_low": float(np.quantile(samples, tail)),
        "ci_high": float(np.quantile(samples, 1.0 - tail)),
        "n_subjects": int(len(data)),
    }


def subject_motion_alpha_check(
    subjects: np.ndarray,
    labels: np.ndarray,
    alpha: np.ndarray,
    component_means: Sequence[dict[str, float]],
    seed: int = 42,
) -> dict[str, object]:
    """Compare high- vs low-head-motion facial clusters within each held-out subject."""
    if len(component_means) < 2:
        return {"status": "not_enough_clusters", "subject_differences": {}}
    motion = np.asarray([item.get("head_motion_energy", 0.0) for item in component_means])
    low_cluster, high_cluster = int(np.argmin(motion)), int(np.argmax(motion))
    subject_differences: dict[str, float] = {}
    for subject in np.unique(subjects):
        mask = np.asarray(subjects) == subject
        low = np.asarray(alpha)[mask & (np.asarray(labels) == low_cluster)]
        high = np.asarray(alpha)[mask & (np.asarray(labels) == high_cluster)]
        if len(low) and len(high):
            subject_differences[str(subject)] = float(high.mean() - low.mean())
    differences = list(subject_differences.values())
    nonzero = [value for value in differences if value != 0.0]
    sign_p = float(binomtest(sum(value > 0 for value in nonzero), len(nonzero), 0.5).pvalue) if nonzero else 1.0
    return {
        "contrast": "high-motion minus low-motion facial state; alpha",
        "subject_differences": subject_differences,
        "sign_test_p_value": sign_p,
        **bootstrap_mean_ci(differences, seed=seed),
    }


def artifact_cluster_ids(component_means: Sequence[dict[str, float]], threshold: float = 0.5) -> list[int]:
    """Suggest clusters dominated by blink variability or head movement (normalized units)."""
    return [
        index for index, means in enumerate(component_means)
        if means.get("blink_rate", 0.0) > threshold
        or means.get("ear_std", 0.0) > threshold
        or means.get("head_motion_energy", 0.0) > threshold
    ]


def cluster_exclusion_statistics(
    labels: np.ndarray,
    targets: np.ndarray,
    excluded_clusters: Sequence[int],
) -> dict[str, object]:
    keep = ~np.isin(labels, list(excluded_clusters))
    remaining = np.unique(np.asarray(labels)[keep])
    if len(remaining) < 2:
        return {"excluded_clusters": list(excluded_clusters), "status": "fewer_than_two_clusters_remain"}
    return {
        "excluded_clusters": list(excluded_clusters),
        "status": "tested",
        "statistics": cluster_band_statistics(np.asarray(labels)[keep], np.asarray(targets)[keep]),
    }