"""SYNTHETIC SANITY CHECK tests for unsupervised facial-state clustering."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from neurovision.clustering.analysis import cluster_band_statistics, permutation_kw_test
from neurovision.clustering.features import extract_window_features, window_feature_names
from neurovision.clustering.gmm import GMMClusterer
from neurovision.clustering.kmeans import KMeansClusterer
from neurovision.clustering.runner import dataset_is_synthetic
from neurovision.clustering.stability import subject_bootstrap_indices
from neurovision.data.synthetic import generate_synthetic_dataset


def test_compact_window_feature_extraction_names_and_shape():
    facial = np.ones((4, 64, 28), dtype=np.float32)
    features, names = extract_window_features(facial)
    assert features.shape == (4, 144)
    assert names == window_feature_names()
    assert "blink_rate_mean" in names
    assert "ear_avg_std" in names
    assert "movement_energy_mean" in names


def test_window_extraction_rejects_noncompact_features():
    with pytest.raises(ValueError, match="compact facial features"):
        extract_window_features(np.ones((2, 64, 10), dtype=np.float32))


def test_synthetic_mixture_selection_is_deterministic():
    """SYNTHETIC SANITY CHECK: separated known facial states recover three components."""
    rng = np.random.RandomState(17)
    centers = np.array([[-5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    labels = np.repeat(np.arange(3), 40)
    X = centers[labels] + rng.normal(scale=0.35, size=(len(labels), 2))

    first = GMMClusterer(components=range(2, 5), random_state=21).fit(X)
    second = GMMClusterer(components=range(2, 5), random_state=21).fit(X)
    baseline = KMeansClusterer(components=range(2, 5), random_state=21).fit(X)

    assert first.n_components_ == 3
    assert adjusted_rand_score(labels, first.predict(X)) > 0.95
    assert np.array_equal(first.predict(X), second.predict(X))
    assert baseline.n_components_ == 3
    assert adjusted_rand_score(labels, baseline.predict(X)) > 0.95
    assert baseline.model_.n_init == 20


def test_held_out_cluster_statistics_and_permutation_are_reproducible():
    labels = np.repeat([0, 1, 2], 30)
    targets = np.zeros((len(labels), 5), dtype=np.float64)
    targets[:, 2] = np.repeat([-2.0, 0.0, 2.0], 30)
    targets[:, 1] = np.repeat([1.0, 0.0, -1.0], 30)
    targets[:, 3] = np.repeat([0.0, 1.0, -1.0], 30)
    stats = cluster_band_statistics(labels, targets)
    first = permutation_kw_test(labels, targets[:, 2], n_permutations=199, seed=7)
    second = permutation_kw_test(labels, targets[:, 2], n_permutations=199, seed=7)

    assert stats["alpha"]["p_adjusted_holm"] < 0.01
    assert stats["alpha"]["epsilon_squared"] > 0.5
    assert first == second
    assert first["p_value"] == 0.005


def test_permutation_is_neutral_when_heldout_assignment_has_one_cluster():
    result = permutation_kw_test(np.zeros(12, dtype=int), np.arange(12), n_permutations=25, seed=8)
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0
    assert result["null_distribution"] == [0.0] * 25


def test_subject_bootstrap_keeps_each_subject_whole():
    subjects = np.repeat(["s1", "s2", "s3"], 5)
    for indices in subject_bootstrap_indices(subjects, n_bootstrap=4, seed=12):
        sampled = subjects[indices]
        assert len(sampled) == 15
        assert all(np.count_nonzero(sampled == subject) % 5 == 0 for subject in np.unique(sampled))


def test_synthetic_archive_contains_machine_readable_provenance(tmp_path):
    archive_path = tmp_path / "small_dataset.npz"
    generate_synthetic_dataset(
        n_subjects=2, windows_per_subject=2, sequence_length=64,
        eeg_samples=128, seed=5, output_path=archive_path,
    )
    assert dataset_is_synthetic(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["synthetic"] is True
    assert metadata["label"] == "SYNTHETIC SANITY CHECK"
    assert metadata["seed"] == 5


def test_synthetic_three_state_alpha_recovery_and_shuffle_control():
    """SYNTHETIC SANITY CHECK: known facial states have planted alpha means."""
    rng = np.random.RandomState(91)
    state_ids = np.repeat([0, 1, 2], 60)
    centers = np.array([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    X = centers[state_ids] + rng.normal(scale=0.3, size=(len(state_ids), 2))
    alpha = np.array([-1.5, 0.0, 1.5])[state_ids] + rng.normal(scale=0.2, size=len(state_ids))
    target = np.zeros((len(alpha), 5), dtype=np.float64)
    target[:, 2] = alpha

    model = GMMClusterer(components=range(2, 5), random_state=19).fit(X)
    assigned = model.predict(X)
    recovered = cluster_band_statistics(assigned, target)
    shuffled_target = target.copy()
    shuffled_target[:, 2] = rng.permutation(alpha)
    shuffled = cluster_band_statistics(assigned, shuffled_target)
    permutation = permutation_kw_test(assigned, alpha, n_permutations=999, seed=31)

    assert adjusted_rand_score(state_ids, assigned) > 0.95
    assert recovered["alpha"]["p_value"] < 0.001
    assert shuffled["alpha"]["p_value"] > 0.05
    assert permutation["p_value"] < 0.01


def test_permutation_p_values_are_approximately_uniform_on_noise():
    labels = np.repeat([0, 1, 2], 24)
    p_values = []
    for seed in range(24):
        noise = np.random.RandomState(seed).normal(size=len(labels))
        p_values.append(permutation_kw_test(labels, noise, n_permutations=99, seed=seed)["p_value"])
    assert 0.2 < float(np.mean(p_values)) < 0.8


def test_cli_smoke_and_group_fold_leakage_metadata(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "clustering"
    result = subprocess.run(
        [
            sys.executable, str(repo / "main.py"), "--config", str(repo / "configs/config.yaml"),
            "cluster", "--dataset", str(repo / "synthetic_data.npz"),
            "--out", str(output), "--folds", "2", "--seed", "42",
            "--permutations", "1000", "--bootstrap", "1", "--save-model",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=58,
        check=True,
    )
    assert "SYNTHETIC SANITY CHECK" in result.stdout
    for filename in (
        "cluster_results.json", "cluster_table.csv", "selection_curves.png",
        "heldout_tsne.png", "cluster_profiles.png", "alpha_by_cluster.png",
        "cluster_timeline.png", "embedding_diagnostic.json",
    ):
        assert (output / filename).exists()

    results = json.loads((output / "cluster_results.json").read_text())
    assert results["synthetic"] is True
    assert len(results["aggregate"]["alpha_permutation"]["null_distribution"]) == 1000
    table = pd.read_csv(output / "cluster_table.csv")
    assert set(table["data_label"]) == {"SYNTHETIC SANITY CHECK"}
    assert {"p_alpha_holm", "p_theta_holm", "p_beta_holm", "ridge_cluster_probability_mae"} <= set(table.columns)
    diagnostic = json.loads((output / "embedding_diagnostic.json").read_text())
    assert diagnostic["label"] == "SYNTHETIC SANITY CHECK"
    artifact = joblib.load(tmp_path / "models/gmm.joblib")
    assert artifact["synthetic_training_data"] is True
    assert artifact["window_frames"] == 64
    for fold in results["folds"]:
        assert set(fold["train_subjects"]).isdisjoint(fold["heldout_subjects"])
        assert set(fold["normalizer_fit_subjects"]) == set(fold["train_subjects"])
        assert (
            fold["normalizer_fit_n_samples"]
            == fold["pca_fit_n_samples"]
            == fold["gmm_fit_n_samples"]
            == fold["kmeans_fit_n_samples"]
        )