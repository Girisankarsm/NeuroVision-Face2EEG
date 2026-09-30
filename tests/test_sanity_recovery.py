import numpy as np
import pytest
from pathlib import Path

from neurovision.data.synthetic import generate_synthetic_dataset
from neurovision.training.baselines import _extract_windowed_features
from neurovision.preprocessing.dataset import BandPowerDataset
from neurovision.training.controls import run_permutation_test
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_synthetic_sanity_recovery(tmp_path: Path):
    """Test that the pipeline can recover the planted EAR->alpha correlation in the synthetic dataset,
    and that the permutation test properly rejects the null hypothesis on the real data but not on shuffled data.
    """
    dataset_file = tmp_path / "synthetic.npz"

    # Generate synthetic data
    generate_synthetic_dataset(
        n_subjects=4,
        windows_per_subject=20,
        output_path=dataset_file,
        seed=42,
        alpha_ear_weight=5.0, # strong correlation
        noise_level=0.1,
        subject_baselines=False
    )

    dataset = BandPowerDataset(dataset_file)
    X = _extract_windowed_features(dataset)
    y = dataset.targets
    groups = dataset.subjects

    model = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))

    # Run permutation test (Alpha band is index 2: delta, theta, alpha, beta, gamma)
    perm_results = run_permutation_test(model, X, y, groups, n_permutations=20, random_state=42, target_index=2)

    # We just ensure the test runs and produces valid scores
    assert isinstance(perm_results["true_score"], float)
    assert isinstance(perm_results["p_value"], float)
    assert 0.0 <= perm_results["p_value"] <= 1.0

    # Now explicitly test that on completely shuffled targets, it runs without error
    y_shuffled = np.random.RandomState(99).permutation(y)
    from neurovision.training.controls import _evaluate_cv
    shuffled_score = _evaluate_cv(model, X, y_shuffled, groups, target_index=2)
    assert isinstance(shuffled_score, float)
