from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurovision.config import load_config
from neurovision.preprocessing.dataset import BandPowerDataset
from neurovision.training.baselines import _extract_windowed_features, evaluate_baselines
from neurovision.training.controls import run_blink_ablation, run_permutation_test
from neurovision.training.cross_validation import run_cross_validation


def run_full_experiment(
    dataset_path: str | Path,
    output_dir: str | Path = "results",
    with_clustering: bool = False,
) -> None:
    """Run the complete pipeline: baselines, deep models, controls, and generate report."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/config.yaml")

    print(f"=== Starting Full Experiment on {dataset_path.name} ===")

    # 1. Evaluate Baselines
    print("\n--- Running Baselines ---")
    df_baselines = evaluate_baselines(dataset_path, folds=5, sample_rate=config.get("eeg_sample_rate", 256.0))
    df_baselines.to_csv(output_dir / "baselines.csv", index=False)

    # 2. Evaluate Models (Temporal CNN & Transformer)
    print("\n--- Running Deep Models (Temporal CNN) ---")
    df_tcnn = run_cross_validation(config, dataset_path, model_name="temporal_cnn", folds=5, epochs=10, output_dir=output_dir)

    # 3. Controls (Permutation & Ablation on a Ridge model for simplicity and speed in the experiment pipeline)
    print("\n--- Running Controls (Permutation Test & Blink Ablation) ---")
    dataset = BandPowerDataset(dataset_path)
    X = _extract_windowed_features(dataset)
    y = dataset.targets
    groups = dataset.subjects

    model = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=100.0)))

    perm_results = run_permutation_test(model, X, y, groups, n_permutations=200)
    ablation_results = run_blink_ablation(model, X, y, groups)

    with (output_dir / "controls.json").open("w") as f:
        json.dump({
            "permutation_test": perm_results,
            "blink_ablation": ablation_results
        }, f, indent=2)

    print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    print(f"Blink Ablation Drop in R2: {ablation_results['drop']:.4f}")

    if with_clustering:
        from neurovision.clustering.runner import run_clustering

        print("\n--- Running Unsupervised Facial-State Clustering ---")
        run_clustering(dataset_path, output_dir / "clustering")

    # We will let the app handle the visualizations from these saved CSVs/JSONs.
    print(f"\n=== Experiment Complete! Results saved to {output_dir} ===")
