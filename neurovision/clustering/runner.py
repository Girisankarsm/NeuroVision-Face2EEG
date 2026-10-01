"""Subject-independent unsupervised facial-state evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import wilcoxon

from neurovision.clustering.analysis import (
    artifact_cluster_ids,
    bootstrap_mean_ci,
    cluster_band_statistics,
    cluster_exclusion_statistics,
    permutation_kw_test,
    subject_motion_alpha_check,
)
from neurovision.clustering.features import (
    INTERPRETABLE_FEATURES,
    SubjectFeatureNormalizer,
    extract_window_features,
)
from neurovision.clustering.gmm import GMMClusterer
from neurovision.clustering.kmeans import KMeansClusterer
from neurovision.clustering.selection import save_selection_curves
from neurovision.clustering.stability import mean_pairwise_ari, subject_bootstrap_indices
from neurovision.preprocessing.dataset import BandPowerDataset


def suggest_cluster_names(component_means: list[dict[str, float]]) -> list[str]:
    """Suggest facial-state labels from normalized feature means; manual review required."""
    names = []
    for means in component_means:
        candidates = [
            (means.get("blink_rate", 0.0), "frequent blinking"),
            (means.get("ear_std", 0.0), "variable eye openness"),
            (means.get("head_motion_energy", 0.0), "high head motion"),
            (means.get("mar", 0.0), "increased mouth opening"),
        ]
        score, label = max(candidates)
        if score < 0.5:
            label = "low-motion facial state"
        names.append(f"{label} (suggestion; review manually)")
    return names


def dataset_is_synthetic(dataset_path: str | Path) -> bool:
    path = Path(dataset_path)
    if "synthetic" in path.stem.lower() or "synth" in path.stem.lower():
        return True
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            return False
        metadata = str(archive["metadata"].item()).lower()
        return "synthetic" in metadata or '"synthetic": true' in metadata


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    alpha_true, alpha_pred = y_true[:, 2], y_pred[:, 2]
    corr = float(np.corrcoef(alpha_true, alpha_pred)[0, 1]) if np.std(alpha_true) and np.std(alpha_pred) else float("nan")
    return {
        "mae": float(np.mean(np.abs(alpha_true - alpha_pred))),
        "r2": float(r2_score(alpha_true, alpha_pred)) if len(alpha_true) > 1 else float("nan"),
        "pearson_r": corr,
    }


def _subject_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    model_name: str,
    feature_set: str,
) -> list[dict[str, Any]]:
    rows = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        metrics = _target_metrics(y_true[mask], y_pred[mask])
        rows.append({"subject": str(subject), "model": model_name, "feature_set": feature_set, **metrics})
    return rows


def _regression_estimators(seed: int) -> dict[str, Any]:
    return {
        "ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "svr": MultiOutputRegressor(SVR(C=1.0, kernel="rbf")),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=seed)
        ),
    }


def _regression_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name in sorted({row["model"] for row in rows}):
        result[model_name] = {}
        for feature_set in ("compact", "cluster_probabilities"):
            subset = [row for row in rows if row["model"] == model_name and row["feature_set"] == feature_set]
            result[model_name][feature_set] = {
                metric: bootstrap_mean_ci([row[metric] for row in subset], seed=seed + offset)
                for offset, metric in enumerate(("pearson_r", "r2", "mae"))
            }
        compact = {row["subject"]: row for row in rows if row["model"] == model_name and row["feature_set"] == "compact"}
        augmented = {row["subject"]: row for row in rows if row["model"] == model_name and row["feature_set"] == "cluster_probabilities"}
        paired = {}
        for offset, metric in enumerate(("pearson_r", "r2", "mae")):
            differences = [
                augmented[subj][metric] - compact[subj][metric]
                for subj in sorted(compact.keys() & augmented.keys())
            ]
            finite = np.asarray(differences, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            p_value = float(wilcoxon(finite).pvalue) if len(finite) > 1 and np.any(finite != 0.0) else 1.0
            paired[metric] = {
                **bootstrap_mean_ci(differences, seed=seed + 20 + offset),
                "wilcoxon_p_value": p_value,
            }
        result[model_name]["paired_augmented_minus_compact"] = paired
    return result


def _pca(seed: int) -> PCA:
    return PCA(n_components=0.95, svd_solver="full", random_state=seed)


def _fit_gmm_pipeline(
    X: np.ndarray,
    subjects: np.ndarray,
    targets: np.ndarray,
    seed: int,
    columns: np.ndarray | None = None,
) -> tuple[SubjectFeatureNormalizer, PCA, np.ndarray, GMMClusterer]:
    columns = np.arange(X.shape[1]) if columns is None else columns
    normalizer = SubjectFeatureNormalizer().fit(X[:, columns], subjects, targets)
    normalized = normalizer.transform(X[:, columns], subjects)
    pca = _pca(seed).fit(normalized)
    projected = pca.transform(normalized)
    clusterer = GMMClusterer(random_state=seed).fit(projected)
    return normalizer, pca, projected, clusterer


def run_clustering(
    dataset_path: str | Path,
    output_dir: str | Path = "results/clustering",
    folds: int = 5,
    seed: int = 42,
    n_permutations: int = 1000,
    n_bootstrap: int = 5,
    save_model: bool = False,
    fps: float = 18.0,
    sample_rate: float = 256.0,
) -> dict[str, Any]:
    """Run subject-independent clustering and held-out analysis, saving reproducible outputs."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = dataset_is_synthetic(dataset_path)
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if folds < 2 or n_permutations < 1000 or n_bootstrap < 1 or seed < 0:
        raise ValueError("folds must be >=2, permutations >=1000, bootstrap >=1, and seed >=0")
    dataset = BandPowerDataset(dataset_path, sample_rate=sample_rate)
    X, feature_names = extract_window_features(dataset.facial)
    window_seconds = dataset.facial.shape[1] / fps
    if not 2.0 <= window_seconds <= 4.0:
        raise ValueError(f"Facial windows must span 2–4 seconds; found {window_seconds:.2f}s at {fps:g} FPS")
    targets = dataset.targets.astype(np.float32)
    subjects = dataset.subjects.astype(str)
    unique_subjects = np.unique(subjects)
    actual_folds = min(int(folds), len(unique_subjects))
    if actual_folds < 2:
        raise ValueError("Clustering evaluation requires at least two distinct subjects")
    splitter = GroupKFold(n_splits=actual_folds)
    fold_records: list[dict[str, Any]] = []
    oof_labels: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []
    oof_subjects: list[np.ndarray] = []
    oof_folds: list[np.ndarray] = []
    oof_indices: list[np.ndarray] = []
    blink_free_labels: list[np.ndarray] = []
    motion_differences: dict[str, float] = {}
    regression_rows: list[dict[str, Any]] = []
    oof_features: list[np.ndarray] = []
    gmm_fold_reference: list[np.ndarray] = []
    kmeans_fold_reference: list[np.ndarray] = []

    blink_columns = np.asarray([
        index for index, name in enumerate(feature_names)
        if name.startswith("blink_event") or name.startswith("blink_rate")
    ], dtype=int)
    no_blink_columns = np.setdiff1d(np.arange(X.shape[1]), blink_columns)

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, targets, groups=subjects)):
        train_subjects = np.unique(subjects[train_idx])
        test_subjects = np.unique(subjects[test_idx])
        if set(train_subjects) & set(test_subjects):
            raise AssertionError(f"Subject leakage in fold {fold_index + 1}")

        normalizer = SubjectFeatureNormalizer().fit(X[train_idx], subjects[train_idx], targets[train_idx])
        X_train = normalizer.transform(X[train_idx], subjects[train_idx])
        X_test = normalizer.transform(X[test_idx], subjects[test_idx])
        y_train = normalizer.transform_targets(targets[train_idx], subjects[train_idx])
        y_test = normalizer.transform_targets(targets[test_idx], subjects[test_idx])
        pca = _pca(seed + fold_index).fit(X_train)
        train_embedding, test_embedding = pca.transform(X_train), pca.transform(X_test)

        gmm = GMMClusterer(random_state=seed + fold_index).fit(train_embedding)
        kmeans = KMeansClusterer(random_state=seed + fold_index).fit(train_embedding)
        gmm_test, kmeans_test = gmm.predict(test_embedding), kmeans.predict(test_embedding)
        gmm_probability_train = gmm.predict_proba(train_embedding)
        gmm_probability_test = gmm.predict_proba(test_embedding)
        component_means = gmm.interpretable_means(pca, feature_names, INTERPRETABLE_FEATURES)
        cluster_names = suggest_cluster_names(component_means)
        fold_stats = cluster_band_statistics(gmm_test, y_test)
        excluded = artifact_cluster_ids(component_means)
        artifact_stats = cluster_exclusion_statistics(gmm_test, y_test, excluded)
        motion_check = subject_motion_alpha_check(
            subjects[test_idx], gmm_test, y_test[:, 2], component_means, seed=seed + fold_index
        )
        motion_differences.update(motion_check.get("subject_differences", {}))

        fold_offset = (fold_index + 1) * 16
        oof_labels.append(gmm_test + fold_offset)
        oof_targets.append(y_test)
        oof_features.append(X_test)
        oof_subjects.append(subjects[test_idx])
        oof_folds.append(np.full(len(test_idx), fold_index, dtype=int))
        oof_indices.append(test_idx)

        # Refit without blink event/rate statistics; every learned transform remains fold-local.
        blink_normalizer = SubjectFeatureNormalizer().fit(
            X[train_idx][:, no_blink_columns], subjects[train_idx], targets[train_idx]
        )
        X_train_no_blink = blink_normalizer.transform(X[train_idx][:, no_blink_columns], subjects[train_idx])
        X_test_no_blink = blink_normalizer.transform(X[test_idx][:, no_blink_columns], subjects[test_idx])
        blink_pca = _pca(seed + fold_index).fit(X_train_no_blink)
        blink_gmm = GMMClusterer(random_state=seed + fold_index).fit(blink_pca.transform(X_train_no_blink))
        blink_free_labels.append(blink_gmm.predict(blink_pca.transform(X_test_no_blink)) + fold_offset)

        # Compare compact-only regressors with the same fold's train-fitted GMM probabilities.
        for model_name, estimator in _regression_estimators(seed + fold_index).items():
            compact_model = make_pipeline(StandardScaler(), clone(estimator))
            compact_model.fit(X_train, y_train)
            compact_pred = compact_model.predict(X_test)
            augmented_model = make_pipeline(StandardScaler(), clone(estimator))
            augmented_model.fit(np.column_stack([X_train, gmm_probability_train]), y_train)
            augmented_pred = augmented_model.predict(np.column_stack([X_test, gmm_probability_test]))
            regression_rows.extend(_subject_metric_rows(y_test, compact_pred, subjects[test_idx], model_name, "compact"))
            regression_rows.extend(_subject_metric_rows(y_test, augmented_pred, subjects[test_idx], model_name, "cluster_probabilities"))

        # Bootstrap whole training subjects; refit normalizer, PCA and both clusterers per resample.
        gmm_boot_ari, kmeans_boot_ari = [], []
        for boot_index, local_indices in enumerate(
            subject_bootstrap_indices(subjects[train_idx], n_bootstrap, seed + 1000 + fold_index)
        ):
            boot_idx = train_idx[local_indices]
            boot_normalizer = SubjectFeatureNormalizer().fit(X[boot_idx], subjects[boot_idx], targets[boot_idx])
            boot_train = boot_normalizer.transform(X[boot_idx], subjects[boot_idx])
            boot_test = boot_normalizer.transform(X[test_idx], subjects[test_idx])
            boot_pca = _pca(seed + boot_index + fold_index * 100).fit(boot_train)
            boot_train_embedding = boot_pca.transform(boot_train)
            boot_test_embedding = boot_pca.transform(boot_test)
            boot_gmm = GaussianMixture(
                n_components=gmm.n_components_, covariance_type=gmm.covariance_type_,
                random_state=seed + boot_index, n_init=2,
            ).fit(boot_train_embedding)
            boot_kmeans = KMeansClusterer(
                components=(kmeans.n_components_,), random_state=seed + boot_index, n_init=20
            ).fit(boot_train_embedding)
            gmm_boot_ari.append(adjusted_rand_score(gmm_test, boot_gmm.predict(boot_test_embedding)))
            kmeans_boot_ari.append(adjusted_rand_score(kmeans_test, boot_kmeans.predict(boot_test_embedding)))

        # Apply each fold estimator to the same reference windows for label-invariant cross-fold ARI.
        X_reference = normalizer.transform(X, subjects)
        reference_embedding = pca.transform(X_reference)
        gmm_fold_reference.append(gmm.predict(reference_embedding))
        kmeans_fold_reference.append(kmeans.predict(reference_embedding))

        fold_records.append({
            "fold": fold_index + 1,
            "train_subjects": train_subjects.tolist(),
            "heldout_subjects": test_subjects.tolist(),
            "normalizer_fit_subjects": list(normalizer.fitted_subjects_),
            "normalizer_fit_n_samples": normalizer.fitted_n_samples_,
            "pca_fit_n_samples": int(pca.n_samples_),
            "gmm_fit_n_samples": gmm.fitted_n_samples_,
            "kmeans_fit_n_samples": kmeans.fitted_n_samples_,
            "gmm_components": gmm.n_components_,
            "gmm_covariance_type": gmm.covariance_type_,
            "gmm_bic": gmm.bic_,
            "gmm_silhouette": gmm.silhouette_,
            "gmm_bic_curve": gmm.bic_curve_,
            "gmm_silhouette_curve": gmm.silhouette_curve_,
            "kmeans_components": kmeans.n_components_,
            "kmeans_silhouette": kmeans.silhouette_,
            "kmeans_silhouette_curve": kmeans.silhouette_curve_,
            "gmm_kmeans_ari_heldout": adjusted_rand_score(gmm_test, kmeans_test),
            "gmm_bootstrap_ari": float(np.mean(gmm_boot_ari)) if gmm_boot_ari else None,
            "kmeans_bootstrap_ari": float(np.mean(kmeans_boot_ari)) if kmeans_boot_ari else None,
            "component_means_normalized": component_means,
            "cluster_name_suggestions": cluster_names,
            "cluster_eeg_statistics": fold_stats,
            "artifact_exclusion": artifact_stats,
            "motion_alpha_check": motion_check,
        })

    labels = np.concatenate(oof_labels)
    target_oof = np.concatenate(oof_targets)
    subjects_oof = np.concatenate(oof_subjects)
    folds_oof = np.concatenate(oof_folds)
    indices_oof = np.concatenate(oof_indices)
    no_blink_oof = np.concatenate(blink_free_labels)
    aggregate_stats = cluster_band_statistics(labels, target_oof)
    permutation_strata = np.asarray([f"{fold}:{subject}" for fold, subject in zip(folds_oof, subjects_oof)])
    permutation = permutation_kw_test(
        labels, target_oof[:, 2], n_permutations=n_permutations, seed=seed, strata=permutation_strata
    )
    motion_values = list(motion_differences.values())
    from scipy.stats import binomtest
    nonzero_motion = [value for value in motion_values if value != 0.0]
    motion_summary = {
        "contrast": "high-motion minus low-motion facial state; alpha",
        "subject_differences": motion_differences,
        "sign_test_p_value": float(binomtest(sum(value > 0 for value in nonzero_motion), len(nonzero_motion), 0.5).pvalue)
        if nonzero_motion else 1.0,
        **bootstrap_mean_ci(motion_values, seed=seed + 1),
    }
    artifact_summaries = []
    for fold in fold_records:
        exclusion = fold["artifact_exclusion"]
        exclusion_alpha = exclusion.get("statistics", {}).get("alpha", {})
        artifact_summaries.append({
            "fold": fold["fold"],
            **exclusion,
            "alpha_effect_survives_holm_0_05": exclusion_alpha.get("p_adjusted_holm", 1.0) < 0.05,
        })
    blink_free_stats = cluster_band_statistics(no_blink_oof, target_oof)
    blink_free_alpha_survives = blink_free_stats["alpha"].get("p_adjusted_holm", 1.0) < 0.05
    gmm_cross_fold_ari = mean_pairwise_ari(gmm_fold_reference)
    kmeans_cross_fold_ari = mean_pairwise_ari(kmeans_fold_reference)

    regression = _regression_summary(regression_rows, seed)
    table_rows = []
    for fold in fold_records:
        row = {
            "data_label": "SYNTHETIC SANITY CHECK" if synthetic else "REAL DATA",
            "synthetic": synthetic,
            "fold": fold["fold"], "gmm_k": fold["gmm_components"],
            "covariance_type": fold["gmm_covariance_type"], "bic": fold["gmm_bic"],
            "gmm_silhouette": fold["gmm_silhouette"], "kmeans_k": fold["kmeans_components"],
            "kmeans_silhouette": fold["kmeans_silhouette"],
            "facial_state_suggestions": " | ".join(fold["cluster_name_suggestions"]),
            "gmm_kmeans_ari": fold["gmm_kmeans_ari_heldout"],
            "gmm_bootstrap_ari": fold["gmm_bootstrap_ari"],
            "kmeans_bootstrap_ari": fold["kmeans_bootstrap_ari"],
            "kw_alpha": fold["cluster_eeg_statistics"]["alpha"]["statistic"],
            "p_alpha_holm": fold["cluster_eeg_statistics"]["alpha"].get("p_adjusted_holm", 1.0),
            "epsilon_squared_alpha": fold["cluster_eeg_statistics"]["alpha"]["epsilon_squared"],
            "alpha_permutation_p": permutation["p_value"],
        }
        for band in ("theta", "beta"):
            row[f"kw_{band}"] = fold["cluster_eeg_statistics"][band]["statistic"]
            row[f"p_{band}_holm"] = fold["cluster_eeg_statistics"][band].get("p_adjusted_holm", 1.0)
            row[f"epsilon_squared_{band}"] = fold["cluster_eeg_statistics"][band]["epsilon_squared"]
        for model_name in regression:
            for metric in ("pearson_r", "r2", "mae"):
                row[f"{model_name}_compact_{metric}"] = regression[model_name]["compact"][metric]["mean"]
                row[f"{model_name}_cluster_probability_{metric}"] = regression[model_name]["cluster_probabilities"][metric]["mean"]
                row[f"{model_name}_paired_{metric}_delta"] = regression[model_name]["paired_augmented_minus_compact"][metric]["mean"]
        table_rows.append(row)

    results: dict[str, Any] = {
        "label": "SYNTHETIC SANITY CHECK" if synthetic else None,
        "synthetic": synthetic,
        "methods": {
            "folds": actual_folds, "splitter": "GroupKFold by subject", "seed": seed,
            "window_features": len(feature_names), "window_frames": int(dataset.facial.shape[1]),
            "fps": float(fps),
            "estimated_window_seconds": float(dataset.facial.shape[1] / fps),
            "pca_variance": 0.95, "gmm_selection": "BIC; k=2..8; diag/full",
            "kmeans_selection": "training silhouette; k=2..8; n_init=20",
            "normalization": "SubjectNormalizer fit on training subjects; held-out subjects use training-global statistics",
            "artifact_cluster_threshold_normalized_units": 0.5,
            "n_permutations": n_permutations, "n_subject_bootstrap": n_bootstrap,
        },
        "folds": fold_records,
        "aggregate": {
            "cluster_eeg_statistics": aggregate_stats,
            "alpha_permutation": permutation,
            "per_subject_motion_alpha": motion_summary,
            "artifact_cluster_exclusion": artifact_summaries,
            "blink_features_removed_statistics": blink_free_stats,
            "artifact_sensitivity": {
                "alpha_survives_blink_feature_removal": blink_free_alpha_survives,
                "alpha_survives_excluding_dominated_clusters_by_fold": [
                    item["alpha_effect_survives_holm_0_05"] for item in artifact_summaries
                ],
                "interpretation": "A surviving effect does not prove neural origin; blink, EMG, and other confounds remain possible.",
            },
            "cross_fold_stability": {
                "gmm_mean_pairwise_ari": gmm_cross_fold_ari,
                "kmeans_mean_pairwise_ari": kmeans_cross_fold_ari,
                "gmm_vs_kmeans_heldout_ari_mean": float(np.mean([f["gmm_kmeans_ari_heldout"] for f in fold_records])),
            },
            "regression_feature_comparison": regression,
            "interpretation": "Window-level Kruskal-Wallis tests do not model repeated windows within subjects; use the subject-stratified permutation as a pairing control. Clusters describe facial patterns only, and any surviving association does not prove neural origin or exclude blink/EMG artifacts.",
        },
    }
    results = _json_value(results)
    with (output_dir / "cluster_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    pd.DataFrame(table_rows).to_csv(output_dir / "cluster_table.csv", index=False)

    save_selection_curves(fold_records, output_dir / "selection_curves.png", synthetic=synthetic)
    _save_analysis_figures(
        output_dir, np.concatenate(oof_features), labels, subjects_oof, target_oof[:, 2], indices_oof,
        fold_records, permutation, synthetic, seed,
    )

    if save_model:
        final_normalizer, final_pca, _, final_gmm = _fit_gmm_pipeline(
            X, subjects, targets, seed
        )
        final_means = final_gmm.interpretable_means(final_pca, feature_names, INTERPRETABLE_FEATURES)
        artifact_path = Path("models/gmm.joblib")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": final_normalizer,
            "pca": final_pca,
            "gmm": final_gmm.model_,
            "cluster_names": suggest_cluster_names(final_means),
            "component_means_normalized": final_means,
            "feature_names": feature_names,
            "window_frames": int(dataset.facial.shape[1]),
            "fps": float(fps),
            "seed": seed,
            "synthetic_training_data": synthetic,
        }, artifact_path)
        results["saved_model"] = str(artifact_path)
        with (output_dir / "cluster_results.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_value(results), stream, indent=2, allow_nan=False)

    print("SYNTHETIC SANITY CHECK" if synthetic else "REAL DATA ANALYSIS")
    print(f"Saved clustering output to {output_dir}")
    print(f"Held-out alpha cluster permutation p-value: {permutation['p_value']:.6g}")
    print("Cluster associations are exploratory and do not establish neural origin.")
    return results


def _save_analysis_figures(
    output_dir: Path,
    heldout_features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    alpha: np.ndarray,
    original_indices: np.ndarray,
    folds: list[dict],
    permutation: dict[str, Any],
    synthetic: bool,
    seed: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "SYNTHETIC SANITY CHECK · " if synthetic else ""
    rng = np.random.RandomState(seed)
    sample_size = min(len(labels), 1800)
    sample = rng.choice(len(labels), size=sample_size, replace=False) if sample_size < len(labels) else np.arange(len(labels))
    perplexity = min(30.0, max(2.0, (len(sample) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="random", learning_rate="auto")
    embedding = tsne.fit_transform(heldout_features[sample])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = ((labels[sample], "Facial cluster"), (subjects[sample], "Subject"), (alpha[sample], "z-scored log alpha"))
    for axis, (color, title) in zip(axes, panels):
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=pd.Categorical(color).codes if np.asarray(color).dtype.kind in "USO" else color,
                               cmap="tab20" if np.asarray(color).dtype.kind in "USO" else "viridis", s=14, alpha=0.8)
        axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
        fig.colorbar(scatter, ax=axis, fraction=0.045)
    subject_silhouette = float("nan")
    cluster_silhouette = float("nan")
    if 1 < len(np.unique(subjects[sample])) < len(sample):
        subject_silhouette = float(silhouette_score(embedding, subjects[sample]))
    if 1 < len(np.unique(labels[sample])) < len(sample):
        cluster_silhouette = float(silhouette_score(embedding, labels[sample]))
    interpretation = "features separate by subject" if subject_silhouette > cluster_silhouette else "facial clusters dominate subject identity"
    fig.suptitle(f"{prefix}Held-out t-SNE · perplexity={perplexity:g}; subject silhouette={subject_silhouette:.3f}; cluster silhouette={cluster_silhouette:.3f}; {interpretation}")
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    profile_rows = []
    for fold in folds:
        profile_rows.extend(fold["component_means_normalized"])
    profile_matrix = np.asarray([
        [profile.get(key, 0.0) for key in INTERPRETABLE_FEATURES]
        for profile in profile_rows
    ], dtype=np.float64)
    if profile_matrix.size:
        scale = profile_matrix.std(axis=0)
        profile_z = (profile_matrix - profile_matrix.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
        fig, axis = plt.subplots(figsize=(8, max(3, 0.3 * len(profile_z))))
        image = axis.imshow(profile_z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        axis.set(xticks=np.arange(len(INTERPRETABLE_FEATURES)), xticklabels=list(INTERPRETABLE_FEATURES),
                 ylabel="Fold/component", title=f"{prefix}Facial cluster profiles (standardized)")
        fig.colorbar(image, ax=axis, label="z-score")
        fig.tight_layout()
        fig.savefig(output_dir / "cluster_profiles.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    unique_labels = np.unique(labels)
    groups = [alpha[labels == label] for label in unique_labels]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.violinplot(groups, showmeans=True, showextrema=True)
    axis.set(xticks=np.arange(1, len(unique_labels) + 1), xticklabels=[str(value) for value in unique_labels],
             xlabel="Fold-qualified facial cluster", ylabel="z-scored log alpha",
             title=f"{prefix}Held-out alpha by facial cluster · permutation p={permutation['p_value']:.4g}")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    example_subject = str(subjects[0])
    example_rows = np.flatnonzero(subjects == example_subject)
    order = np.argsort(original_indices[example_rows])
    example_rows = example_rows[order]
    fig, axis = plt.subplots(figsize=(12, 2.8))
    axis.imshow(labels[example_rows][None, :], aspect="auto", interpolation="nearest", cmap="tab20")
    axis.set(yticks=[], xlabel="Input window order (timestamps/session IDs unavailable)",
             title=f"{prefix}Facial-state assignments · example subject {example_subject}")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "embedding_diagnostic.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "synthetic": synthetic,
            "label": "SYNTHETIC SANITY CHECK" if synthetic else None,
            "perplexity": perplexity,
            "subject_silhouette": subject_silhouette,
            "cluster_silhouette": cluster_silhouette,
            "interpretation": interpretation,
            "caveat": "Embedding is a visualization only; t-SNE is fit on held-out windows and is not used for model fitting or inference.",
        }, stream, indent=2)