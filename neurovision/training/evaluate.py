from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neurovision.models.factory import build_model
from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from neurovision.preprocessing.eeg import band_power
from neurovision.training.metrics import regression_metrics
from neurovision.training.train import _device
from neurovision.visualization.research_figures import (
    plot_band_power_comparison,
    plot_loss_curves,
    plot_per_subject_metrics,
    plot_prediction_vs_target,
    plot_residuals,
)


@torch.no_grad()
def evaluate_checkpoint(
    config: dict,
    dataset_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str = "results",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model_name = checkpoint.get("model_name", "transformer")

    model = build_model(model_name, model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    sample_rate = float(model_config.get("eeg_sample_rate", 256.0))
    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=sample_rate)
    loader = DataLoader(dataset, batch_size=int(model_config.get("training", {}).get("batch_size", 32)), shuffle=False)

    predictions = []
    targets = []
    all_subjects = []

    for features, target_dict, subjects in loader:
        outputs = model(features.to(device))
        predictions.append(outputs["waveform"].cpu().numpy())
        targets.append(target_dict["waveform"].numpy())
        all_subjects.extend(subjects)

    if not predictions:
        raise ValueError(f"Dataset at {dataset_path} contains no windows to evaluate.")

    pred_arr = np.concatenate(predictions, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    subjects_arr = np.asarray(all_subjects)

    # Overall Metrics
    overall_metrics = regression_metrics(pred_arr, target_arr, sample_rate=sample_rate)

    # Per-Subject Metrics Breakdown
    per_subject: dict[str, dict[str, float]] = {}
    unique_subjs = np.unique(subjects_arr)
    for subj in unique_subjs:
        idx = np.where(subjects_arr == subj)[0]
        per_subject[str(subj)] = regression_metrics(pred_arr[idx], target_arr[idx], sample_rate=sample_rate)

    # Spectral band comparisons
    actual_bands = band_power(target_arr.reshape(-1), sample_rate)
    pred_bands = band_power(pred_arr.reshape(-1), sample_rate)

    # Generate Research Figures
    plot_prediction_vs_target(target_arr, pred_arr, sample_rate, output_dir / "prediction_vs_ground_truth.png")
    plot_residuals(target_arr, pred_arr, output_dir / "residuals.png")
    plot_per_subject_metrics(per_subject, output_dir / "per_subject_metrics.png")
    plot_band_power_comparison(actual_bands, pred_bands, output_dir / "band_powers.png")

    if "history" in checkpoint and checkpoint["history"]:
        plot_loss_curves(checkpoint["history"], output_dir / "loss_curves.png")

    # Save metrics JSON & CSV summary
    report = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "total_windows": len(dataset),
        "subjects": list(unique_subjs),
        "overall_metrics": overall_metrics,
        "per_subject_metrics": per_subject,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_summary = pd.DataFrame([{"model": model_name, **overall_metrics}])
    df_summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    print(f"\n=== Evaluation Summary [{model_name.upper()}] ===")
    print(f"Dataset Windows: {len(dataset)} across {len(unique_subjs)} subjects: {list(unique_subjs)}")
    print(f"MAE: {overall_metrics['mae']:.5f} | RMSE: {overall_metrics['rmse']:.5f} | R²: {overall_metrics['r2']:.5f}")
    print(f"Pearson r: {overall_metrics['pearson_correlation']:.4f} | Spearman rho: {overall_metrics['spearman_correlation']:.4f}")
    print(f"Spectral Corr: {overall_metrics['spectral_correlation']:.4f}")
    print(f"Saved evaluation figures and reports to: {output_dir.resolve()}\n")

    return report
