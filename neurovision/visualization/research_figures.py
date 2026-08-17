from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_vs_target(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_rate: float,
    output: str | Path,
    max_samples: int = 1000,
) -> None:
    act = np.asarray(actual).reshape(-1)[:max_samples]
    pred = np.asarray(predicted).reshape(-1)[:max_samples]
    t = np.arange(len(act)) / sample_rate

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(t, act, label="Ground Truth EEG", color="#2563eb", linewidth=1.4, alpha=0.85)
    ax.plot(t, pred, label="AI-Predicted EEG", color="#059669", linewidth=1.4, linestyle="--", alpha=0.85)
    ax.set_title("Ground Truth EEG vs AI-Predicted Neural Waveform", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Normalized Amplitude", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    output: str | Path,
) -> None:
    act = np.asarray(actual).reshape(-1)
    pred = np.asarray(predicted).reshape(-1)
    residuals = act - pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    # Residuals vs Predicted
    ax1.scatter(pred[:1500], residuals[:1500], alpha=0.35, color="#4f46e5", edgecolors="none", s=18)
    ax1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax1.set_title("Residuals vs. Predicted Values", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Amplitude", fontsize=10)
    ax1.set_ylabel("Residual (Target - Predicted)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Residual Distribution / Histogram
    ax2.hist(residuals, bins=50, color="#0ea5e9", edgecolor="#0284c7", alpha=0.75, density=True)
    ax2.axvline(0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax2.set_title("Error Distribution (Residuals)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_curves(
    history: dict[str, list[float]],
    output: str | Path,
) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    if "train_loss" in history and history["train_loss"]:
        ax.plot(epochs, history["train_loss"], label="Training Loss", color="#2563eb", linewidth=1.6)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#dc2626", linewidth=1.6)

    ax.set_title("Multi-Task Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Total Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_subject_metrics(
    subject_metrics: dict[str, dict[str, float]],
    output: str | Path,
) -> None:
    subjects = list(subject_metrics.keys())
    if not subjects:
        return

    corrs = [subject_metrics[s].get("pearson_correlation", 0.0) for s in subjects]
    rmses = [subject_metrics[s].get("rmse", 0.0) for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, len(subjects) * 1.2), 4.5), constrained_layout=True)
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width / 2, corrs, width, label="Pearson Correlation (r)", color="#3b82f6", alpha=0.85)
    rects2 = ax2.bar(x + width / 2, rmses, width, label="RMSE", color="#f97316", alpha=0.85)

    ax1.set_title("Subject-Independent Performance Evaluation", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Subject ID", fontsize=10)
    ax1.set_ylabel("Pearson Correlation", color="#1d4ed8", fontsize=10)
    ax2.set_ylabel("Root Mean Squared Error (RMSE)", color="#c2410c", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=25)
    ax1.grid(True, linestyle=":", alpha=0.5)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_band_power_comparison(
    actual_bands: dict[str, float],
    predicted_bands: dict[str, float],
    output: str | Path,
) -> None:
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    labels = [b.title() for b in bands]
    act_vals = [actual_bands.get(b, 0.0) for b in bands]
    pred_vals = [predicted_bands.get(b, 0.0) for b in bands]

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, act_vals, width, label="Ground Truth", color="#3b82f6", alpha=0.85)
    ax.bar(x + width / 2, pred_vals, width, label="AI-Predicted", color="#10b981", alpha=0.85)

    ax.set_title("Frequency Band Power Spectral Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Band Power (µV² / Hz)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, output: str | Path) -> None:
    required = {"model", "rmse", "mae", "pearson_correlation"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results table missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ["rmse", "mae", "pearson_correlation"]):
        ax.bar(results["model"], results[metric], color="#4f46e5", alpha=0.85)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, linestyle=":", alpha=0.6)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
