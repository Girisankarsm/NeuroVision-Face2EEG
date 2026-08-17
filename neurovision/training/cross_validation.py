from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import DataLoader, Subset

from neurovision.models.factory import build_model
from neurovision.models.losses import multitask_loss
from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from neurovision.training.metrics import regression_metrics
from neurovision.training.train import _batch_to_device, _device, _validate


def grouped_fold_indices(dataset_path: Path, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset = SynchronizedWindowDataset(dataset_path)
    unique_subjects = np.unique(dataset.subjects)
    if len(unique_subjects) < folds:
        raise ValueError(f"Need at least {folds} subjects for GroupKFold; found {len(unique_subjects)}.")
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(np.arange(len(dataset)), groups=dataset.subjects))


def run_cross_validation(
    config: dict,
    dataset_path: Path | str,
    model_name: str = "transformer",
    folds: int = 5,
    epochs: int = 20,
    output_dir: Path | str = "results/cv",
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = float(config.get("eeg_sample_rate", 256.0))
    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=sample_rate)
    unique_subjs = np.unique(dataset.subjects)
    actual_folds = min(folds, len(unique_subjs))

    if actual_folds < 2:
        raise ValueError("Cross-validation requires at least 2 subjects.")

    splitter = GroupKFold(n_splits=actual_folds)
    device = _device()
    fold_records: list[dict[str, Any]] = []

    print(f"=== Running {actual_folds}-Fold Subject-Independent Cross Validation [{model_name.upper()}] ===")
    print(f"Total Windows: {len(dataset)} | Total Subjects: {len(unique_subjs)}: {list(unique_subjs)}")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(dataset)), groups=dataset.subjects), start=1):
        train_subjs = set(dataset.subjects[train_idx])
        val_subjs = set(dataset.subjects[val_idx])
        assert len(train_subjs.intersection(val_subjs)) == 0, f"Subject leakage in fold {fold}!"

        print(f"\n--- Fold {fold}/{actual_folds} | Train subjects: {sorted(train_subjs)} | Val subjects: {sorted(val_subjs)} ---")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
        )

        model = build_model(model_name, config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                features, targets = _batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)
                loss, _ = multitask_loss(outputs, targets)
                loss.backward()
                optimizer.step()

            val_loss, _ = _validate(model, val_loader, device)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best state on val set
        model.load_state_dict(best_state)
        model.eval()
        preds = []
        targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                feats, trgs = _batch_to_device(batch, device)
                outs = model(feats)
                preds.append(outs["waveform"].cpu().numpy())
                targets_list.append(trgs["waveform"].cpu().numpy())

        fold_preds = np.concatenate(preds, axis=0)
        fold_targets = np.concatenate(targets_list, axis=0)
        metrics = regression_metrics(fold_preds, fold_targets, sample_rate=sample_rate)

        fold_record = {
            "model": model_name,
            "fold": fold,
            "val_subjects": ",".join(sorted(val_subjs)),
            **metrics,
        }
        fold_records.append(fold_record)
        print(f"Fold {fold} Results -> MAE: {metrics['mae']:.5f} | RMSE: {metrics['rmse']:.5f} | Pearson r: {metrics['pearson_correlation']:.4f} | R²: {metrics['r2']:.5f}")

    df_results = pd.DataFrame(fold_records)
    df_results.to_csv(output_dir / "cv_results.csv", index=False)

    print("\n=== Cross-Validation Summary (Mean ± Std) ===")
    for metric in ["mae", "rmse", "pearson_correlation", "spearman_correlation", "r2", "spectral_correlation"]:
        m_mean = df_results[metric].mean()
        m_std = df_results[metric].std()
        print(f"{metric.upper()}: {m_mean:.5f} ± {m_std:.5f}")

    return df_results
