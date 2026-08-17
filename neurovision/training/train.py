from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from neurovision.models.factory import build_model
from neurovision.models.losses import multitask_loss
from neurovision.preprocessing.dataset import SynchronizedWindowDataset


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_device(batch, device: torch.device):
    features, targets, _subjects = batch
    return features.to(device), {key: value.to(device) for key, value in targets.items() if value.numel() > 0}


def train_model(
    config: dict,
    dataset_path: Path,
    model_name: str,
    output_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=float(config["eeg_sample_rate"]))
    groups = dataset.subjects
    unique_subjects = list(set(groups))
    if len(unique_subjects) < 2:
        raise ValueError(f"Subject-independent validation requires at least 2 distinct subjects, found: {unique_subjects}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    # Assert zero subject leakage
    assert len(train_subjects.intersection(val_subjects)) == 0, "Subject leakage detected in split!"

    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )

    device = _device()
    model = build_model(model_name, config).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(config["training"].get("scheduler_patience", 3)),
        min_lr=1e-6,
    )

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "waveform_loss": [],
        "spectral_loss": [],
        "band_power_loss": [],
        "correlation_loss": [],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Training [{model_name.upper()}] ===")
    print(f"Architecture: {model_name} | Parameters: {param_count:,} | Device: {device}")
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)} | Val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"Train windows: {len(train_subset)} | Val windows: {len(val_subset)}")

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    grad_clip = float(config["training"]["gradient_clip"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        comp_losses = {"waveform_loss": [], "spectral_loss": [], "band_power_loss": [], "correlation_loss": []}

        for batch in train_loader:
            features, targets = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss, parts = multitask_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            for k in comp_losses:
                if k in parts:
                    comp_losses[k].append(parts[k])

        val_loss, val_parts = _validate(model, val_loader, device)
        scheduler.step(val_loss)

        avg_train = float(np.mean(train_losses))
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        for k in comp_losses:
            history[k].append(float(np.mean(comp_losses[k])) if comp_losses[k] else 0.0)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs:03d} | Train: {avg_train:.5f} | Val: {val_loss:.5f} | LR: {current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "config": config,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "epoch": best_epoch,
                    "param_count": param_count,
                    "history": history,
                    "train_subjects": list(train_subjects),
                    "val_subjects": list(val_subjects),
                },
                output_path,
            )
            print(f"  --> Saved new best checkpoint to {output_path} (Val Loss: {best_val:.5f})")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience: {patience}).")
                break

    print(f"=== Training Complete | Best Val Loss: {best_val:.5f} (Epoch {best_epoch}) ===")
    return {
        "model_name": model_name,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(output_path),
        "param_count": param_count,
    }


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    comp_accum: dict[str, list[float]] = {}
    for batch in loader:
        features, targets = _batch_to_device(batch, device)
        outputs = model(features)
        loss, parts = multitask_loss(outputs, targets)
        losses.append(float(loss.detach().cpu()))
        for k, v in parts.items():
            comp_accum.setdefault(k, []).append(v)

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    mean_parts = {k: float(np.mean(v)) for k, v in comp_accum.items()}
    return mean_loss, mean_parts
