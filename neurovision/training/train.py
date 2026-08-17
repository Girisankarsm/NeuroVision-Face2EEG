from __future__ import annotations

from pathlib import Path

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


def train_model(config: dict, dataset_path: Path, model_name: str, output_path: Path) -> None:
    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=float(config["eeg_sample_rate"]))
    groups = dataset.subjects
    if len(set(groups)) < 2:
        raise ValueError("Subject-independent validation needs at least two subjects.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=int(config["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=int(config["training"]["batch_size"]))

    device = _device()
    model = build_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    best_val = float("inf")
    stale_epochs = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        train_losses = []
        for batch in train_loader:
            features, targets = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss, _ = multitask_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = _validate(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"epoch={epoch + 1} train_loss={np.mean(train_losses):.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save(
                {"model_name": model_name, "config": config, "state_dict": model.state_dict(), "best_val_loss": best_val},
                output_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["training"]["patience"]):
                break


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        features, targets = _batch_to_device(batch, device)
        outputs = model(features)
        loss, _ = multitask_loss(outputs, targets)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")
