from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurovision.models.factory import build_model
from neurovision.preprocessing.dataset import SynchronizedWindowDataset
from neurovision.training.metrics import regression_metrics
from neurovision.training.train import _device


@torch.no_grad()
def evaluate_checkpoint(config: dict, dataset_path: Path, checkpoint_path: Path) -> dict[str, float]:
    device = _device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_model(checkpoint["model_name"], model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = SynchronizedWindowDataset(dataset_path, sample_rate=float(model_config["eeg_sample_rate"]))
    loader = DataLoader(dataset, batch_size=int(model_config["training"]["batch_size"]))
    predictions = []
    targets = []
    for features, target, _subjects in loader:
        outputs = model(features.to(device))
        predictions.append(outputs["waveform"].cpu().numpy())
        targets.append(target["waveform"].numpy())
    if not predictions:
        raise ValueError("Dataset contains no windows to evaluate.")
    return regression_metrics(np.concatenate(predictions), np.concatenate(targets), sample_rate=float(model_config["eeg_sample_rate"]))
