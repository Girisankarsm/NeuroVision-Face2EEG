from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "sequence_length": 64,
    "eeg_window_samples": 128,
    "eeg_sample_rate": 256,
    "bands": {
        "delta": [0.5, 4.0],
        "theta": [4.0, 8.0],
        "alpha": [8.0, 13.0],
        "beta": [13.0, 30.0],
        "gamma": [30.0, 100.0],
    },
    "model": {
        "feature_dim": 4233,
        "hidden_dim": 256,
        "dropout": 0.15,
        "attention_heads": 4,
        "transformer_layers": 3,
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "patience": 8,
        "gradient_clip": 1.0,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(config, loaded)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
