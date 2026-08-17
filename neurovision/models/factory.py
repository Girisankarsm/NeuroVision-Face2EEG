from __future__ import annotations

from torch import nn

from neurovision.models.cnn_lstm import CNNLSTM
from neurovision.models.mlp import MLPBaseline
from neurovision.models.temporal_cnn import TemporalCNN
from neurovision.models.transformer import FacialEEGTransformer


def build_model(name: str, config: dict) -> nn.Module:
    model_config = config["model"]
    args = {
        "feature_dim": int(model_config["feature_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "eeg_window_samples": int(config["eeg_window_samples"]),
        "dropout": float(model_config["dropout"]),
    }
    if name == "mlp":
        return MLPBaseline(**args)
    if name == "temporal_cnn":
        return TemporalCNN(**args)
    if name == "cnn_lstm":
        return CNNLSTM(**args)
    if name == "transformer":
        return FacialEEGTransformer(
            **args,
            layers=int(model_config["transformer_layers"]),
            heads=int(model_config["attention_heads"]),
        )
    raise ValueError(f"Unknown model: {name}")
