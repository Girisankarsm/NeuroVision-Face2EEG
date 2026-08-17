from __future__ import annotations

from torch import nn

from neurovision.models.cnn_lstm import CNNLSTM
from neurovision.models.mlp import MLPBaseline
from neurovision.models.multimodal import MultimodalFusionModel
from neurovision.models.temporal_cnn import TemporalCNN
from neurovision.models.transformer import FacialEEGTransformer


def build_model(name: str, config: dict) -> nn.Module:
    model_config = config.get("model", {})
    args = {
        "feature_dim": int(model_config.get("feature_dim", 4233)),
        "hidden_dim": int(model_config.get("hidden_dim", 256)),
        "eeg_window_samples": int(config.get("eeg_window_samples", 128)),
        "dropout": float(model_config.get("dropout", 0.15)),
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
            layers=int(model_config.get("transformer_layers", 3)),
            heads=int(model_config.get("attention_heads", 4)),
        )
    if name == "multimodal":
        return MultimodalFusionModel(
            **args,
        )
    raise ValueError(f"Unknown model architecture: '{name}'. Supported: ['mlp', 'temporal_cnn', 'cnn_lstm', 'transformer', 'multimodal']")
