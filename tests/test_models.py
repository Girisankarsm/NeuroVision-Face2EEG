import torch

from neurovision.config import DEFAULT_CONFIG
from neurovision.models.factory import build_model


def test_transformer_multitask_shapes():
    config = DEFAULT_CONFIG.copy()
    config["model"] = dict(DEFAULT_CONFIG["model"], hidden_dim=32, attention_heads=4, transformer_layers=1)
    model = build_model("transformer", config)
    out = model(torch.zeros(2, config["sequence_length"], config["model"]["feature_dim"]))
    assert out["waveform"].shape == (2, config["eeg_window_samples"])
    assert out["band_power"].shape == (2, 5)
    assert out["uncertainty"].shape == (2, 6)
