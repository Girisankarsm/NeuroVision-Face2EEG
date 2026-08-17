import pytest
import torch

from neurovision.config import DEFAULT_CONFIG
from neurovision.models.factory import build_model
from neurovision.models.losses import multitask_loss


@pytest.mark.parametrize("model_name", ["mlp", "temporal_cnn", "cnn_lstm", "transformer", "multimodal"])
def test_all_models_forward_and_output_shapes(model_name: str):
    config = DEFAULT_CONFIG.copy()
    config["model"] = dict(
        DEFAULT_CONFIG["model"],
        feature_dim=4233,
        hidden_dim=32,
        attention_heads=4,
        transformer_layers=1,
    )
    model = build_model(model_name, config)
    seq_len = 16
    x = torch.randn(2, seq_len, 4233)
    out = model(x)
    assert "waveform" in out
    assert "band_power" in out
    assert "spectral" in out
    assert "uncertainty" in out
    assert out["waveform"].shape == (2, config["eeg_window_samples"])
    assert out["band_power"].shape == (2, 5)
    assert out["spectral"].shape == (2, 64)
    assert out["uncertainty"].shape == (2, 6)

    # Test loss calculation
    targets = {
        "waveform": torch.randn(2, config["eeg_window_samples"]),
        "band_power": torch.rand(2, 5),
    }
    loss, parts = multitask_loss(out, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "waveform_loss" in parts
    assert "band_power_loss" in parts
