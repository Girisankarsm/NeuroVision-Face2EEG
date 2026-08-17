from __future__ import annotations

import torch
import torch.nn.functional as F


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    corr = (pred * target).sum(dim=-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + eps)
    return 1.0 - corr.mean()


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft(pred, dim=-1).abs()
    target_fft = torch.fft.rfft(target, dim=-1).abs()
    return F.smooth_l1_loss(pred_fft, target_fft)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {"waveform": 1.0, "spectral": 0.4, "band_power": 0.6, "correlation": 0.3}
    waveform = F.smooth_l1_loss(outputs["waveform"], targets["waveform"])
    spectral = spectral_loss(outputs["waveform"], targets["waveform"])
    band_power = F.mse_loss(outputs["band_power"], targets["band_power"])
    correlation = pearson_loss(outputs["waveform"], targets["waveform"])
    total = (
        weights["waveform"] * waveform
        + weights["spectral"] * spectral
        + weights["band_power"] * band_power
        + weights["correlation"] * correlation
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "waveform_loss": float(waveform.detach().cpu()),
        "spectral_loss": float(spectral.detach().cpu()),
        "band_power_loss": float(band_power.detach().cpu()),
        "correlation_loss": float(correlation.detach().cpu()),
    }
    return total, parts
