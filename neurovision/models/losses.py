from __future__ import annotations

import torch
import torch.nn.functional as F


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    corr = (pred * target).sum(dim=-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + eps)
    return 1.0 - corr.mean()


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.abs(torch.fft.rfft(pred.float(), dim=-1))
    target_fft = torch.abs(torch.fft.rfft(target.float(), dim=-1))
    return F.smooth_l1_loss(pred_fft, target_fft)


def uncertainty_loss(pred: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL loss using predicted uncertainty (variance)."""
    # uncertainty[:, 0] corresponds to waveform uncertainty
    var = uncertainty[:, 0:1] + 1e-6
    sq_err = (pred - target) ** 2
    nll = 0.5 * (sq_err / var + torch.log(var))
    return nll.mean()


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {
        "waveform": 1.0,
        "spectral": 0.4,
        "band_power": 0.6,
        "correlation": 0.3,
        "uncertainty": 0.1,
    }
    waveform = F.smooth_l1_loss(outputs["waveform"], targets["waveform"])
    spectral = spectral_loss(outputs["waveform"], targets["waveform"])
    band_power = F.mse_loss(outputs["band_power"], targets["band_power"]) if "band_power" in targets else torch.tensor(0.0, device=waveform.device)
    correlation = pearson_loss(outputs["waveform"], targets["waveform"])

    unc_loss = torch.tensor(0.0, device=waveform.device)
    if "uncertainty" in outputs:
        unc_loss = uncertainty_loss(outputs["waveform"], targets["waveform"], outputs["uncertainty"])

    total = (
        weights["waveform"] * waveform
        + weights["spectral"] * spectral
        + weights["band_power"] * band_power
        + weights["correlation"] * correlation
        + weights["uncertainty"] * unc_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "waveform_loss": float(waveform.detach().cpu()),
        "spectral_loss": float(spectral.detach().cpu()),
        "band_power_loss": float(band_power.detach().cpu()),
        "correlation_loss": float(correlation.detach().cpu()),
        "uncertainty_loss": float(unc_loss.detach().cpu()),
    }
    return total, parts
