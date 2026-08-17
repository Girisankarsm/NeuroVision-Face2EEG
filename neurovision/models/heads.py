from __future__ import annotations

import torch
from torch import nn


class MultiTaskEEGHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eeg_window_samples: int,
        band_count: int = 5,
        spectral_bins: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.waveform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, eeg_window_samples),
        )
        self.band_power = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, band_count),
        )
        self.spectral = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, spectral_bins),
        )
        # Predictive uncertainty (log-variance / standard deviation)
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, band_count + 1),
            nn.Softplus(),
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.dropout(latent)
        return {
            "waveform": self.waveform(latent),
            "band_power": self.band_power(latent),
            "spectral": self.spectral(latent),
            "uncertainty": self.uncertainty(latent),
        }
