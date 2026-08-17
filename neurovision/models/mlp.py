from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class MLPBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, eeg_window_samples: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (B, sequence_length, feature_dim)
        # Apply MLP to each time step and pool temporally
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.heads(pooled)
