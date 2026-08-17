from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class MLPBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, eeg_window_samples: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.encoder(x))
