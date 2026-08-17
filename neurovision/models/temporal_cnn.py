from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class TemporalCNN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, eeg_window_samples: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.net(x.transpose(1, 2)))
