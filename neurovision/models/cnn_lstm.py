from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class CNNLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, eeg_window_samples: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, D) -> Conv1d: (B, hidden_dim, T) -> transpose: (B, T, hidden_dim)
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(features)
        # Pool across sequence and normalize
        pooled = self.layer_norm(lstm_out.mean(dim=1))
        return self.heads(pooled)
