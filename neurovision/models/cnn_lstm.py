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
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(features)
        return self.heads(self.dropout(sequence[:, -1]))
