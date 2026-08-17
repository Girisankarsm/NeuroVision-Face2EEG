from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(self.act1(self.norm1(self.conv1(x))))
        out = self.drop2(self.act2(self.norm2(self.conv2(out))))
        return out + res


class TemporalCNN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, eeg_window_samples: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (B, T, D) -> transpose to (B, D, T)
        x_trans = x.transpose(1, 2)
        proj = self.input_proj(x_trans)
        features = self.blocks(proj)
        pooled = self.pool(features).squeeze(-1)
        return self.heads(pooled)
