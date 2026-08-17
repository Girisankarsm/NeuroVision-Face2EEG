from __future__ import annotations

import math

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FacialEEGTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        eeg_window_samples: int,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.position = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.position(self.projection(x)))
        return self.heads(encoded.mean(dim=1))
