from __future__ import annotations

import torch
from torch import nn

from neurovision.models.heads import MultiTaskEEGHeads
from neurovision.models.transformer import PositionalEncoding


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion architecture for facial-dynamics to EEG prediction.
    Disentangles spatial landmarks, geometric invariants, temporal kinematics,
    and blendshape action units into specialized subnetworks, fusing them with
    cross-modal attention and temporal modeling.
    """

    def __init__(
        self,
        feature_dim: int = 4233,
        hidden_dim: int = 256,
        eeg_window_samples: int = 128,
        spatial_dim: int = 1415,
        dynamics_dim: int = 2810,
        blendshape_dim: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.dynamics_dim = dynamics_dim
        self.blendshape_dim = blendshape_dim

        # Stream 1: Spatial Landmark + Geometric Features
        self.spatial_net = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stream 2: Kinematics (Velocity, Acceleration, Energy)
        self.dynamics_net = nn.Sequential(
            nn.Conv1d(dynamics_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stream 3: Blendshape / Action Unit Intensities
        self.blendshape_net = nn.Sequential(
            nn.LayerNorm(blendshape_dim),
            nn.Linear(blendshape_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion & Gating Layer
        fused_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal Sequence Processing
        self.position = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        # Multi-task prediction heads
        self.heads = MultiTaskEEGHeads(hidden_dim, eeg_window_samples, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (B, T, feature_dim)
        b, t, d = x.shape
        if d >= self.spatial_dim + self.dynamics_dim + self.blendshape_dim:
            spatial_in = x[..., : self.spatial_dim]
            dynamics_in = x[..., self.spatial_dim : self.spatial_dim + self.dynamics_dim]
            blendshapes_in = x[..., self.spatial_dim + self.dynamics_dim : self.spatial_dim + self.dynamics_dim + self.blendshape_dim]
        else:
            # Fallback if dimension is smaller or uniform
            spatial_in = x
            dynamics_in = x
            blendshapes_in = x[..., : min(d, self.blendshape_dim)]

        spatial_feat = self.spatial_net(spatial_in)  # (B, T, hidden_dim // 2)

        dynamics_trans = dynamics_in.transpose(1, 2)  # (B, dynamics_dim, T)
        dynamics_feat = self.dynamics_net(dynamics_trans).transpose(1, 2)  # (B, T, hidden_dim // 2)

        blendshape_feat = self.blendshape_net(blendshapes_in)  # (B, T, hidden_dim // 4)

        fused = torch.cat([spatial_feat, dynamics_feat, blendshape_feat], dim=-1)
        projected = self.fusion_proj(fused)

        encoded = self.temporal_encoder(self.position(projected))
        pooled = encoded.mean(dim=1)
        return self.heads(pooled)
