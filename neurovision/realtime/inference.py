from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neurovision.models.factory import build_model


@dataclass
class Prediction:
    waveform: np.ndarray
    band_power: np.ndarray
    spectral: np.ndarray
    uncertainty: np.ndarray
    confidence: float


class EEGPredictor:
    def __init__(self, checkpoint_path: str | Path | None = None, fallback_config: dict | None = None) -> None:
        self.model: torch.nn.Module | None = None
        self.status = "MODEL NOT LOADED"
        self.config = fallback_config
        self.device = self._device()
        if checkpoint_path:
            self.load(checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.model = build_model(checkpoint["model_name"], self.config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.status = "READY"

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, mc_dropout_passes: int = 1) -> Prediction | None:
        if self.model is None:
            self.status = "MODEL NOT LOADED"
            return None
        x = torch.from_numpy(sequence.astype(np.float32)).to(self.device)
        outputs = []
        if mc_dropout_passes > 1:
            self.model.train()
        for _ in range(max(1, mc_dropout_passes)):
            outputs.append(self.model(x))
        self.model.eval()
        waveform = torch.stack([o["waveform"] for o in outputs]).mean(dim=0)[0].cpu().numpy()
        band = torch.stack([o["band_power"] for o in outputs]).mean(dim=0)[0].cpu().numpy()
        spectral = torch.stack([o["spectral"] for o in outputs]).mean(dim=0)[0].cpu().numpy()
        uncertainty = torch.stack([o["uncertainty"] for o in outputs]).mean(dim=0)[0].cpu().numpy()
        confidence = float(np.clip(1.0 / (1.0 + uncertainty.mean()), 0.0, 1.0))
        return Prediction(waveform, band, spectral, uncertainty, confidence)

    @staticmethod
    def _device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
