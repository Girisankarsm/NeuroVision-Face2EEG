from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback

import numpy as np
import torch

from neurovision.models.factory import build_model


@dataclass
class Prediction:
    waveform: np.ndarray
    band_power: np.ndarray
    spectral: np.ndarray
    uncertainty: np.ndarray
    confidence: float | None = None
    std_waveform: np.ndarray | None = None
    is_calibrated: bool = False


@dataclass
class ModelMetadata:
    model_name: str = "Unknown"
    checkpoint_path: str = "None"
    param_count: int = 0
    input_dim: tuple[int, int] = (64, 4233)
    output_dim: tuple[int, int] = (128, 5)
    device: str = "CPU"
    best_val_loss: float | None = None


class EEGPredictor:
    def __init__(self, checkpoint_path: str | Path | None = None, fallback_config: dict | None = None) -> None:
        self.model: torch.nn.Module | None = None
        self.status = "MODEL NOT LOADED"
        self.status_detail = "No checkpoint supplied."
        self.config = fallback_config
        self.device = self._device()
        self.metadata = ModelMetadata(device=str(self.device).upper())
        if checkpoint_path:
            self.load(checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> bool:
        path = Path(checkpoint_path)
        if not path.exists():
            self.status = "MODEL NOT LOADED"
            self.status_detail = f"Checkpoint file not found: {path}"
            return False

        try:
            self.status = "LOADING MODEL..."
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.config = checkpoint.get("config", self.config or {})
            model_name = checkpoint.get("model_name", "transformer")
            self.model = build_model(model_name, self.config).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

            # Parameter count
            param_count = sum(p.numel() for p in self.model.parameters())
            seq_len = int(self.config.get("sequence_length", 64))
            feat_dim = int(self.config.get("model", {}).get("feature_dim", 4233))
            eeg_samples = int(self.config.get("eeg_window_samples", 128))

            dev_name = "Apple MPS" if self.device.type == "mps" else ("NVIDIA CUDA" if self.device.type == "cuda" else "Host CPU")
            self.metadata = ModelMetadata(
                model_name=model_name.upper(),
                checkpoint_path=path.name,
                param_count=param_count,
                input_dim=(seq_len, feat_dim),
                output_dim=(eeg_samples, 5),
                device=dev_name,
                best_val_loss=checkpoint.get("best_val_loss"),
            )
            self.status = "READY"
            self.status_detail = f"Loaded {model_name} ({param_count / 1e6:.2f}M params) on {dev_name}"
            return True
        except Exception as exc:
            self.model = None
            self.status = "MODEL ERROR"
            self.status_detail = f"Failed to load checkpoint: {exc}"
            traceback.print_exc()
            return False

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, mc_dropout_passes: int = 1) -> Prediction | None:
        if self.model is None or self.status != "READY":
            return None

        x = torch.from_numpy(sequence.astype(np.float32)).to(self.device)
        outputs = []

        if mc_dropout_passes > 1:
            self.model.train()
            for _ in range(mc_dropout_passes):
                outputs.append(self.model(x))
            self.model.eval()
        else:
            outputs.append(self.model(x))

        waveforms = torch.stack([o["waveform"] for o in outputs])  # (passes, B, samples)
        bands = torch.stack([o["band_power"] for o in outputs])
        spectrals = torch.stack([o["spectral"] for o in outputs])
        uncertainties = torch.stack([o["uncertainty"] for o in outputs])

        mean_waveform = waveforms.mean(dim=0)[0].cpu().numpy()
        mean_band = bands.mean(dim=0)[0].cpu().numpy()
        mean_spectral = spectrals.mean(dim=0)[0].cpu().numpy()
        mean_uncertainty = uncertainties.mean(dim=0)[0].cpu().numpy()

        std_waveform = None
        if mc_dropout_passes > 1:
            std_waveform = waveforms.std(dim=0)[0].cpu().numpy()

        # Calibration: If uncertainty head produces positive variance, compute inverse variance confidence
        # Otherwise if uncalibrated, confidence is explicit
        variance_val = float(mean_uncertainty.mean())
        confidence = float(np.clip(1.0 / (1.0 + variance_val), 0.0, 1.0)) if variance_val > 0 else None

        return Prediction(
            waveform=mean_waveform,
            band_power=mean_band,
            spectral=mean_spectral,
            uncertainty=mean_uncertainty,
            confidence=confidence,
            std_waveform=std_waveform,
            is_calibrated=confidence is not None,
        )

    @staticmethod
    def _device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
