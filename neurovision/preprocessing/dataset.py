from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from neurovision.preprocessing.eeg import band_power


class SynchronizedWindowDataset(Dataset):
    """Loads real synchronized windows; this class intentionally does not synthesize targets."""

    def __init__(self, path: str | Path, sample_rate: float = 256.0) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        required = {"facial", "eeg", "subjects"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")
        self.facial = archive["facial"].astype(np.float32)
        self.eeg = archive["eeg"].astype(np.float32)
        self.subjects = archive["subjects"].astype(str)
        self.sample_rate = sample_rate
        if len(self.facial) != len(self.eeg) or len(self.facial) != len(self.subjects):
            raise ValueError("facial, eeg, and subjects arrays must contain the same number of windows.")

    def __len__(self) -> int:
        return len(self.facial)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        eeg = self.eeg[index]
        powers = band_power(eeg, self.sample_rate)
        target = {
            "waveform": torch.from_numpy(eeg.astype(np.float32)),
            "band_power": torch.tensor([powers[k] for k in ["delta", "theta", "alpha", "beta", "gamma"]], dtype=torch.float32),
            "spectral": torch.empty(0),
        }
        return torch.from_numpy(self.facial[index]), target, self.subjects[index]


class BandPowerDataset(Dataset):
    """Dataset that returns log band power as the primary target.

    Target is log10(band_power + eps) for each of the 5 standard bands.
    This is more realistic than predicting raw waveforms.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: float = 256.0,
        log_eps: float = 1e-10,
    ) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        required = {"facial", "eeg", "subjects"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")
        self.facial = archive["facial"].astype(np.float32)
        self.eeg_raw = archive["eeg"].astype(np.float32)
        self.subjects = archive["subjects"].astype(str)
        self.sample_rate = sample_rate
        self.log_eps = log_eps

        if len(self.facial) != len(self.eeg_raw) or len(self.facial) != len(self.subjects):
            raise ValueError("facial, eeg, and subjects arrays must contain the same number of windows.")

        # Pre-compute log band power targets
        self.band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        self._log_band_powers = self._compute_all_log_band_powers()

    def _compute_all_log_band_powers(self) -> np.ndarray:
        """Pre-compute log10(band_power + eps) for all windows."""
        powers = np.zeros((len(self.eeg_raw), len(self.band_names)), dtype=np.float32)
        for i in range(len(self.eeg_raw)):
            bp = band_power(self.eeg_raw[i], self.sample_rate)
            for j, name in enumerate(self.band_names):
                powers[i, j] = np.log10(bp.get(name, 0.0) + self.log_eps)
        return powers

    @property
    def targets(self) -> np.ndarray:
        """All log band power targets, shape (N, 5)."""
        return self._log_band_powers

    def __len__(self) -> int:
        return len(self.facial)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return (
            torch.from_numpy(self.facial[index]),
            torch.from_numpy(self._log_band_powers[index]),
            self.subjects[index],
        )


def load_band_power_arrays(
    path: str | Path,
    sample_rate: float = 256.0,
    log_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset and return (features_flat, log_band_power, subjects).

    Features are flattened from (N, T, D) to (N, T*D) for sklearn models.
    """
    ds = BandPowerDataset(path, sample_rate=sample_rate, log_eps=log_eps)
    X = ds.facial.reshape(len(ds), -1)
    y = ds.targets
    return X, y, ds.subjects
