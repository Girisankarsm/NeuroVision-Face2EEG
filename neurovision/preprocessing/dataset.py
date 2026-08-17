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
