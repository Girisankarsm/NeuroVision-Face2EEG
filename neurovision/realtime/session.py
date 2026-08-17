from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class SessionRecorder:
    def __init__(self, root: str | Path = "neurovision/sessions") -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root) / timestamp
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "plots").mkdir(exist_ok=True)
        self.features: list[np.ndarray] = []
        self.predictions: list[np.ndarray] = []
        self.band_power: list[np.ndarray] = []

    def append(self, feature: np.ndarray | None, waveform: np.ndarray | None, band_power: np.ndarray | None) -> None:
        if feature is not None:
            self.features.append(feature)
        if waveform is not None:
            self.predictions.append(waveform)
        if band_power is not None:
            self.band_power.append(band_power)

    def save(self, metadata: dict) -> Path:
        if self.features:
            pd.DataFrame(np.asarray(self.features)).to_csv(self.path / "facial_features.csv", index=False)
        if self.predictions:
            np.save(self.path / "eeg_prediction.npy", np.asarray(self.predictions))
            pd.DataFrame(np.asarray(self.predictions)).to_csv(self.path / "predictions.csv", index=False)
        if self.band_power:
            pd.DataFrame(np.asarray(self.band_power), columns=["delta", "theta", "alpha", "beta", "gamma"]).to_csv(
                self.path / "band_power.csv", index=False
            )
        with (self.path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return self.path
