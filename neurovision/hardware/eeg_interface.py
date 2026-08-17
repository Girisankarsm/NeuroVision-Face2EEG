from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EEGSample:
    timestamp: float
    channels: np.ndarray
    sample_rate: float


class ActualEEGSource:
    def read(self) -> EEGSample | None:
        raise NotImplementedError("Implement this for a specific EEG device SDK.")

    def close(self) -> None:
        pass
