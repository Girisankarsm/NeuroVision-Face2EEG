from __future__ import annotations

import numpy as np


def band_percentages(band_power: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(band_power, dtype=np.float32), 0.0)
    total = float(values.sum()) + 1e-12
    return values / total
