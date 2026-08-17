from __future__ import annotations

from collections import deque

import numpy as np


class TemporalFeatureBuffer:
    def __init__(self, sequence_length: int, feature_dim: int) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._frames: deque[np.ndarray] = deque(maxlen=sequence_length)

    def append(self, feature: np.ndarray) -> None:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature.shape[-1]}.")
        self._frames.append(feature)

    @property
    def fill_ratio(self) -> float:
        return len(self._frames) / self.sequence_length

    def ready(self) -> bool:
        return len(self._frames) == self.sequence_length

    def tensor(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("Feature buffer is not full yet.")
        return np.stack(self._frames, axis=0)[None, ...]
