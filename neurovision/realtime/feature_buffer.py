from __future__ import annotations

from collections import deque
import time

import numpy as np


class TemporalFeatureBuffer:
    def __init__(self, sequence_length: int, feature_dim: int) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._frames: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._timestamps: deque[float] = deque(maxlen=sequence_length)
        self._dropped_frames: int = 0
        self._expected_interval: float = 1.0 / 30.0

    def append(self, feature: np.ndarray, timestamp: float | None = None) -> None:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature.shape[-1]}.")
        curr_time = timestamp if timestamp is not None else time.time()
        if self._timestamps:
            delta = curr_time - self._timestamps[-1]
            if delta > self._expected_interval * 1.8:
                # Estimate missed frames
                missed = int(round(delta / self._expected_interval)) - 1
                self._dropped_frames += max(1, missed)
        self._frames.append(feature)
        self._timestamps.append(curr_time)

    def reset(self) -> None:
        self._frames.clear()
        self._timestamps.clear()

    @property
    def fill_ratio(self) -> float:
        return len(self._frames) / float(self.sequence_length)

    @property
    def current_length(self) -> int:
        return len(self._frames)

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def ready(self) -> bool:
        return len(self._frames) == self.sequence_length

    def tensor(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError(f"Feature buffer is not full yet ({len(self._frames)}/{self.sequence_length}).")
        return np.stack(self._frames, axis=0)[None, ...]

    def timestamps(self) -> np.ndarray:
        return np.asarray(self._timestamps, dtype=np.float64)

    def stats(self) -> dict[str, float]:
        if not self._frames:
            return {"mean": 0.0, "std": 0.0, "energy": 0.0, "jitter_ms": 0.0}
        arr = np.stack(self._frames, axis=0)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        energy = float(np.mean(np.square(arr)))
        jitter_ms = 0.0
        if len(self._timestamps) > 2:
            intervals = np.diff(self._timestamps) * 1000.0
            jitter_ms = float(np.std(intervals))
        return {"mean": mean_val, "std": std_val, "energy": energy, "jitter_ms": jitter_ms}
