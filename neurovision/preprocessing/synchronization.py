from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedWindow:
    subject_id: str
    facial_window: np.ndarray
    eeg_window: np.ndarray
    start_time: float
    end_time: float


def align_windows(
    facial_features: np.ndarray,
    facial_timestamps: np.ndarray,
    eeg: np.ndarray,
    eeg_timestamps: np.ndarray,
    subject_id: str,
    sequence_length: int,
    eeg_window_samples: int,
    stride_frames: int = 1,
) -> list[AlignedWindow]:
    facial_features = np.asarray(facial_features, dtype=np.float32)
    facial_timestamps = np.asarray(facial_timestamps, dtype=np.float64)
    eeg = np.asarray(eeg, dtype=np.float32)
    eeg_timestamps = np.asarray(eeg_timestamps, dtype=np.float64)
    if eeg.ndim > 1:
        eeg = eeg.mean(axis=0)

    windows: list[AlignedWindow] = []
    for start in range(0, len(facial_features) - sequence_length + 1, stride_frames):
        end = start + sequence_length
        start_time = float(facial_timestamps[start])
        end_time = float(facial_timestamps[end - 1])
        center_time = (start_time + end_time) / 2.0
        eeg_center = int(np.searchsorted(eeg_timestamps, center_time))
        eeg_start = eeg_center - eeg_window_samples // 2
        eeg_end = eeg_start + eeg_window_samples
        if eeg_start < 0 or eeg_end > len(eeg):
            continue
        windows.append(
            AlignedWindow(
                subject_id=subject_id,
                facial_window=facial_features[start:end],
                eeg_window=eeg[eeg_start:eeg_end],
                start_time=start_time,
                end_time=end_time,
            )
        )
    return windows
