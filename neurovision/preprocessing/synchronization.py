"""Temporal alignment of facial feature windows to EEG segments.

Alignment Convention
--------------------
Each facial feature window spans ``sequence_length`` consecutive video frames
with timestamps ``[t_start, t_end]``.

The **corresponding EEG segment** is the ``eeg_window_samples``-long slice of
the (possibly multichannel, averaged to single-channel) EEG signal whose
**temporal center** coincides with the **temporal center** of the facial window:

    facial window:  [t_start ............. t_end]
                              ^
                        t_center = (t_start + t_end) / 2

    EEG segment:        [eeg_start ... eeg_end]
                          centered on t_center
                          length = eeg_window_samples

So the EEG slice ``[eeg_center - W/2 : eeg_center + W/2]`` where
``eeg_center = searchsorted(eeg_timestamps, t_center)`` and
``W = eeg_window_samples``.

This means the model predicts the EEG activity that **co-occurs with**
(not follows) the facial dynamics, which is the most conservative
assumption for a correlation study. Future work could add a configurable
lag/lead offset.

Diagram::

    Time ─────────────────────────────────────────────────────►
    Video frames:  f1  f2  f3 ... f64
                   |__________________|
                   t_start    t_center    t_end
                                 │
                                 ▼
    EEG:          ━━━━━━━━━[===CENTER===]━━━━━━━━━
                          eeg_start  eeg_end
"""
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
    """Create aligned (facial, EEG) window pairs.

    Parameters
    ----------
    facial_features : array, shape (N_frames, D)
    facial_timestamps : array, shape (N_frames,)
    eeg : array, shape (C, N_eeg_samples) or (N_eeg_samples,)
        If multichannel, averaged to a single channel.
    eeg_timestamps : array, shape (N_eeg_samples,)
    subject_id : str
    sequence_length : int
        Number of consecutive facial frames per window.
    eeg_window_samples : int
        Number of EEG samples per window.
    stride_frames : int
        Stride between consecutive windows in facial frames.

    Returns
    -------
    list[AlignedWindow]
        Aligned pairs. Windows that would exceed EEG bounds are dropped.
    """
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


def verify_alignment(
    window: AlignedWindow,
    eeg_timestamps: np.ndarray,
    eeg_window_samples: int,
    tolerance_sec: float = 0.1,
) -> bool:
    """Verify that an aligned window satisfies the centering invariant.

    The EEG segment center should be within ``tolerance_sec`` of the
    facial window center.
    """
    facial_center = (window.start_time + window.end_time) / 2.0
    eeg_ts = np.asarray(eeg_timestamps, dtype=np.float64)
    eeg_center_idx = int(np.searchsorted(eeg_ts, facial_center))
    eeg_start = eeg_center_idx - eeg_window_samples // 2
    eeg_end = eeg_start + eeg_window_samples
    if eeg_start < 0 or eeg_end > len(eeg_ts):
        return False
    eeg_center_time = float(eeg_ts[eeg_center_idx])
    return abs(eeg_center_time - facial_center) < tolerance_sec
