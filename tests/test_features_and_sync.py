import numpy as np

from neurovision.preprocessing.facial import FacialFeatureState, extract_facial_feature_vector
from neurovision.preprocessing.synchronization import align_windows


def test_facial_feature_vector_has_configured_width():
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.1, 0.9, 468)
    landmarks[:, 1] = np.linspace(0.2, 0.8, 468)
    features, _ = extract_facial_feature_vector(landmarks, FacialFeatureState())
    assert features.shape == (4233,)


def test_alignment_uses_timestamps_not_matching_indices():
    facial = np.ones((10, 4), dtype=np.float32)
    facial_timestamps = np.linspace(10.0, 10.9, 10)
    eeg = np.arange(1000, dtype=np.float32)
    eeg_timestamps = np.linspace(9.0, 12.0, 1000)
    windows = align_windows(
        facial,
        facial_timestamps,
        eeg,
        eeg_timestamps,
        subject_id="s1",
        sequence_length=4,
        eeg_window_samples=20,
        stride_frames=2,
    )
    assert windows
    assert windows[0].subject_id == "s1"
    assert windows[0].facial_window.shape == (4, 4)
    assert windows[0].eeg_window.shape == (20,)
