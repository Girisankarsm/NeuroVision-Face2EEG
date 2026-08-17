import numpy as np

from neurovision.preprocessing.facial import (
    FacialFeatureState,
    compute_ear,
    compute_facial_symmetry,
    compute_mar,
    estimate_head_pose,
    extract_facial_feature_vector,
)
from neurovision.preprocessing.synchronization import align_windows


def test_facial_feature_vector_has_configured_width():
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.1, 0.9, 468)
    landmarks[:, 1] = np.linspace(0.2, 0.8, 468)
    features, state = extract_facial_feature_vector(landmarks, FacialFeatureState())
    assert features.shape == (4233,)
    assert state.left_ear >= 0.0
    assert state.mar >= 0.0
    assert len(state.head_pose) == 3


def test_geometric_calculations():
    landmarks = np.zeros((468, 3), dtype=np.float32)
    # Set eye landmarks
    landmarks[33] = [0.2, 0.3, 0.0]
    landmarks[133] = [0.4, 0.3, 0.0]
    landmarks[160] = [0.3, 0.28, 0.0]
    landmarks[158] = [0.35, 0.28, 0.0]
    landmarks[144] = [0.3, 0.32, 0.0]
    landmarks[153] = [0.35, 0.32, 0.0]

    ear = compute_ear(landmarks, [33, 160, 158, 133, 153, 144])
    assert ear > 0.0

    # Set mouth landmarks
    landmarks[61] = [0.3, 0.7, 0.0]
    landmarks[291] = [0.7, 0.7, 0.0]
    landmarks[13] = [0.5, 0.68, 0.0]
    landmarks[14] = [0.5, 0.72, 0.0]
    landmarks[82] = [0.4, 0.68, 0.0]
    landmarks[87] = [0.4, 0.72, 0.0]
    landmarks[312] = [0.6, 0.68, 0.0]
    landmarks[317] = [0.6, 0.72, 0.0]

    mar = compute_mar(landmarks)
    assert mar > 0.0

    symmetry = compute_facial_symmetry(landmarks)
    assert 0.0 <= symmetry <= 1.0

    rvec, tvec, (pitch, yaw, roll) = estimate_head_pose(landmarks)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert isinstance(roll, float)


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
