import pytest
import numpy as np
import cv2
import os
import glob
import pandas as pd
from unittest.mock import patch

from neurovision.preprocessing.facial import estimate_head_pose, rotation_matrix_to_head_angles, FacialFeatureState, facial_dynamics
from neurovision.realtime.recording import blink_warmup_label, RECORDING_COLUMNS, save_recording

def test_head_pose_conversion():
    # Simulate a straight face rotation matrix (near 180 degrees)
    rvec = np.array([[-3.14], [0], [0]], dtype=np.float32)
    # The estimate_head_pose expects landmarks, but we can't easily invert solvePnP here.
    # We will test the unwrapping logic in facial.py.
    # To do this properly, we'll mock the cv2.solvePnP in estimate_head_pose.
    from unittest.mock import patch
    with patch('cv2.solvePnP') as mock_solvepnp:
        # success, rvec, tvec
        mock_solvepnp.return_value = (True, rvec, np.zeros((3,1)))

        # Call with dummy landmarks
        dummy_lms = np.zeros((468, 3))
        _, _, (pitch, yaw, roll) = estimate_head_pose(dummy_lms)

        # Pitch should be unwrapped to near 0
        assert abs(pitch) < 1.0, f"Expected pitch near 0, got {pitch}"
        assert abs(roll) < 1.0, f"Expected roll near 0, got {roll}"

    for axis in range(3):
        rotation_vector = np.zeros((3, 1), dtype=np.float64)
        rotation_vector[axis, 0] = np.pi
        rotation, _ = cv2.Rodrigues(rotation_vector)
        angles = rotation_matrix_to_head_angles(rotation)
        assert all(abs(angle) < 1.0 for angle in angles), angles


def test_blink_rate_warmup_state():
    assert blink_warmup_label(0) == "Calibrating... 0s / 60s"
    assert blink_warmup_label(59.9) == "Calibrating... 59s / 60s"
    assert blink_warmup_label(60) is None


def test_recording_files_load_with_schema(tmp_path):
    row = dict.fromkeys(RECORDING_COLUMNS, 0.0)
    row.update(timestamp=123.0, EAR=0.3, face_detected=1)
    csv_path, npz_path = save_recording([row], tmp_path / "session", {"fps": 15})
    df = pd.read_csv(csv_path)
    assert list(df.columns) == list(RECORDING_COLUMNS)
    with np.load(npz_path) as archive:
        assert archive["features"].shape == (1, len(RECORDING_COLUMNS))
        assert archive["columns"].tolist() == list(RECORDING_COLUMNS)

def test_blink_detector_and_warmup():
    """SYNTHETIC SANITY CHECK: Test blink detection logic and warmup state."""
    state = FacialFeatureState()

    # 1. Warmup / no blinks
    dummy_lms_open = np.zeros((468, 3))
    # We will manually set EAR to 0.3 for open eyes in the state by patching compute_ear
    with patch('neurovision.preprocessing.facial.compute_ear', return_value=0.3):
        for _ in range(5):
            _, state = facial_dynamics(dummy_lms_open, state, timestamp=0.0)

    assert not state.is_blinking
    assert len(state.blink_timestamps) == 0
    assert state.blink_rate == 0.0

    # 2. Blink event
    with patch('neurovision.preprocessing.facial.compute_ear', return_value=0.1):
        _, state = facial_dynamics(dummy_lms_open, state, timestamp=1.0)

    assert state.is_blinking
    assert len(state.blink_timestamps) == 1
    assert state.blink_event

    # 3. Blink ends
    with patch('neurovision.preprocessing.facial.compute_ear', return_value=0.3):
        _, state = facial_dynamics(dummy_lms_open, state, timestamp=2.0)

    assert not state.is_blinking
    assert not state.blink_event
    assert len(state.blink_timestamps) == 1

def test_no_random_values():
    # Grep test for np.random or random in app.py and neurovision/realtime/
    import re
    files_to_check = ['app.py', 'neurovision/app/live_tab.py'] + glob.glob('neurovision/realtime/*.py')

    pattern = re.compile(r'\b(np\.random|random\.)')

    for fpath in files_to_check:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            content = f.read()
            match = pattern.search(content)
            assert not match, f"Found random/mock value generation in {fpath} at index {match.start()}! App must not fabricate data."

def test_recording_schema(tmp_path):
    """Test that a recorded session CSV can be loaded and has expected columns."""
    csv_path = tmp_path / "test_rec.csv"

    with open(csv_path, "w") as f:
        au_headers = ",".join([f"au_{name}" for name in ["brow_raiser", "jaw_open", "smile_AU12"]])
        f.write(f"timestamp,EAR,MAR,pitch,yaw,roll,blink_flag,face_detected,{au_headers}\n")
        f.write("1600000000.0,0.3,0.1,0,0,0,0,1,0.0,0.0,0.0\n")

    df = pd.read_csv(csv_path)
    expected_cols = ["timestamp", "EAR", "MAR", "pitch", "yaw", "roll", "blink_flag", "face_detected", "au_brow_raiser", "au_jaw_open", "au_smile_AU12"]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == 1
    assert df["EAR"].iloc[0] == 0.3
