import pytest
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import pandas as pd
import torch
from unittest.mock import patch

from neurovision.preprocessing.facial import (
    FacialFeatureState,
    calibrated_blink_detected,
    calibrated_blink_threshold,
    estimate_head_pose,
    facial_dynamics,
    rotation_matrix_to_head_angles,
)
from neurovision.app.live_tab import _estimate_facial_state, _model_feature
from neurovision.config import load_config
from neurovision.realtime.recording import (
    LIVE_TARGET_FPS,
    RECORDING_COLUMNS,
    blink_warmup_label,
    relative_chart_time,
    save_recording,
)
from neurovision.realtime.inference import EEGPredictor

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


def test_live_chart_uses_session_relative_seconds():
    assert LIVE_TARGET_FPS == 18
    assert relative_chart_time(1_700_000_012.5, 1_700_000_000.0) == 12.5


def test_live_cluster_artifact_is_optional_and_uses_probabilities():
    class Scaler:
        def transform(self, values, subjects):
            assert subjects.tolist() == ["live"]
            return values

    class PCA:
        def transform(self, values):
            return values[:, :2]

    class GMM:
        def predict_proba(self, values):
            return np.asarray([[0.1, 0.9]])

    assert _estimate_facial_state([], None) == "N/A"
    assert _estimate_facial_state([np.zeros(28)] * 64, {"synthetic_training_data": True}) == "N/A"
    artifact = {
        "window_frames": 64,
        "scaler": Scaler(),
        "pca": PCA(),
        "gmm": GMM(),
        "cluster_names": ["low motion", "frequent blinking (suggestion; review manually)"],
    }
    assert _estimate_facial_state([np.zeros(28)] * 63, artifact) == "N/A"
    estimate = _estimate_facial_state([np.zeros(28)] * 64, artifact)
    assert estimate.startswith("frequent blinking")
    assert "90%" in estimate


def test_model_feature_matches_checkpoint_dimensions():
    landmarks = np.zeros((468, 3), dtype=np.float32)
    state = FacialFeatureState()
    compact = np.ones(28, dtype=np.float32)

    compact_result, compact_state = _model_feature(
        28, compact, landmarks, state, None, 1.0, (480, 640)
    )
    assert compact_result is compact
    assert compact_state is state

    full = np.zeros(4233, dtype=np.float32)
    with patch("neurovision.app.live_tab.extract_full_feature_vector", return_value=(full, state)):
        full_result, full_state = _model_feature(
            4233, compact, landmarks, state, None, 1.0, (480, 640)
        )
    assert full_result.shape == (4233,)
    assert full_state is state
    assert _model_feature(123, compact, landmarks, state, None, 1.0, (480, 640))[0] is None


def test_checked_in_eeg_checkpoint_predicts_expected_shapes(monkeypatch):
    """SYNTHETIC SANITY CHECK: verify checkpoint inference from extracted full facial features."""
    checkpoint = Path(__file__).resolve().parents[1] / "neurovision/models/checkpoints/best.pt"
    assert checkpoint.exists()
    monkeypatch.setattr(EEGPredictor, "_device", staticmethod(lambda: torch.device("cpu")))
    predictor = EEGPredictor(checkpoint, fallback_config=load_config("configs/config.yaml"))

    assert predictor.status == "READY", predictor.status_detail
    assert predictor.metadata.input_dim == (64, 4233)
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.1, 0.9, len(landmarks))
    landmarks[:, 1] = np.linspace(0.2, 0.8, len(landmarks))
    full_feature, _ = _model_feature(
        4233, np.zeros(28, dtype=np.float32), landmarks, FacialFeatureState(),
        None, 1.0, (480, 640),
    )
    assert full_feature.shape == (4233,)
    sequence = np.broadcast_to(full_feature, (1, 64, 4233)).copy()
    prediction = predictor.predict(sequence)

    assert prediction is not None
    assert prediction.waveform.shape == (128,)
    assert prediction.band_power.shape == (5,)
    assert np.isfinite(prediction.waveform).all()
    assert np.isfinite(prediction.band_power).all()


def test_missing_eeg_checkpoint_remains_unloaded(tmp_path):
    predictor = EEGPredictor(
        tmp_path / "missing.pt",
        fallback_config=load_config("configs/config.yaml"),
    )
    assert predictor.status == "MODEL NOT LOADED"
    assert "not found" in predictor.status_detail


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


def test_calibrated_blink_detection_at_18_fps():
    baseline_ear = 0.3
    threshold = calibrated_blink_threshold(baseline_ear)
    state = FacialFeatureState(blink_threshold=threshold)
    landmarks = np.zeros((468, 3))

    assert not calibrated_blink_detected(0.3, threshold)
    assert calibrated_blink_detected(0.1, threshold)
    eye_readings = np.repeat([0.3] * 18 + [0.1] * 3 + [0.3] * 3, 2).tolist()
    with patch("neurovision.preprocessing.facial.compute_ear", side_effect=eye_readings):
        events = []
        for frame in range(24):
            _, state = facial_dynamics(landmarks, state, timestamp=frame / 18)
            events.append(state.blink_event)

    assert sum(events) == 1
    assert events[18]
    assert list(state.blink_timestamps) == [18 / 18]

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
