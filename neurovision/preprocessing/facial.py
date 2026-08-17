from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import cv2
import numpy as np


# 3D canonical model facial landmark coordinates for head pose estimation (in mm)
CANONICAL_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
        [0.0, -330.0, -65.0],     # Chin (landmark 152 or 199)
        [-225.0, 170.0, -135.0],  # Left eye left corner (landmark 33)
        [225.0, 170.0, -135.0],   # Right eye right corner (landmark 263)
        [-150.0, -150.0, -125.0], # Left Mouth corner (landmark 61)
        [150.0, -150.0, -125.0],  # Right mouth corner (landmark 291)
    ],
    dtype=np.float64,
)

HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class FacialFeatureState:
    previous_landmarks: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    movement_energy: float = 0.0
    blink_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    is_blinking: bool = False
    blink_rate: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    mar: float = 0.0
    eyebrow_distance: float = 0.0
    facial_symmetry: float = 1.0
    head_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (pitch, yaw, roll) in degrees
    rvec: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    mean_velocity: float = 0.0
    mean_acceleration: float = 0.0


def normalized_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    center = pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts[33, :2] - pts[263, :2]) if len(pts) > 263 else np.std(pts[:, :2])
    return (pts - center) / max(float(scale), 1e-6)


def compute_ear(pts: np.ndarray, eye_indices: list[int]) -> float:
    """Compute Eye Aspect Ratio (EAR) given 6 landmark coordinates [p0, p1, p2, p3, p4, p5]."""
    if max(eye_indices) >= len(pts):
        return 0.0
    p0 = pts[eye_indices[0], :2]
    p1 = pts[eye_indices[1], :2]
    p2 = pts[eye_indices[2], :2]
    p3 = pts[eye_indices[3], :2]
    p4 = pts[eye_indices[4], :2]
    p5 = pts[eye_indices[5], :2]

    v1 = np.linalg.norm(p1 - p5)
    v2 = np.linalg.norm(p2 - p4)
    h = np.linalg.norm(p0 - p3)
    if h <= 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def compute_mar(pts: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR)."""
    # 61: left corner, 291: right corner, 13: upper lip, 14: lower lip
    # 82, 87: left inner, 312, 317: right inner
    indices = [61, 291, 13, 14, 82, 87, 312, 317]
    if max(indices) >= len(pts):
        return 0.0
    h = np.linalg.norm(pts[61, :2] - pts[291, :2])
    if h <= 1e-6:
        return 0.0
    v1 = np.linalg.norm(pts[13, :2] - pts[14, :2])
    v2 = np.linalg.norm(pts[82, :2] - pts[87, :2])
    v3 = np.linalg.norm(pts[312, :2] - pts[317, :2])
    return float((v1 + v2 + v3) / (3.0 * h))


def compute_facial_symmetry(pts: np.ndarray) -> float:
    """Compute bilateral symmetry ratio comparing left and right facial landmark distances."""
    pairs = [(33, 263), (61, 291), (70, 300), (159, 386), (145, 374)]
    nose_idx = 1
    if max(max(a, b) for a, b in pairs) >= len(pts) or nose_idx >= len(pts):
        return 1.0
    nose = pts[nose_idx, :2]
    diffs = []
    for left, right in pairs:
        d_left = np.linalg.norm(pts[left, :2] - nose)
        d_right = np.linalg.norm(pts[right, :2] - nose)
        total = d_left + d_right
        if total > 1e-6:
            diffs.append(1.0 - abs(d_left - d_right) / total)
    return float(np.mean(diffs)) if diffs else 1.0


def estimate_head_pose(
    landmarks: np.ndarray,
    image_shape: tuple[int, int] = (480, 640),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Estimate 3D head pose (Pitch, Yaw, Roll in degrees) using cv2.solvePnP."""
    h, w = image_shape[:2]
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    if max(HEAD_POSE_LANDMARK_INDICES) >= len(pts):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        return rvec, tvec, (0.0, 0.0, 0.0)

    # 2D image points
    image_points = np.array(
        [
            [pts[1, 0] * w, pts[1, 1] * h],
            [pts[152, 0] * w, pts[152, 1] * h],
            [pts[33, 0] * w, pts[33, 1] * h],
            [pts[263, 0] * w, pts[263, 1] * h],
            [pts[61, 0] * w, pts[61, 1] * h],
            [pts[291, 0] * w, pts[291, 1] * h],
        ],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        CANONICAL_FACE_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), (0.0, 0.0, 0.0)

    rmat, _ = cv2.Rodrigues(rvec)
    # Extract Euler angles (pitch, yaw, roll)
    # Using cv2.RQDecomp3x3
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])  # Nodding up/down
    yaw = float(angles[1])    # Turning left/right
    roll = float(angles[2])   # Tilting side-to-side

    return rvec, tvec, (pitch, yaw, roll)


def geometric_features(landmarks: np.ndarray) -> np.ndarray:
    pts = normalized_landmarks(landmarks)
    pairs = {
        "inter_eye": (33, 263),
        "mouth_width": (61, 291),
        "mouth_open": (13, 14),
        "left_eye_open": (159, 145),
        "right_eye_open": (386, 374),
        "brow_distance": (70, 300),
        "jaw_open": (152, 13),
        "nose_mouth": (1, 13),
    }
    values = []
    for a, b in pairs.values():
        if max(a, b) < len(pts):
            values.append(np.linalg.norm(pts[a] - pts[b]))
        else:
            values.append(0.0)
    yaw_hint = pts[1, 0] if len(pts) > 1 else 0.0
    pitch_hint = pts[1, 1] if len(pts) > 1 else 0.0
    roll_hint = float(np.arctan2(*(pts[263, :2] - pts[33, :2])[::-1])) if len(pts) > 263 else 0.0
    return np.asarray(values + [yaw_hint, pitch_hint, roll_hint], dtype=np.float32)


def facial_dynamics(
    landmarks: np.ndarray,
    state: FacialFeatureState,
    alpha: float = 0.9,
    timestamp: float | None = None,
    image_shape: tuple[int, int] = (480, 640),
) -> tuple[np.ndarray, FacialFeatureState]:
    curr_time = timestamp if timestamp is not None else time.time()
    current = normalized_landmarks(landmarks).reshape(-1)
    previous = state.previous_landmarks if state.previous_landmarks is not None else current
    velocity = current - previous
    previous_velocity = state.previous_velocity if state.previous_velocity is not None else np.zeros_like(velocity)
    acceleration = velocity - previous_velocity
    magnitude = float(np.linalg.norm(velocity))
    energy = alpha * state.movement_energy + (1.0 - alpha) * magnitude

    # Detailed geometric measurements
    pts_raw = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    left_ear = compute_ear(pts_raw, [33, 160, 158, 133, 153, 144])
    right_ear = compute_ear(pts_raw, [362, 385, 387, 263, 373, 380])
    avg_ear = float((left_ear + right_ear) / 2.0)
    mar = compute_mar(pts_raw)
    brow_dist = float(np.linalg.norm(pts_raw[70, :2] - pts_raw[300, :2])) if len(pts_raw) > 300 else 0.0
    symmetry = compute_facial_symmetry(pts_raw)

    # Blink detection & blink rate over 60 seconds rolling window
    blink_timestamps = state.blink_timestamps
    is_blinking = state.is_blinking
    ear_threshold = 0.20
    if avg_ear < ear_threshold and not is_blinking:
        is_blinking = True
        blink_timestamps.append(curr_time)
    elif avg_ear >= ear_threshold and is_blinking:
        is_blinking = False

    # Purge blinks older than 60s
    cutoff = curr_time - 60.0
    while blink_timestamps and blink_timestamps[0] < cutoff:
        blink_timestamps.popleft()

    blink_rate = float(len(blink_timestamps))  # blinks per minute

    # Head pose estimation
    rvec, tvec, (pitch, yaw, roll) = estimate_head_pose(landmarks, image_shape)

    mean_vel = float(np.mean(np.abs(velocity)))
    mean_acc = float(np.mean(np.abs(acceleration)))

    next_state = FacialFeatureState(
        previous_landmarks=current,
        previous_velocity=velocity,
        movement_energy=energy,
        blink_timestamps=blink_timestamps,
        is_blinking=is_blinking,
        blink_rate=blink_rate,
        left_ear=left_ear,
        right_ear=right_ear,
        avg_ear=avg_ear,
        mar=mar,
        eyebrow_distance=brow_dist,
        facial_symmetry=symmetry,
        head_pose=(pitch, yaw, roll),
        rvec=rvec,
        tvec=tvec,
        mean_velocity=mean_vel,
        mean_acceleration=mean_acc,
    )
    dynamics = np.concatenate([velocity, acceleration, np.asarray([magnitude, energy], dtype=np.float32)])
    return dynamics.astype(np.float32), next_state


def expression_features(blendshapes: dict[str, float] | None) -> np.ndarray:
    names = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
    if not blendshapes:
        return np.zeros(len(names) + 1, dtype=np.float32)
    values = np.asarray([blendshapes.get(name, 0.0) for name in names], dtype=np.float32)
    intensity = float(values[1:].max(initial=0.0))
    return np.concatenate([values, np.asarray([intensity], dtype=np.float32)])


def extract_facial_feature_vector(
    landmarks: np.ndarray,
    state: FacialFeatureState,
    blendshapes: dict[str, float] | None = None,
    timestamp: float | None = None,
    image_shape: tuple[int, int] = (480, 640),
) -> tuple[np.ndarray, FacialFeatureState]:
    pts = normalized_landmarks(landmarks).reshape(-1)
    geom = geometric_features(landmarks)
    dyn, next_state = facial_dynamics(landmarks, state, timestamp=timestamp, image_shape=image_shape)
    expr = expression_features(blendshapes)
    return np.concatenate([pts, geom, dyn, expr]).astype(np.float32), next_state
