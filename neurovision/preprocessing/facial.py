from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FacialFeatureState:
    previous_landmarks: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    movement_energy: float = 0.0


def normalized_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    center = pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts[33, :2] - pts[263, :2]) if len(pts) > 263 else np.std(pts[:, :2])
    return (pts - center) / max(float(scale), 1e-6)


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


def facial_dynamics(landmarks: np.ndarray, state: FacialFeatureState, alpha: float = 0.9) -> tuple[np.ndarray, FacialFeatureState]:
    current = normalized_landmarks(landmarks).reshape(-1)
    previous = state.previous_landmarks if state.previous_landmarks is not None else current
    velocity = current - previous
    previous_velocity = state.previous_velocity if state.previous_velocity is not None else np.zeros_like(velocity)
    acceleration = velocity - previous_velocity
    magnitude = float(np.linalg.norm(velocity))
    energy = alpha * state.movement_energy + (1.0 - alpha) * magnitude
    next_state = FacialFeatureState(current, velocity, energy)
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
) -> tuple[np.ndarray, FacialFeatureState]:
    pts = normalized_landmarks(landmarks).reshape(-1)
    geom = geometric_features(landmarks)
    dyn, next_state = facial_dynamics(landmarks, state)
    expr = expression_features(blendshapes)
    return np.concatenate([pts, geom, dyn, expr]).astype(np.float32), next_state
