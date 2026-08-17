from __future__ import annotations

import numpy as np


ABLATIONS = {
    "landmarks_only": "normalized landmark coordinates",
    "landmarks_movement": "landmarks plus velocity and acceleration",
    "movement_expression": "movement features plus expression probabilities",
    "full_features": "complete facial feature vector",
    "full_features_transformer": "complete features with temporal transformer",
}


def select_feature_slice(features: np.ndarray, experiment: str, landmark_dim: int = 1404, geometry_dim: int = 11) -> np.ndarray:
    if experiment == "landmarks_only":
        return features[..., :landmark_dim]
    if experiment == "landmarks_movement":
        dynamics_start = landmark_dim + geometry_dim
        dynamics_end = dynamics_start + landmark_dim * 2 + 2
        return np.concatenate([features[..., :landmark_dim], features[..., dynamics_start:dynamics_end]], axis=-1)
    if experiment == "movement_expression":
        dynamics_start = landmark_dim + geometry_dim
        return features[..., dynamics_start:]
    if experiment in {"full_features", "full_features_transformer"}:
        return features
    raise ValueError(f"Unknown ablation experiment: {experiment}")
