import numpy as np

from neurovision.preprocessing.facial import (
    FacialFeatureState,
    extract_compact_features,
    COMPACT_FEATURE_DIM
)

def test_compact_feature_extraction():
    landmarks = np.zeros((468, 3), dtype=np.float32)
    # create some dummy data
    landmarks[:, 0] = np.linspace(0.1, 0.9, 468)
    landmarks[:, 1] = np.linspace(0.2, 0.8, 468)

    state = FacialFeatureState()
    features, state = extract_compact_features(landmarks, state, blendshapes=None)

    assert features.shape == (COMPACT_FEATURE_DIM,)
    assert isinstance(features, np.ndarray)

    # Check that state is updated
    assert state.left_ear >= 0.0
    assert state.right_ear >= 0.0
    assert len(state.head_pose) == 3
