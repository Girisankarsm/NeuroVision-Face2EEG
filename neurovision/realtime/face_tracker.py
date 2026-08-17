from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceTrackingResult:
    landmarks: np.ndarray | None
    tracking_quality: float
    status: str
    annotated_frame: np.ndarray


class MediaPipeFaceTracker:
    def __init__(self, max_num_faces: int = 1) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for live face tracking.") from exc
        self.mp = mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def process(self, frame: np.ndarray) -> FaceTrackingResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        annotated = frame.copy()
        if not result.multi_face_landmarks:
            return FaceTrackingResult(None, 0.0, "FACE NOT DETECTED", annotated)
        face = result.multi_face_landmarks[0]
        height, width = frame.shape[:2]
        pts = np.asarray([(lm.x, lm.y, lm.z) for lm in face.landmark], dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]
        visible_area = max(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 0.0)
        quality = min(1.0, visible_area * 12.0)
        self.drawer.draw_landmarks(
            image=annotated,
            landmark_list=face,
            connections=self.mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style(),
        )
        if quality < 0.35:
            status = "LOW TRACKING QUALITY"
        else:
            status = "TRACKING"
        return FaceTrackingResult(pts, quality, status, annotated)

    def close(self) -> None:
        self.mesh.close()
