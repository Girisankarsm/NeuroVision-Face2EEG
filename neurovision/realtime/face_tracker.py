from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class FaceTrackingResult:
    landmarks: np.ndarray | None
    tracking_quality: float
    status: str
    annotated_frame: np.ndarray


class MediaPipeFaceTracker:
    def __init__(self, max_num_faces: int = 1, model_path: str | Path = "neurovision/models/mediapipe/face_landmarker.task") -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for live face tracking.") from exc
        self.mp = mp
        self.vision = vision
        model_path = Path(model_path)
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe face landmarker model not found at {model_path}. "
                "Download face_landmarker.task before running the live dashboard."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> FaceTrackingResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, int(time.time() * 1000))
        annotated = frame.copy()
        if not result.face_landmarks:
            return FaceTrackingResult(None, 0.0, "FACE NOT DETECTED", annotated)
        face = result.face_landmarks[0]
        pts = np.asarray([(lm.x, lm.y, lm.z) for lm in face[:468]], dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]
        visible_area = max(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 0.0)
        quality = min(1.0, visible_area * 12.0)
        self._draw_landmarks(annotated, pts)
        if quality < 0.35:
            status = "LOW TRACKING QUALITY"
        else:
            status = "TRACKING"
        return FaceTrackingResult(pts, quality, status, annotated)

    def close(self) -> None:
        self.landmarker.close()

    def _draw_landmarks(self, frame: np.ndarray, pts: np.ndarray) -> None:
        height, width = frame.shape[:2]
        for lm in pts:
            cv2.circle(frame, (int(lm[0] * width), int(lm[1] * height)), 1, (80, 210, 255), -1)
        for connection in self.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            if hasattr(connection, "start"):
                start = connection.start
                end = connection.end
            else:
                start, end = connection
            if start >= len(pts) or end >= len(pts):
                continue
            p1 = (int(pts[start, 0] * width), int(pts[start, 1] * height))
            p2 = (int(pts[end, 0] * width), int(pts[end, 1] * height))
            cv2.line(frame, p1, p2, (45, 120, 170), 1)
