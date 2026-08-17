from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class FaceTrackingResult:
    landmarks: np.ndarray | None
    blendshapes: dict[str, float]
    tracking_quality: float
    status: str
    face_detected: bool
    face_count: int
    bounding_box: tuple[int, int, int, int] | None
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

    def process(self, frame: np.ndarray, rvec: np.ndarray | None = None, tvec: np.ndarray | None = None) -> FaceTrackingResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, int(time.time() * 1000))
        annotated = frame.copy()

        if not result.face_landmarks:
            return FaceTrackingResult(
                landmarks=None,
                blendshapes={},
                tracking_quality=0.0,
                status="FACE NOT DETECTED",
                face_detected=False,
                face_count=0,
                bounding_box=None,
                annotated_frame=annotated,
            )

        face_count = len(result.face_landmarks)
        face = result.face_landmarks[0]
        blendshapes = self._blendshapes(result)
        pts = np.asarray([(lm.x, lm.y, lm.z) for lm in face[:468]], dtype=np.float32)

        xs = pts[:, 0]
        ys = pts[:, 1]
        h, w = frame.shape[:2]

        x_min, x_max = int(np.clip(xs.min() * w, 0, w)), int(np.clip(xs.max() * w, 0, w))
        y_min, y_max = int(np.clip(ys.min() * h, 0, h)), int(np.clip(ys.max() * h, 0, h))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Quality metric based on facial coverage and landmark spread
        visible_area = max(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 0.0)
        quality = min(1.0, max(0.0, visible_area * 14.0))

        if quality < 0.35:
            status = "LOW TRACKING QUALITY"
        else:
            status = "TRACKING"

        # Draw clean, uncluttered visual aids
        self._draw_face_mesh(annotated, pts)
        self._draw_bounding_box(annotated, bbox, status == "TRACKING")

        if rvec is not None and tvec is not None:
            self._draw_head_pose_axes(annotated, pts, rvec, tvec)

        return FaceTrackingResult(
            landmarks=pts,
            blendshapes=blendshapes,
            tracking_quality=quality,
            status=status,
            face_detected=True,
            face_count=face_count,
            bounding_box=bbox,
            annotated_frame=annotated,
        )

    def close(self) -> None:
        self.landmarker.close()

    def _draw_face_mesh(self, frame: np.ndarray, pts: np.ndarray) -> None:
        height, width = frame.shape[:2]
        # Draw subtle landmark points for key facial features (eyes, eyebrows, nose, lips)
        key_indices = [
            # Eyes
            33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
            # Brows
            70, 63, 105, 66, 107, 300, 293, 334, 296, 336,
            # Nose
            1, 2, 98, 327,
            # Lips
            61, 291, 13, 14, 78, 308,
        ]
        for idx in key_indices:
            if idx < len(pts):
                cx, cy = int(pts[idx, 0] * width), int(pts[idx, 1] * height)
                cv2.circle(frame, (cx, cy), 2, (76, 205, 255), -1)

        # Subtle tessellation contour lines
        for connection in self.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            if hasattr(connection, "start"):
                start = connection.start
                end = connection.end
            else:
                start, end = connection
            if start >= len(pts) or end >= len(pts):
                continue
            p1 = (int(pts[start, 0] * width), int(pts[start, 1] * height))
            p2 = (int(pts[end, 0] * width), int(pts[end, 1] * height))
            cv2.line(frame, p1, p2, (38, 140, 195), 1, cv2.LINE_AA)

    def _draw_bounding_box(self, frame: np.ndarray, bbox: tuple[int, int, int, int], is_good: bool) -> None:
        x, y, w, h = bbox
        color = (52, 211, 153) if is_good else (245, 158, 11)  # Emerald or Amber
        # Draw corner brackets
        length = min(20, w // 4, h // 4)
        thickness = 2
        # Top-left
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def _draw_head_pose_axes(self, frame: np.ndarray, pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Projects 3D coordinate frame axes (Pitch/X=Red, Yaw/Y=Green, Roll/Z=Blue) from nose tip."""
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D axis points (length 150mm)
        axis_3d = np.float64([[150, 0, 0], [0, 150, 0], [0, 0, -150], [0, 0, 0]])
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

        if imgpts is not None and len(imgpts) == 4:
            p_nose = (int(pts[1, 0] * w), int(pts[1, 1] * h))
            p_x = (int(imgpts[0].ravel()[0]), int(imgpts[0].ravel()[1]))
            p_y = (int(imgpts[1].ravel()[0]), int(imgpts[1].ravel()[1]))
            p_z = (int(imgpts[2].ravel()[0]), int(imgpts[2].ravel()[1]))

            cv2.line(frame, p_nose, p_x, (0, 0, 255), 2, cv2.LINE_AA)    # X-axis (Pitch/Red)
            cv2.line(frame, p_nose, p_y, (0, 255, 0), 2, cv2.LINE_AA)    # Y-axis (Yaw/Green)
            cv2.line(frame, p_nose, p_z, (255, 120, 0), 2, cv2.LINE_AA)  # Z-axis (Roll/Blue)

    @staticmethod
    def _blendshapes(result) -> dict[str, float]:
        if not result.face_blendshapes:
            return {}
        values = {}
        for category in result.face_blendshapes[0]:
            name = category.category_name or category.display_name
            values[name] = float(category.score)
        return values
