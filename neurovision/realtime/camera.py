from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        self.capture.release()
