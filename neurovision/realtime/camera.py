from __future__ import annotations

import time
import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._last_reconnect_time = 0.0

    @property
    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> np.ndarray | None:
        if not self.capture.isOpened():
            now = time.time()
            if now - self._last_reconnect_time > 2.0:
                self._last_reconnect_time = now
                self.capture = cv2.VideoCapture(self.index)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.capture.isOpened():
                return None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
