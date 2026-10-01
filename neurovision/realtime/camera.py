from __future__ import annotations

import atexit
import threading
import time
import cv2
import numpy as np


_ACTIVE_CAPTURES: dict[int, object] = {}
_CAPTURE_LOCK = threading.RLock()


def open_capture(source: int | str):
    """Open and register a capture so it can be released on process shutdown."""
    capture = cv2.VideoCapture(source)
    with _CAPTURE_LOCK:
        _ACTIVE_CAPTURES[id(capture)] = capture
    return capture


def release_capture(capture) -> None:
    """Release one handle exactly once and remove it from the shutdown registry."""
    if capture is None:
        return
    with _CAPTURE_LOCK:
        registered = _ACTIVE_CAPTURES.pop(id(capture), None)
    if registered is not None:
        registered.release()


def _release_all_captures() -> None:
    with _CAPTURE_LOCK:
        captures = list(_ACTIVE_CAPTURES.values())
        _ACTIVE_CAPTURES.clear()
    for capture in captures:
        try:
            capture.release()
        except Exception:
            # Continue releasing remaining camera devices during interpreter exit.
            pass


atexit.register(_release_all_captures)


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = self._open()
        self._last_reconnect_time = 0.0

    def _open(self):
        capture = open_capture(self.index)
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except BaseException:
            release_capture(capture)
            raise
        return capture

    @property
    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        if not self.capture.isOpened():
            now = time.time()
            if now - self._last_reconnect_time > 2.0:
                self._last_reconnect_time = now
                release_capture(self.capture)
                self.capture = self._open()
            if not self.capture.isOpened():
                return None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        release_capture(self.capture)
        self.capture = None
