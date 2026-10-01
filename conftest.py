"""Safety fixtures and a timeout fallback for minimal development environments."""
from __future__ import annotations

import importlib.util
import signal
from unittest.mock import Mock

import cv2
import pytest


_HAS_PYTEST_TIMEOUT = importlib.util.find_spec("pytest_timeout") is not None


def pytest_addoption(parser):
    # requirements.txt installs pytest-timeout in normal environments. This
    # fallback keeps the same configured limit active if a developer runs tests
    # before installing project requirements.
    if not _HAS_PYTEST_TIMEOUT:
        parser.addini("timeout", "Per-test timeout in seconds (fallback).", default="60")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if _HAS_PYTEST_TIMEOUT or not hasattr(signal, "setitimer"):
        yield
        return

    seconds = float(item.config.getini("timeout"))
    if seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def timed_out(_signum, _frame):
        raise TimeoutError(f"Test exceeded the {seconds:g}s timeout")

    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


@pytest.fixture(autouse=True)
def never_open_real_camera(monkeypatch):
    """Replace every OpenCV capture constructor and fail on leaked mock handles."""
    captures = []

    def fake_video_capture(*args, **kwargs):
        capture = Mock(name="VideoCapture")
        capture.isOpened.return_value = False
        capture.read.return_value = (False, None)
        capture.get.return_value = 0
        capture.set.return_value = True
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    yield captures
    unreleased = [capture for capture in captures if not capture.release.called]
    assert not unreleased, f"Test leaked {len(unreleased)} mocked camera handle(s)."
