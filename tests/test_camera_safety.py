from neurovision.realtime.camera import Camera, _release_all_captures, open_capture
from neurovision.config import DEFAULT_CONFIG
from neurovision.realtime import dashboard


def test_camera_reconnect_releases_every_handle(never_open_real_camera):
    camera = Camera(0)
    original = never_open_real_camera[-1]

    # The failed initial handle is released before the reconnection is attempted.
    camera.read()
    assert original.release.called
    assert len(never_open_real_camera) == 2

    replacement = never_open_real_camera[-1]
    camera.release()
    assert replacement.release.called


def test_shutdown_releases_registered_capture(never_open_real_camera):
    capture = open_capture(1)
    _release_all_captures()
    assert capture.release.called


def test_dashboard_releases_camera_if_tracker_startup_fails(monkeypatch, never_open_real_camera):
    def fail_tracker():
        raise RuntimeError("tracker initialization failed")

    monkeypatch.setattr(dashboard, "MediaPipeFaceTracker", fail_tracker)
    try:
        dashboard.run_dashboard(DEFAULT_CONFIG)
    except RuntimeError as exc:
        assert "tracker initialization failed" in str(exc)
    else:
        raise AssertionError("Expected tracker initialization failure")

    assert never_open_real_camera
    assert all(capture.release.called for capture in never_open_real_camera)


def test_dashboard_releases_camera_on_normal_exit(monkeypatch, never_open_real_camera):
    class Tracker:
        close_called = False

        def __init__(self):
            self.close_called = False

        def close(self):
            self.close_called = True

    tracker = Tracker()
    monkeypatch.setattr(dashboard, "MediaPipeFaceTracker", lambda: tracker)
    monkeypatch.setattr(dashboard, "EEGPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(dashboard.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(dashboard.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(dashboard.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(dashboard.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(dashboard.cv2, "waitKey", lambda *_: ord("q"))
    monkeypatch.setattr(dashboard, "_render_scientific_dashboard", lambda **kwargs: None)

    dashboard.run_dashboard(DEFAULT_CONFIG)

    assert tracker.close_called
    assert never_open_real_camera
    assert all(capture.release.called for capture in never_open_real_camera)
