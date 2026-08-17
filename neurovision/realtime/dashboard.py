from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from neurovision.preprocessing.facial import FacialFeatureState, extract_facial_feature_vector
from neurovision.realtime.camera import Camera
from neurovision.realtime.eeg_prediction import band_percentages
from neurovision.realtime.face_tracker import MediaPipeFaceTracker
from neurovision.realtime.feature_buffer import TemporalFeatureBuffer
from neurovision.realtime.inference import EEGPredictor


def run_dashboard(config: dict, checkpoint: str | None, camera_index: int = 0) -> None:
    camera = Camera(camera_index)
    tracker = MediaPipeFaceTracker()
    predictor = EEGPredictor(checkpoint, fallback_config=config)
    buffer = TemporalFeatureBuffer(int(config["sequence_length"]), int(config["model"]["feature_dim"]))
    state = FacialFeatureState()
    waveform_history: deque[float] = deque(maxlen=512)
    band_values = np.zeros(5, dtype=np.float32)
    confidence = 0.0
    frame_count = 0
    last_fps_time = time.perf_counter()
    fps = 0.0
    latency_ms = 0.0

    try:
        while True:
            started = time.perf_counter()
            frame = camera.read()
            if frame is None:
                break
            tracking = tracker.process(frame)
            feature = None
            if tracking.landmarks is not None:
                feature, state = extract_facial_feature_vector(tracking.landmarks, state)
                buffer.append(feature)
                if buffer.ready():
                    pred_started = time.perf_counter()
                    prediction = predictor.predict(buffer.tensor(), mc_dropout_passes=3)
                    latency_ms = (time.perf_counter() - pred_started) * 1000.0
                    if prediction is not None:
                        waveform_history.extend(prediction.waveform.tolist())
                        band_values = prediction.band_power
                        confidence = prediction.confidence

            frame_count += 1
            elapsed = time.perf_counter() - last_fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = time.perf_counter()

            panel = _render_panel(
                tracking.annotated_frame,
                waveform_history,
                band_values,
                confidence,
                buffer.fill_ratio,
                tracking.status,
                predictor.status,
                fps,
                latency_ms,
            )
            cv2.imshow("NEUROVISION - AI-PREDICTED EEG", panel)
            if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                break
            _ = feature
            _ = started
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def _render_panel(
    camera_frame: np.ndarray,
    waveform_history: deque[float],
    band_values: np.ndarray,
    confidence: float,
    buffer_fill: float,
    face_status: str,
    model_status: str,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    height = 900
    width = 1400
    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    cam = cv2.resize(camera_frame, (620, 420))
    panel[95:515, 40:660] = cam
    _text(panel, "NEUROVISION", (40, 48), 1.2, (235, 240, 245), 2)
    _text(panel, "Camera-based neural activity prediction", (40, 78), 0.65, (170, 180, 190), 1)
    _text(panel, "AI-PREDICTED EEG", (720, 120), 0.8, (120, 200, 255), 2)
    _text(panel, "Camera-derived model estimate, not physiological EEG measurement", (720, 150), 0.5, (170, 180, 190), 1)
    _waveform(panel, waveform_history, (720, 180, 620, 210), model_status)
    _bars(panel, "FACIAL DYNAMICS", [("Feature buffer", buffer_fill), ("Tracking quality", 1.0 if face_status == "TRACKING" else 0.25)], 40, 570)
    percentages = band_percentages(band_values)
    _bars(
        panel,
        "PREDICTED NEURAL OSCILLATIONS",
        [(name.title(), float(value)) for name, value in zip(["delta", "theta", "alpha", "beta", "gamma"], percentages)],
        720,
        460,
    )
    _text(panel, f"Face: {face_status}", (40, 540), 0.65, (220, 220, 220), 1)
    _text(panel, f"Prediction confidence: {confidence * 100:.0f}%", (720, 720), 0.7, (220, 220, 220), 1)
    if confidence and confidence < 0.45:
        _text(panel, "LOW-CONFIDENCE PREDICTION", (720, 750), 0.7, (80, 180, 255), 2)
    _text(panel, f"FPS: {fps:.1f}", (40, 830), 0.65, (190, 200, 210), 1)
    _text(panel, f"Inference: {latency_ms:.1f} ms", (180, 830), 0.65, (190, 200, 210), 1)
    _text(panel, f"Model: {model_status}", (420, 830), 0.65, (190, 200, 210), 1)
    _text(panel, "Mode: CAMERA-BASED PREDICTION", (720, 830), 0.65, (190, 200, 210), 1)
    return panel


def _text(panel: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int) -> None:
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _waveform(panel: np.ndarray, history: deque[float], rect: tuple[int, int, int, int], model_status: str) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 52, 60), 1)
    if model_status != "READY":
        _text(panel, "MODEL NOT LOADED", (x + 170, y + 115), 0.9, (100, 190, 255), 2)
        return
    if len(history) < 2:
        _text(panel, "MODEL NOT TRAINED OR BUFFER FILLING", (x + 85, y + 115), 0.65, (100, 190, 255), 2)
        return
    values = np.asarray(history, dtype=np.float32)
    values = values[-w:]
    values = values / (np.max(np.abs(values)) + 1e-6)
    pts = []
    for i, value in enumerate(values):
        px = x + int(i * w / max(len(values) - 1, 1))
        py = y + h // 2 - int(value * h * 0.42)
        pts.append((px, py))
    cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, (80, 210, 255), 2)
    _text(panel, "Amplitude (normalized)", (x + 8, y + 24), 0.45, (160, 170, 180), 1)
    _text(panel, "Time ->", (x + w - 88, y + h - 10), 0.45, (160, 170, 180), 1)


def _bars(panel: np.ndarray, title: str, values: list[tuple[str, float]], x: int, y: int) -> None:
    _text(panel, title, (x, y), 0.72, (225, 230, 235), 2)
    for idx, (label, value) in enumerate(values):
        yy = y + 42 + idx * 42
        value = float(np.clip(value, 0.0, 1.0))
        _text(panel, label, (x, yy), 0.55, (190, 200, 210), 1)
        cv2.rectangle(panel, (x + 170, yy - 18), (x + 420, yy), (55, 60, 68), -1)
        cv2.rectangle(panel, (x + 170, yy - 18), (x + 170 + int(250 * value), yy), (80, 170, 255), -1)
        _text(panel, f"{value * 100:.0f}%", (x + 440, yy), 0.5, (190, 200, 210), 1)
