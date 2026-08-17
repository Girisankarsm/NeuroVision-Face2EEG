from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np

from neurovision.preprocessing.facial import FacialFeatureState, extract_facial_feature_vector
from neurovision.realtime.camera import Camera
from neurovision.realtime.eeg_prediction import band_percentages
from neurovision.realtime.face_tracker import FaceTrackingResult, MediaPipeFaceTracker
from neurovision.realtime.feature_buffer import TemporalFeatureBuffer
from neurovision.realtime.inference import EEGPredictor, Prediction


@dataclass
class DashboardStats:
    frames_processed: int = 0
    frames_dropped: int = 0
    fps: float = 0.0
    frame_latency_ms: float = 0.0
    feat_latency_ms: float = 0.0
    infer_latency_ms: float = 0.0
    start_time: float = 0.0
    face_count: int = 0
    tracking_quality: float = 0.0
    face_status: str = "INITIALIZING"
    predicted_rms: float = 0.0
    predicted_peak_to_peak: float = 0.0
    dominant_band: str = "N/A"
    confidence_display: str = "Confidence: N/A"
    uncertainty_display: str = "Uncertainty: N/A"


def _create_camera_fallback_frame(width: int = 670, height: int = 400, message: str = "CAMERA OFFLINE / PERMISSION REQUIRED") -> np.ndarray:
    frame = np.full((height, width, 3), (20, 24, 30), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (38, 48, 60), 1)
    _text(frame, "● CAMERA OFFLINE", (width // 2 - 130, height // 2 - 20), 0.68, (240, 140, 80), 2)
    _text(frame, message, (width // 2 - 190, height // 2 + 18), 0.44, (160, 175, 190), 1)
    _text(frame, "Check camera index or grant Terminal Camera permissions.", (width // 2 - 210, height // 2 + 48), 0.40, (120, 135, 150), 1)
    return frame


def run_dashboard(
    config: dict,
    checkpoint: str | None = None,
    camera_index: int = 0,
    mode: str = "research",
    validation_eeg: np.ndarray | None = None,
) -> None:
    camera = Camera(camera_index)
    tracker = MediaPipeFaceTracker()
    predictor = EEGPredictor(checkpoint, fallback_config=config)
    seq_len = int(config.get("sequence_length", 64))
    feat_dim = int(config.get("model", {}).get("feature_dim", 4233))
    buffer = TemporalFeatureBuffer(seq_len, feat_dim)
    state = FacialFeatureState()

    waveform_history: deque[float] = deque(maxlen=512)
    band_history: dict[str, deque[float]] = {
        "delta": deque(maxlen=60),
        "theta": deque(maxlen=60),
        "alpha": deque(maxlen=60),
        "beta": deque(maxlen=60),
        "gamma": deque(maxlen=60),
    }
    band_values = np.zeros(5, dtype=np.float32)

    stats = DashboardStats(start_time=time.time())
    fps_counter = 0
    last_fps_time = time.perf_counter()
    window_name = f"NEUROVISION — Real-Time Research Instrument [{mode.upper()}]"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 960)

    try:
        while True:
            loop_start = time.perf_counter()
            frame = camera.read()

            if frame is None:
                stats.face_status = "CAMERA ERROR"
                stats.face_count = 0
                stats.tracking_quality = 0.0
                annotated_camera = _create_camera_fallback_frame(670, 400)
            else:
                stats.frames_processed += 1
                h, w = frame.shape[:2]

                # Face Tracking & Geometry
                tracking = tracker.process(frame, rvec=state.rvec, tvec=state.tvec)
                stats.face_status = tracking.status
                stats.face_count = tracking.face_count
                stats.tracking_quality = tracking.tracking_quality
                annotated_camera = tracking.annotated_frame

                if tracking.landmarks is not None:
                    feat_start = time.perf_counter()
                    feature, state = extract_facial_feature_vector(
                        tracking.landmarks,
                        state,
                        tracking.blendshapes,
                        timestamp=time.time(),
                        image_shape=(h, w),
                    )
                    stats.feat_latency_ms = (time.perf_counter() - feat_start) * 1000.0
                    buffer.append(feature, timestamp=time.time())
                    stats.frames_dropped = buffer.dropped_frames

                    # Run EEG prediction only if buffer is full and checkpoint is loaded
                    if buffer.ready() and predictor.status == "READY":
                        pred_start = time.perf_counter()
                        prediction = predictor.predict(buffer.tensor(), mc_dropout_passes=3)
                        stats.infer_latency_ms = (time.perf_counter() - pred_start) * 1000.0

                        if prediction is not None:
                            waveform_history.extend(prediction.waveform.tolist())
                            band_values = prediction.band_power
                            stats.predicted_rms = float(np.sqrt(np.mean(np.square(prediction.waveform))))
                            stats.predicted_peak_to_peak = float(np.ptp(prediction.waveform))
                            stats.dominant_band = _dominant_band(band_values)

                            for idx, name in enumerate(["delta", "theta", "alpha", "beta", "gamma"]):
                                band_history[name].append(float(band_values[idx]))

                            if prediction.confidence is not None:
                                stats.confidence_display = f"Prediction Confidence: {prediction.confidence * 100:.0f}%"
                            else:
                                stats.confidence_display = "Confidence: N/A"

                            if prediction.uncertainty is not None and len(prediction.uncertainty) > 0:
                                stats.uncertainty_display = f"Uncertainty: ±{float(prediction.uncertainty[0]):.3f}"
                            else:
                                stats.uncertainty_display = "Uncertainty: N/A"
                    else:
                        stats.infer_latency_ms = 0.0
                        stats.confidence_display = "Confidence: N/A"
                        stats.uncertainty_display = "Uncertainty: N/A"
                else:
                    stats.feat_latency_ms = 0.0
                    stats.infer_latency_ms = 0.0

            # Calculate FPS
            fps_counter += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 0.5:
                stats.fps = fps_counter / elapsed
                fps_counter = 0
                last_fps_time = now

            stats.frame_latency_ms = (now - loop_start) * 1000.0

            # Render complete scientific panel
            panel = _render_scientific_dashboard(
                camera_frame=annotated_camera,
                waveform_history=waveform_history,
                band_values=band_values,
                band_history=band_history,
                state=state,
                stats=stats,
                predictor=predictor,
                buffer=buffer,
                mode=mode,
                validation_eeg=validation_eeg,
            )

            cv2.imshow(window_name, panel)
            wait_time = 1 if frame is not None else 30
            key = cv2.waitKey(wait_time) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def _render_scientific_dashboard(
    camera_frame: np.ndarray,
    waveform_history: deque[float],
    band_values: np.ndarray,
    band_history: dict[str, deque[float]],
    state: FacialFeatureState,
    stats: DashboardStats,
    predictor: EEGPredictor,
    buffer: TemporalFeatureBuffer,
    mode: str,
    validation_eeg: np.ndarray | None = None,
) -> np.ndarray:
    height = 960
    width = 1600
    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

    # 1. Top Status Bar
    _draw_top_bar(panel, stats, predictor, mode)

    # 2. Main Camera Panel (Left Top)
    cam_h, cam_w = 400, 670
    cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))
    panel[88 : 88 + cam_h, 30 : 30 + cam_w] = cam_resized
    _draw_panel_frame(panel, (30, 88, cam_w, cam_h), "LIVE CAMERA & FACIAL GEOMETRY", tag=f"FACE: {stats.face_status}")

    # 3. Facial Dynamics & Kinematics Panel (Left Bottom)
    _draw_facial_dynamics_panel(panel, (30, 502, cam_w, 400), state, stats)

    # 4. AI-Predicted EEG Waveform Panel (Right Top)
    _draw_eeg_waveform_panel(panel, (720, 88, 850, 310), waveform_history, predictor, buffer, stats, mode, validation_eeg)

    # 5. Predicted Neural Oscillations / Band Power Panel (Right Middle)
    _draw_neural_oscillations_panel(panel, (720, 410, 850, 240), band_values, band_history, predictor)

    # 6. Live System Telemetry & Model Status Panel (Right Bottom)
    _draw_telemetry_panel(panel, (720, 662, 850, 240), stats, predictor, buffer, mode)

    # 7. Bottom Scientific Safety Bar
    _draw_bottom_safety_bar(panel)

    return panel


def _text(
    panel: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_panel_frame(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    tag: str | None = None,
) -> None:
    x, y, w, h = rect
    # Outer border
    cv2.rectangle(panel, (x, y), (x + w, y + h), (44, 52, 62), 1)
    # Header bar
    cv2.rectangle(panel, (x, y), (x + w, y + 32), (28, 34, 42), -1)
    cv2.line(panel, (x, y + 32), (x + w, y + 32), (50, 60, 72), 1)
    # Title
    _text(panel, title, (x + 14, y + 22), 0.52, (230, 238, 245), 1)
    if tag:
        _text(panel, tag, (x + w - 12 * len(tag) - 10, y + 22), 0.44, (120, 200, 255), 1)


def _draw_top_bar(panel: np.ndarray, stats: DashboardStats, predictor: EEGPredictor, mode: str) -> None:
    # Bar background
    cv2.rectangle(panel, (0, 0), (1600, 72), (12, 14, 18), -1)
    cv2.line(panel, (0, 72), (1600, 72), (36, 44, 54), 1)

    # Pulsing live indicator
    pulse = int(128 + 127 * np.sin(time.time() * 4.0))
    cv2.circle(panel, (36, 36), 6, (0, pulse, 0), -1)
    cv2.circle(panel, (36, 36), 8, (0, 255, 0), 1)
    _text(panel, "LIVE", (52, 42), 0.62, (0, 255, 120), 2)

    # Title & Subtitle
    _text(panel, "NEUROVISION", (130, 34), 0.85, (250, 252, 255), 2)
    _text(panel, "RESEARCH INSTRUMENT", (130, 56), 0.40, (130, 145, 160), 1)

    # Telemetry metrics in status bar
    fps_color = (80, 220, 140) if stats.fps >= 24 else (240, 180, 50)
    _text(panel, f"{stats.fps:.1f} FPS", (430, 42), 0.58, fps_color, 1)
    _text(panel, f"{stats.frame_latency_ms:.1f} ms", (550, 42), 0.58, (200, 212, 224), 1)

    # Tracking Badge
    track_good = stats.face_status == "TRACKING"
    track_bg = (30, 80, 48) if track_good else (70, 40, 30)
    track_fg = (100, 255, 160) if track_good else (255, 140, 100)
    cv2.rectangle(panel, (680, 22), (860, 52), track_bg, -1)
    cv2.rectangle(panel, (680, 22), (860, 52), track_fg, 1)
    _text(panel, stats.face_status[:16], (692, 42), 0.44, track_fg, 1)

    # Model Badge
    model_good = predictor.status == "READY"
    model_bg = (24, 70, 95) if model_good else (50, 50, 58)
    model_fg = (90, 210, 255) if model_good else (170, 180, 190)
    cv2.rectangle(panel, (880, 22), (1070, 52), model_bg, -1)
    cv2.rectangle(panel, (880, 22), (1070, 52), model_fg, 1)
    _text(panel, predictor.status[:18], (892, 42), 0.44, model_fg, 1)

    # Mode Selector Tabs
    tabs = [("DEMO", mode == "demo"), ("RESEARCH", mode == "research"), ("VALIDATION", mode == "validation")]
    x_tab = 1130
    for label, active in tabs:
        t_bg = (45, 120, 195) if active else (26, 32, 40)
        t_fg = (255, 255, 255) if active else (140, 150, 160)
        cv2.rectangle(panel, (x_tab, 22), (x_tab + 130, 52), t_bg, -1)
        cv2.rectangle(panel, (x_tab, 22), (x_tab + 130, 52), (55, 68, 84), 1)
        _text(panel, label, (x_tab + 20, 42), 0.46, t_fg, 1)
        x_tab += 140


def _draw_facial_dynamics_panel(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    state: FacialFeatureState,
    stats: DashboardStats,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (22, 26, 32), -1)
    _draw_panel_frame(panel, rect, "FACIAL DYNAMICS & KINEMATICS", tag="MEASURED PHYSIOLOGICAL SIGNALS")

    # Metric Cards Grid (2 columns x 3 rows)
    cards = [
        ("Blink Rate", f"{state.blink_rate:.0f} / min", f"Left EAR: {state.left_ear:.2f} | Right: {state.right_ear:.2f}"),
        ("Mouth Openness (MAR)", f"{state.mar:.3f}", "Norm. lip aperture ratio"),
        ("Head Pose (Euler)", f"Y: {state.head_pose[1]:+.1f}° | P: {state.head_pose[0]:+.1f}° | R: {state.head_pose[2]:+.1f}°", "3D SolvePnP canonical pose"),
        ("Kinetic Energy", f"{state.movement_energy:.4f}", "Exp. smoothed landmark motion"),
        ("Mean Velocity / Accel", f"V: {state.mean_velocity:.3f} | A: {state.mean_acceleration:.3f}", "Kinematic 1st/2nd derivatives"),
        ("Bilateral Symmetry", f"{state.facial_symmetry * 100:.1f}%", "Mirror landmark congruence"),
    ]

    card_w = (w - 36) // 2
    card_h = 56
    for idx, (label, val, sub) in enumerate(cards):
        cx = x + 12 + (idx % 2) * (card_w + 12)
        cy = y + 44 + (idx // 2) * (card_h + 10)
        cv2.rectangle(panel, (cx, cy), (cx + card_w, cy + card_h), (28, 34, 42), -1)
        cv2.rectangle(panel, (cx, cy), (cx + card_w, cy + card_h), (42, 50, 60), 1)
        _text(panel, label, (cx + 10, cy + 18), 0.38, (140, 155, 170), 1)
        _text(panel, val, (cx + 10, cy + 38), 0.52, (240, 246, 252), 1)
        _text(panel, sub, (cx + 10, cy + 50), 0.32, (100, 115, 130), 1)

    # Action Unit / Blendshape Intensities Bar Chart
    au_y = y + 252
    _text(panel, "ACTION UNIT / BLENDSHAPE INTENSITIES (NOT PSYCHOLOGICAL TRUTH)", (x + 14, au_y), 0.38, (140, 155, 170), 1)

    au_items = [
        ("Eye Blink Left", state.left_ear < 0.20, max(0.0, 1.0 - state.left_ear / 0.35)),
        ("Eye Blink Right", state.right_ear < 0.20, max(0.0, 1.0 - state.right_ear / 0.35)),
        ("Jaw Open", state.mar > 0.15, min(1.0, state.mar / 0.40)),
        ("Eyebrow Disp.", state.eyebrow_distance > 0.0, min(1.0, state.eyebrow_distance / 2.0)),
        ("Movement Energy", True, min(1.0, state.movement_energy * 25.0)),
    ]

    for i, (au_name, _, intensity) in enumerate(au_items):
        by = au_y + 20 + i * 24
        _text(panel, au_name, (x + 14, by + 12), 0.40, (190, 202, 214), 1)
        bar_x = x + 160
        bar_w = w - 240
        val_norm = float(np.clip(intensity, 0.0, 1.0))
        cv2.rectangle(panel, (bar_x, by), (bar_x + bar_w, by + 14), (32, 38, 48), -1)
        bar_fill = int(bar_w * val_norm)
        if bar_fill > 0:
            cv2.rectangle(panel, (bar_x, by), (bar_x + bar_fill, by + 14), (64, 160, 235), -1)
        _text(panel, f"{val_norm * 100:.0f}%", (bar_x + bar_w + 10, by + 12), 0.38, (210, 220, 230), 1)


def _draw_eeg_waveform_panel(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    waveform_history: deque[float],
    predictor: EEGPredictor,
    buffer: TemporalFeatureBuffer,
    stats: DashboardStats,
    mode: str,
    validation_eeg: np.ndarray | None = None,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (20, 24, 30), -1)
    _draw_panel_frame(panel, rect, "AI-PREDICTED EEG WAVEFORM", tag="ESTIMATED NEURAL OSCILLATION")

    graph_x = x + 16
    graph_y = y + 46
    graph_w = w - 32
    graph_h = h - 62

    # Graph canvas background
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (14, 16, 20), -1)
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (40, 48, 58), 1)

    # Draw grid lines
    for frac in [0.25, 0.5, 0.75]:
        gy = graph_y + int(graph_h * frac)
        cv2.line(panel, (graph_x, gy), (graph_x + graph_w, gy), (26, 32, 40), 1)
    # Zero line
    zero_y = graph_y + graph_h // 2
    cv2.line(panel, (graph_x, zero_y), (graph_x + graph_w, zero_y), (48, 58, 70), 1)

    if predictor.status != "READY":
        # Professional NO FAKE EEG Placeholder
        cv2.rectangle(panel, (graph_x + 40, graph_y + 30), (graph_x + graph_w - 40, graph_y + graph_h - 30), (22, 28, 36), -1)
        cv2.rectangle(panel, (graph_x + 40, graph_y + 30), (graph_x + graph_w - 40, graph_y + graph_h - 30), (55, 70, 88), 1)

        _text(panel, "● EEG MODEL NOT LOADED", (graph_x + 60, graph_y + 68), 0.72, (108, 196, 255), 2)
        _text(panel, "Live facial dynamics and kinematic features are available.", (graph_x + 60, graph_y + 102), 0.48, (210, 220, 230), 1)
        _text(panel, "EEG prediction requires a valid trained checkpoint. No synthetic EEG is shown.", (graph_x + 60, graph_y + 128), 0.44, (150, 165, 180), 1)

        # Buffer readiness progress bar
        buf_len = buffer.current_length
        seq_len = buffer.sequence_length
        buf_ratio = buffer.fill_ratio
        ready_text = "BUFFER READY — MODEL CHECKPOINT REQUIRED" if buf_ratio >= 1.0 else f"BUFFERING: {buf_len} / {seq_len} frames"

        _text(panel, ready_text, (graph_x + 60, graph_y + 168), 0.46, (240, 200, 80) if buf_ratio < 1.0 else (80, 220, 140), 1)
        bbar_x = graph_x + 60
        bbar_y = graph_y + 182
        bbar_w = graph_w - 180
        cv2.rectangle(panel, (bbar_x, bbar_y), (bbar_x + bbar_w, bbar_y + 14), (32, 38, 48), -1)
        fill_w = int(bbar_w * buf_ratio)
        if fill_w > 0:
            cv2.rectangle(panel, (bbar_x, bbar_y), (bbar_x + fill_w, bbar_y + 14), (52, 168, 235), -1)
        return

    if len(waveform_history) < 2:
        _text(panel, "BUFFER ACCUMULATING TEMPORAL FRAMES...", (graph_x + 120, graph_y + graph_h // 2), 0.58, (120, 190, 240), 1)
        return

    # Render REAL predicted waveform
    raw_vals = np.asarray(waveform_history, dtype=np.float32)[-graph_w:]
    max_amp = max(float(np.max(np.abs(raw_vals))), 1e-4)
    norm_vals = raw_vals / max_amp

    pts = []
    for i, val in enumerate(norm_vals):
        px = graph_x + int(i * graph_w / max(len(norm_vals) - 1, 1))
        py = zero_y - int(val * (graph_h * 0.40))
        pts.append((px, py))

    cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, (76, 205, 255), 2, cv2.LINE_AA)

    # Calibration scale and live telemetry labels on waveform
    _text(panel, f"Peak: ±{max_amp:.2f} µV", (graph_x + 10, graph_y + 20), 0.40, (140, 155, 170), 1)
    _text(panel, f"RMS: {stats.predicted_rms:.3f} | P2P: {stats.predicted_peak_to_peak:.3f}", (graph_x + 180, graph_y + 20), 0.40, (190, 205, 220), 1)
    _text(panel, stats.confidence_display, (graph_x + graph_w - 240, graph_y + 20), 0.40, (80, 220, 140), 1)
    _text(panel, "Time ->", (graph_x + graph_w - 70, graph_y + graph_h - 10), 0.38, (120, 135, 150), 1)


def _draw_neural_oscillations_panel(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    band_values: np.ndarray,
    band_history: dict[str, deque[float]],
    predictor: EEGPredictor,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (20, 24, 30), -1)
    _draw_panel_frame(panel, rect, "PREDICTED NEURAL OSCILLATIONS (BAND POWER)", tag="SPECTRAL DECOMPOSITION")

    names = ["Delta (0.5-4Hz)", "Theta (4-8Hz)", "Alpha (8-13Hz)", "Beta (13-30Hz)", "Gamma (30-100Hz)"]
    band_keys = ["delta", "theta", "alpha", "beta", "gamma"]

    if predictor.status != "READY":
        # Clean inactive state
        for idx, name in enumerate(names):
            by = y + 48 + idx * 36
            _text(panel, name, (x + 18, by + 14), 0.42, (130, 142, 155), 1)
            cv2.rectangle(panel, (x + 220, by), (x + 600, by + 18), (28, 34, 42), -1)
            _text(panel, "0.0% (Awaiting Checkpoint)", (x + 616, by + 14), 0.40, (100, 115, 130), 1)
        return

    percentages = band_percentages(band_values)
    for idx, (name, key) in enumerate(zip(names, band_keys)):
        by = y + 48 + idx * 36
        _text(panel, name, (x + 18, by + 14), 0.42, (200, 212, 224), 1)
        bar_w = 380
        val = float(percentages[idx])
        cv2.rectangle(panel, (x + 220, by), (x + 220 + bar_w, by + 18), (28, 34, 42), -1)
        fill = int(bar_w * val)
        if fill > 0:
            cv2.rectangle(panel, (x + 220, by), (x + 220 + fill, by + 18), (64, 168, 240), -1)
        _text(panel, f"{val * 100:.1f}%", (x + 220 + bar_w + 14, by + 14), 0.42, (230, 240, 250), 1)

        # Mini rolling sparkline
        hist = band_history[key]
        if len(hist) > 2:
            sp_x = x + 690
            sp_w = 130
            sp_h = 18
            cv2.rectangle(panel, (sp_x, by), (sp_x + sp_w, by + sp_h), (24, 28, 34), -1)
            arr = np.asarray(hist, dtype=np.float32)
            span = max(float(np.ptp(arr)), 1e-4)
            s_pts = []
            for si, sv in enumerate(arr):
                sx = sp_x + int(si * sp_w / max(len(arr) - 1, 1))
                sy = by + sp_h - int(((sv - arr.min()) / span) * sp_h)
                s_pts.append((sx, sy))
            cv2.polylines(panel, [np.asarray(s_pts, dtype=np.int32)], False, (100, 210, 255), 1)


def _draw_telemetry_panel(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    stats: DashboardStats,
    predictor: EEGPredictor,
    buffer: TemporalFeatureBuffer,
    mode: str,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (20, 24, 30), -1)
    _draw_panel_frame(panel, rect, "LIVE SYSTEM TELEMETRY & MODEL STATUS", tag="REAL-TIME METRICS")

    duration_secs = int(time.time() - stats.start_time)
    mins, secs = divmod(duration_secs, 60)
    hours, mins = divmod(mins, 60)
    dur_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

    meta = predictor.metadata
    col1_items = [
        ("Frames Processed", f"{stats.frames_processed:,} frames"),
        ("Dropped Frames", f"{stats.frames_dropped} ({stats.frames_dropped / max(stats.frames_processed, 1) * 100:.1f}%)"),
        ("Measured FPS", f"{stats.fps:.1f} FPS"),
        ("Avg Frame Latency", f"{stats.frame_latency_ms:.1f} ms"),
        ("Feature Extraction", f"{stats.feat_latency_ms:.1f} ms"),
        ("Inference Latency", f"{stats.infer_latency_ms:.1f} ms" if predictor.status == "READY" else "N/A"),
    ]

    col2_items = [
        ("Session Duration", dur_str),
        ("Feature Buffer", f"{buffer.current_length} / {buffer.sequence_length} frames ({buffer.fill_ratio * 100:.0f}%)"),
        ("Model Architecture", meta.model_name),
        ("Model Parameters", f"{meta.param_count / 1e6:.2f}M params" if meta.param_count > 0 else "0"),
        ("Compute Device", meta.device),
        ("Uncertainty Status", stats.uncertainty_display),
    ]

    # Render 2 columns
    for idx, (lbl, val) in enumerate(col1_items):
        ry = y + 48 + idx * 30
        _text(panel, lbl, (x + 18, ry), 0.38, (140, 155, 170), 1)
        _text(panel, val, (x + 200, ry), 0.44, (225, 235, 245), 1)

    for idx, (lbl, val) in enumerate(col2_items):
        ry = y + 48 + idx * 30
        _text(panel, lbl, (x + 440, ry), 0.38, (140, 155, 170), 1)
        _text(panel, val, (x + 630, ry), 0.44, (225, 235, 245), 1)


def _draw_bottom_safety_bar(panel: np.ndarray) -> None:
    cv2.rectangle(panel, (0, 918), (1600, 960), (12, 14, 18), -1)
    cv2.line(panel, (0, 918), (1600, 918), (36, 44, 54), 1)
    _text(
        panel,
        "MEASURED: Facial Landmarks, 3D Pose, Kinematics  |  PREDICTED: Neural Oscillations (Requires Verified ML Checkpoint)  |  FOR RESEARCH MONITORING ONLY — NOT FOR MEDICAL/CLINICAL DIAGNOSIS",
        (38, 944),
        0.38,
        (130, 145, 160),
        1,
    )


def _dominant_band(values: np.ndarray) -> str:
    names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    if values.size == 0 or np.max(np.abs(values)) <= 1e-12:
        return "N/A"
    return names[int(np.argmax(values))]
