from __future__ import annotations

import datetime
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from neurovision.preprocessing.facial import (
    FacialFeatureState,
    calibrated_blink_threshold,
    compute_windowed_stats,
    extract_compact_features,
)
from neurovision.realtime.camera import open_capture, release_capture
from neurovision.realtime.face_tracker import MediaPipeFaceTracker
from neurovision.realtime.recording import (
    LIVE_TARGET_FPS,
    blink_warmup_label,
    relative_chart_time,
    save_recording,
)


def _badge(icon: str, label: str, state: str) -> str:
    return f'<span class="status-{state}">{icon} {label}</span>'


def _estimate_facial_state(feature_window: list[np.ndarray], artifact: dict | None) -> str:
    if artifact is None or artifact.get("synthetic_training_data", False):
        return "N/A"
    if not np.isclose(float(artifact.get("fps", LIVE_TARGET_FPS)), LIVE_TARGET_FPS):
        return "N/A"
    window_length = int(artifact.get("window_frames", 64))
    if len(feature_window) < window_length:
        return "N/A"
    try:
        window_features = compute_windowed_stats(np.asarray(feature_window[-window_length:], dtype=np.float32))
        normalized = artifact["scaler"].transform(window_features[None, :], np.asarray(["live"]))
        embedding = artifact["pca"].transform(normalized)
        probabilities = artifact["gmm"].predict_proba(embedding)[0]
        cluster = int(np.argmax(probabilities))
        names = artifact.get("cluster_names", [])
        name = names[cluster] if cluster < len(names) else f"cluster {cluster}"
        return f"{name} ({probabilities[cluster]:.0%})"
    except (ValueError, KeyError, RuntimeError, AttributeError, IndexError):
        return "N/A"


def render_live_tab():
    st.header("Live Webcam Feed")
    st.caption("Facial measurements are derived from webcam landmarks. EEG is never measured by this app.")
    ss = st.session_state
    defaults = {"calib_start": None, "calib_ears": [], "baseline_ear": None,
                "blink_total": 0, "record_rows": [], "record_started": None,
                "record_base": None, "markers": [], "overlay_opacity": 0.7,
                "cluster_feature_window": []}
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value

    with st.sidebar:
        st.subheader("Live Controls")
        camera_idx = int(st.number_input("Camera index", min_value=0, max_value=10, value=0))
        mirror = st.checkbox("Mirror video", value=True)
        show_landmarks = st.checkbox("Landmark overlay", value=True)
        ss.overlay_opacity = st.slider("Overlay opacity", 0.1, 1.0, float(ss.overlay_opacity))
        show_advanced = st.toggle("Advanced", value=False) if hasattr(st, "toggle") else st.checkbox("Advanced", value=False)
        st.divider()
        st.caption("Face tracking only · no EEG is measured")
        if st.button("Calibrate neutral face (10 seconds)"):
            ss.calib_start, ss.calib_ears = time.time(), []
            ss.baseline_ear = None
        if ss.baseline_ear is not None:
            st.success(f"Calibrated · baseline EAR {ss.baseline_ear:.3f} · blink threshold {ss.blink_threshold:.3f}")
        st.divider()
        record = st.checkbox("Record feature session", value=False)
        record_video = st.checkbox("Also save raw video", value=False, disabled=not record)
        if record and ss.record_started is None:
            ss.record_started = time.time()
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ss.record_base = Path("data/recordings") / f"session_{stamp}"
            ss.record_rows = []
        elif not record and ss.record_started is not None:
            try:
                csv_path, npz_path = save_recording(ss.record_rows, ss.record_base, {
                    "fps": ss.get("last_fps"), "baseline_ear": ss.baseline_ear,
                    "blink_threshold": ss.get("blink_threshold"), "app_version": "1.0",
                    "event_markers": ss.markers,
                })
                st.sidebar.success(f"Saved {len(ss.record_rows)} frames: {npz_path} and CSV")
            except (OSError, ValueError) as exc:
                st.sidebar.error(f"Could not save recording: {exc}")
            ss.record_started, ss.record_base = None, None
        if ss.record_started:
            st.caption(f"Recording · {time.time()-ss.record_started:.1f}s · {len(ss.record_rows)} frames")
            if st.button("Add event marker"):
                ss.markers.append({"timestamp": time.time(), "elapsed": time.time()-ss.record_started})

    status = st.empty()
    left, right = st.columns([6, 4])
    with left:
        run_camera = st.checkbox("Start webcam tracking", key="live_camera_running")
        frame_slot = st.empty()
        quality_slot = st.empty()
        calibration_slot = st.empty()
    with right:
        st.subheader("Key metrics")
        a, b = st.columns(2)
        blink_metric, ear_metric = a.empty(), b.empty()
        pose_metric, fps_metric = a.empty(), b.empty()
        total_metric = st.empty()
        st.subheader("Facial signals · last 15 seconds")
        st.caption("Horizontal axis: seconds since tracking started")
        chart_slot = st.empty()
        if show_advanced:
            st.subheader("Action unit intensities")
            mar_metric, bars_slot = st.empty(), st.empty()
            cluster_state_metric = st.empty()
            st.caption("Facial-state estimate, not a brain state")
            session_slot, latency_slot = st.empty(), st.empty()
            model_box = st.container(border=True)
            with model_box:
                st.markdown("**EEG MODEL NOT LOADED**")
                st.caption("No model loaded. Load a checkpoint or train one from the Results tab.")
                st.caption("Predictions are predicted, not measured.")

    if not run_camera:
        if show_advanced:
            cluster_state_metric.metric("Facial state", "N/A")
        status.markdown("  ".join([_badge("●", "Camera: not found / stopped", "neutral"),
                                  _badge("●", "Face: not detected", "neutral"),
                                  _badge("●", "Model: EEG MODEL NOT LOADED", "warning"),
                                  _badge("●", "Buffer: 0/64", "neutral")]), unsafe_allow_html=True)
        return

    cap = None
    tracker = None
    try:
        cap = open_capture(camera_idx)
        if not cap.isOpened():
            status.error(f"Camera not found. Check permissions or choose another camera index ({camera_idx}).")
            return
        try:
            tracker = MediaPipeFaceTracker()
        except (RuntimeError, ImportError) as exc:
            status.error(f"Face tracking is unavailable: {exc}")
            return

        state = FacialFeatureState()
        cluster_artifact = None
        if show_advanced:
            artifact_path = Path("models/gmm.joblib")
            if artifact_path.exists():
                try:
                    import joblib

                    cluster_artifact = joblib.load(artifact_path)
                except (OSError, EOFError, ValueError, KeyError, ImportError):
                    cluster_artifact = None
        started, previous, previous_frame = time.time(), time.time(), time.time()
        rows: list[dict] = []
        frames = dropped = 0
        feature_frames = 0
        valid_data_seconds = 0.0
        last_feature_time = None
        blink_total_before = ss.blink_total
        video_writer = None
        if record_video and ss.record_started:
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(str(ss.record_base.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), float(LIVE_TARGET_FPS), (width, height))

        while ss.get("live_camera_running", False):
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                status.error("Camera connection was lost. Check that the camera is still connected.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.time()
            frames += 1
            tracking = tracker.process(frame)
            face_found = tracking.landmarks is not None
            feature = None
            if face_found:
                feature_frames += 1
                if last_feature_time is not None:
                    valid_data_seconds += min(max(0.0, now - last_feature_time), 0.2)
                last_feature_time = now
                height, width = frame.shape[:2]
                feature, state = extract_compact_features(tracking.landmarks, state, tracking.blendshapes,
                                                           timestamp=now, image_shape=(height, width))
                ss.cluster_feature_window.append(feature.copy())
                window_length = int(cluster_artifact.get("window_frames", 64)) if cluster_artifact else 64
                ss.cluster_feature_window = ss.cluster_feature_window[-window_length:]
                if state.blink_event:
                    ss.blink_total += 1
                if ss.calib_start is not None:
                    if now - ss.calib_start < 10:
                        ss.calib_ears.append(state.avg_ear)
                        calibration_slot.progress(min(1.0, (now - ss.calib_start) / 10.0),
                                                  text=f"Neutral face calibration · {max(0, 10-(now-ss.calib_start)):.1f}s remaining")
                    elif ss.calib_ears:
                        ss.baseline_ear = float(np.mean(ss.calib_ears))
                        ss.blink_threshold = calibrated_blink_threshold(ss.baseline_ear)
                        state.blink_threshold = ss.blink_threshold
                        ss.calib_start = None
                        calibration_slot.empty()
                if show_landmarks:
                    overlay = frame.copy()
                    tracker._draw_face_mesh(overlay, tracking.landmarks)
                    frame = cv2.addWeighted(overlay, float(ss.overlay_opacity), frame,
                                            1.0 - float(ss.overlay_opacity), 0.0)
                label = blink_warmup_label(valid_data_seconds)
                blink_metric.metric("Blink rate", label or f"{state.blink_rate:.1f} blinks/min", help="Rolling 60 second blink count per minute.")
                ear_metric.metric("EAR", f"{state.avg_ear:.3f}", help="Eye openness ratio; drops toward 0 during a blink.")
                pose_metric.metric("Head pose", f"{state.head_pose[0]:.1f}°, {state.head_pose[1]:.1f}°, {state.head_pose[2]:.1f}°", help="Pitch, yaw and roll from face landmarks.")
                if show_advanced:
                    mar_metric.metric("MAR", f"{state.mar:.3f}", help="Mouth aspect ratio from lip landmarks.")
                    cluster_state_metric.metric("Facial state", _estimate_facial_state(ss.cluster_feature_window, cluster_artifact))
                au = state.au_intensities
                elapsed = relative_chart_time(now, started)
                rows.append({"time": elapsed, "EAR": state.avg_ear, "Blink": state.avg_ear if state.blink_event else np.nan,
                             "Head motion": state.movement_energy})
                rows = [row for row in rows if elapsed - row["time"] <= 15]
                if frames % 3 == 0:
                    plot = pd.DataFrame(rows).set_index("time")
                    chart_slot.line_chart(plot[["EAR", "Blink", "Head motion"]], height=230)
                if show_advanced:
                    items = [("brow_raise_AU", au.get("brow_raiser", 0)), ("jaw_open_AU", au.get("jaw_open", 0)), ("smile_AU12", au.get("smile_AU12", 0))]
                    bars_slot.markdown("".join(
                        f'<div style="font-size:.85rem;color:#8FA3C2;margin:.35rem 0 .1rem">{name} · {value:.2f}</div>'
                        f'<progress value="{max(0.0, min(1.0, float(value))):.3f}" max="1" style="width:100%;accent-color:#4CC9F0"></progress>'
                        for name, value in items
                    ), unsafe_allow_html=True)
                x, y, w, h = tracking.bounding_box or (0, 0, 0, 0)
                brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                hints = []
                if w < frame.shape[1] * 0.22 or h < frame.shape[0] * 0.22: hints.append("Face may be too far away")
                if x < 4 or y < 4 or x+w >= frame.shape[1]-4 or y+h >= frame.shape[0]-4: hints.append("Face is off-center or clipped")
                if brightness < 45: hints.append("Image is too dark")
                if tracking.tracking_quality < 0.35: hints.append("Low tracking confidence")
                if not hints: hints.append("Face position and lighting look usable")
                quality_slot.caption(" · ".join(hints) + " · Glasses can reduce EAR accuracy.")
                pose = state.head_pose
                values = (state.avg_ear, state.mar, *pose, int(state.blink_event), au.get("brow_raiser", 0), au.get("jaw_open", 0), au.get("smile_AU12", 0), 1)
            else:
                ss.cluster_feature_window = []
                blink_metric.metric("Blink rate", "N/A")
                ear_metric.metric("EAR", "N/A")
                pose_metric.metric("Head pose", "N/A")
                if show_advanced:
                    cluster_state_metric.metric("Facial state", "N/A")
                quality_slot.caption("No face detected. Center your face in the camera view.")
                values = (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0)

            current_fps = 1 / max(now - previous_frame, 1e-6)
            if now - previous_frame > (1 / LIVE_TARGET_FPS) * 1.8:
                dropped += max(1, int((now - previous_frame) * LIVE_TARGET_FPS) - 1)
            previous_frame = now
            ss.last_fps = current_fps
            fps_metric.metric("FPS", f"{current_fps:.1f}", help="Frames processed per second.")
            total_metric.metric("Total blinks", ss.blink_total, help="Blink events observed during this app session.")
            last_blink = state.blink_timestamps[-1] if state.blink_timestamps else None
            total_metric.caption("Time since last blink: " + (f"{max(0, now-last_blink):.1f}s" if last_blink is not None else "N/A"))
            if record and ss.record_started:
                keys = ["timestamp", "EAR", "MAR", "pitch", "yaw", "roll", "blink_flag", "au_brow_raise_AU", "au_jaw_open_AU", "au_smile_AU12", "face_detected"]
                ss.record_rows.append(dict(zip(keys, [now, *values])))
            status.markdown("  ".join([_badge("●", "Camera: OK", "success"),
                                      _badge("●", f"Face: {'detected' if face_found else 'not detected'}", "success" if face_found else "error"),
                                      _badge("●", "Model: EEG MODEL NOT LOADED", "warning"),
                                      _badge("●", f"Buffer: {min(feature_frames,64)}/64", "neutral")]), unsafe_allow_html=True)
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if video_writer is not None:
                video_writer.write(frame)
            if show_advanced:
                session_slot.metric("Session", f"{now-started:.1f}s · {frames} frames · {dropped} dropped")
                latency_slot.metric("Processing latency", f"{(time.time()-loop_start)*1000:.1f} ms")
            delay = max(0.0, (1 / LIVE_TARGET_FPS) - (time.time() - loop_start))
            if delay:
                time.sleep(delay)
    except (cv2.error, OSError) as exc:
        status.error(f"Live tracking stopped: {exc}")
    finally:
        try:
            if cap is not None:
                release_capture(cap)
        finally:
            try:
                if tracker is not None:
                    tracker.close()
            finally:
                if 'video_writer' in locals() and video_writer is not None:
                    video_writer.release()
                ss.blink_total = max(ss.blink_total, blink_total_before if 'blink_total_before' in locals() else ss.blink_total)
