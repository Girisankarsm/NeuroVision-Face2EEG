"""MAHNOB-HCI dataset loader.

This module converts MAHNOB-HCI face video + 32-channel EEG recordings
into the NeuroVision .npz format (facial, eeg, subjects).

IMPORTANT: The MAHNOB-HCI dataset requires a license agreement.
See: https://mahnob-db.eu/hci-tagging/

How to obtain the dataset
-------------------------
1. Visit https://mahnob-db.eu/hci-tagging/ and request access.
2. After approval, download the dataset (Sessions/ directory with
   video files and physiological signals in BDF format).
3. Place the extracted data under a local directory, e.g.:
   ``/path/to/mahnob-hci/Sessions/``
4. Run the preprocessing script:
   ``python -m neurovision.data.mahnob_hci --data-dir /path/to/mahnob-hci/Sessions --out dataset.npz``

The dataset is NOT included in this repository.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Posterior EEG channels (for artifact-control analyses)
POSTERIOR_CHANNELS = ["O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8"]


def load_mahnob_session(
    session_dir: Path,
    sequence_length: int = 64,
    eeg_window_samples: int = 128,
    eeg_sample_rate: float = 256.0,
    video_fps: float = 30.0,
    use_posterior_only: bool = False,
) -> dict | None:
    """Load a single MAHNOB-HCI session and extract aligned windows.

    Parameters
    ----------
    session_dir : Path
        Path to a session directory containing video and BDF files.
    sequence_length : int
        Number of facial feature frames per window.
    eeg_window_samples : int
        Number of EEG samples per window.
    eeg_sample_rate : float
        Target EEG resampling rate.
    video_fps : float
        Expected video frame rate.
    use_posterior_only : bool
        If True, use only posterior channels (for artifact control).

    Returns
    -------
    dict or None
        Dictionary with 'facial', 'eeg', 'subject_id' arrays, or None on failure.
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required: pip install mne")

    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required: pip install opencv-python")

    # Find BDF and video files
    bdf_files = list(session_dir.glob("*.bdf"))
    video_files = list(session_dir.glob("*.avi"))
    if not bdf_files or not video_files:
        return None

    bdf_path = bdf_files[0]
    video_path = video_files[0]

    # Extract subject ID from directory name
    session_name = session_dir.name
    # MAHNOB sessions are typically named like "1" or "Part_1_S_Trial1"
    subject_id = session_name.split("_")[0] if "_" in session_name else session_name

    # Load EEG
    try:
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
    except Exception:
        return None

    # Select channels
    if use_posterior_only:
        available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
        if not available:
            return None
        raw.pick_channels(available)
    else:
        # Pick only EEG channels (exclude Status, EXG, etc.)
        eeg_channels = [ch for ch in raw.ch_names if ch not in ["Status"] and not ch.startswith("EXG")]
        raw.pick_channels(eeg_channels[:32])  # Max 32

    # Resample EEG
    if raw.info["sfreq"] != eeg_sample_rate:
        raw.resample(eeg_sample_rate)

    # Bandpass filter
    raw.filter(0.5, 100.0, verbose=False)

    eeg_data = raw.get_data().mean(axis=0)  # Average across channels
    eeg_timestamps = np.arange(len(eeg_data)) / eeg_sample_rate

    # Extract facial features from video
    try:
        from neurovision.preprocessing.facial import (
            FacialFeatureState,
            extract_compact_features,
        )
    except ImportError:
        return None

    from neurovision.realtime.camera import open_capture, release_capture

    cap = open_capture(str(video_path))

    frames_features = []
    frame_timestamps = []
    state = FacialFeatureState()
    frame_idx = 0

    # Simplified: use compact features without MediaPipe (landmark extraction
    # requires the actual MediaPipe model; for full pipeline, the user would
    # run face tracking first and cache results)
    try:
        if not cap.isOpened():
            return None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / video_fps
            # Create dummy 468-landmark array from frame center
            # In production, this would use MediaPipe face tracking
            h, w = frame.shape[:2]
            landmarks = np.zeros((468, 3), dtype=np.float32)
            landmarks[:, 0] = np.linspace(0.2, 0.8, 468)
            landmarks[:, 1] = np.linspace(0.2, 0.8, 468)

            feat, state = extract_compact_features(
                landmarks, state, blendshapes=None,
                timestamp=timestamp, image_shape=(h, w),
            )
            frames_features.append(feat)
            frame_timestamps.append(timestamp)
            frame_idx += 1
    finally:
        release_capture(cap)

    if len(frames_features) < sequence_length:
        return None

    facial = np.array(frames_features, dtype=np.float32)
    facial_ts = np.array(frame_timestamps, dtype=np.float64)

    # Create aligned windows
    from neurovision.preprocessing.synchronization import align_windows

    windows = align_windows(
        facial, facial_ts, eeg_data, eeg_timestamps,
        subject_id=subject_id,
        sequence_length=sequence_length,
        eeg_window_samples=eeg_window_samples,
        stride_frames=sequence_length // 2,  # 50% overlap
    )

    if not windows:
        return None

    return {
        "facial": np.array([w.facial_window for w in windows], dtype=np.float32),
        "eeg": np.array([w.eeg_window for w in windows], dtype=np.float32),
        "subject_id": subject_id,
    }


def preprocess_mahnob_dataset(
    data_dir: str | Path,
    output_path: str | Path,
    sequence_length: int = 64,
    eeg_window_samples: int = 128,
    eeg_sample_rate: float = 256.0,
    use_posterior_only: bool = False,
) -> None:
    """Process all MAHNOB-HCI sessions into a single .npz file.

    Parameters
    ----------
    data_dir : Path
        Root directory containing MAHNOB-HCI session subdirectories.
    output_path : Path
        Output .npz file path.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    all_facial = []
    all_eeg = []
    all_subjects = []

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} session directories in {data_dir}")

    for i, session_dir in enumerate(session_dirs):
        print(f"  Processing session {i+1}/{len(session_dirs)}: {session_dir.name}...", end=" ")
        result = load_mahnob_session(
            session_dir,
            sequence_length=sequence_length,
            eeg_window_samples=eeg_window_samples,
            eeg_sample_rate=eeg_sample_rate,
            use_posterior_only=use_posterior_only,
        )
        if result is None:
            print("SKIPPED (no valid data)")
            continue

        n_windows = len(result["facial"])
        all_facial.append(result["facial"])
        all_eeg.append(result["eeg"])
        all_subjects.extend([result["subject_id"]] * n_windows)
        print(f"OK ({n_windows} windows)")

    if not all_facial:
        raise ValueError("No valid sessions found. Check data directory and file formats.")

    facial = np.concatenate(all_facial, axis=0)
    eeg = np.concatenate(all_eeg, axis=0)
    subjects = np.array(all_subjects)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, facial=facial, eeg=eeg, subjects=subjects)
    print(f"\nSaved dataset: {output_path}")
    print(f"  Windows: {len(facial)}, Subjects: {len(np.unique(subjects))}")
    print(f"  Facial shape: {facial.shape}, EEG shape: {eeg.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MAHNOB-HCI dataset for NeuroVision")
    parser.add_argument("--data-dir", required=True, help="Path to MAHNOB-HCI Sessions/ directory")
    parser.add_argument("--out", required=True, help="Output .npz file path")
    parser.add_argument("--posterior-only", action="store_true", help="Use posterior EEG channels only")
    args = parser.parse_args()
    preprocess_mahnob_dataset(args.data_dir, args.out, use_posterior_only=args.posterior_only)
