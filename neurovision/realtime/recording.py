"""Compact, timestamped live feature recording helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RECORDING_COLUMNS = (
    "timestamp", "EAR", "MAR", "pitch", "yaw", "roll", "blink_flag",
    "au_brow_raise_AU", "au_jaw_open_AU", "au_smile_AU12", "face_detected",
)


def blink_warmup_label(elapsed_seconds: float, window_seconds: float = 60.0) -> str | None:
    """Return a calibration label while the rolling blink window fills."""
    if elapsed_seconds >= window_seconds:
        return None
    return f"Calibrating... {max(0, int(elapsed_seconds))}s / {int(window_seconds)}s"


def save_recording(rows: list[dict], output_base: str | Path, metadata: dict) -> tuple[Path, Path]:
    """Write matching CSV and NPZ feature records with reproducibility metadata."""
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = base.with_suffix(".csv"), base.with_suffix(".npz")
    normalized = [{column: row.get(column, "") for column in RECORDING_COLUMNS} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORDING_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    values = np.asarray([[row.get(column, np.nan) for column in RECORDING_COLUMNS] for row in normalized], dtype=np.float64)
    np.savez_compressed(npz_path, features=values, columns=np.asarray(RECORDING_COLUMNS), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return csv_path, npz_path
