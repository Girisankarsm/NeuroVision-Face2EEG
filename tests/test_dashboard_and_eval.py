from collections import deque
from pathlib import Path
import numpy as np
import pytest
import torch

from neurovision.config import DEFAULT_CONFIG
from neurovision.preprocessing.facial import FacialFeatureState
from neurovision.realtime.dashboard import DashboardStats, _render_scientific_dashboard
from neurovision.realtime.feature_buffer import TemporalFeatureBuffer
from neurovision.realtime.inference import EEGPredictor
from neurovision.training.evaluate import evaluate_checkpoint
from neurovision.training.metrics import regression_metrics
from neurovision.training.train import train_model


def test_temporal_feature_buffer():
    buf = TemporalFeatureBuffer(sequence_length=10, feature_dim=32)
    assert not buf.ready()
    assert buf.fill_ratio == 0.0

    for i in range(10):
        feat = np.ones(32, dtype=np.float32) * i
        buf.append(feat, timestamp=100.0 + i * 0.033)

    assert buf.ready()
    assert buf.fill_ratio == 1.0
    assert buf.current_length == 10
    tensor = buf.tensor()
    assert tensor.shape == (1, 10, 32)
    stats = buf.stats()
    assert "mean" in stats
    assert "jitter_ms" in stats


def test_dashboard_rendering_smoke():
    # Verify that _render_scientific_dashboard builds a valid 1600x960 image without crashing
    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    waveform_hist = deque([0.1, -0.2, 0.5, 0.3], maxlen=512)
    band_values = np.array([0.2, 0.3, 0.4, 0.1, 0.05], dtype=np.float32)
    band_hist = {k: deque([0.1, 0.2], maxlen=60) for k in ["delta", "theta", "alpha", "beta", "gamma"]}
    state = FacialFeatureState(blink_rate=14.0, left_ear=0.31, right_ear=0.32, mar=0.15)
    stats = DashboardStats(fps=29.8, frame_latency_ms=16.2, face_status="TRACKING")
    predictor = EEGPredictor(fallback_config=DEFAULT_CONFIG)
    buffer = TemporalFeatureBuffer(10, 4233)

    for mode in ["demo", "research", "validation"]:
        panel = _render_scientific_dashboard(
            camera_frame=camera_frame,
            waveform_history=waveform_hist,
            band_values=band_values,
            band_history=band_hist,
            state=state,
            stats=stats,
            predictor=predictor,
            buffer=buffer,
            mode=mode,
        )
        assert panel.shape == (960, 1600, 3)
        assert panel.dtype == np.uint8


def test_regression_metrics_computation():
    target = np.sin(np.linspace(0, 10, 500))
    pred = target + np.random.normal(0, 0.05, 500)
    metrics = regression_metrics(pred, target, sample_rate=256.0)
    assert metrics["mae"] < 0.2
    assert metrics["rmse"] < 0.2
    assert metrics["pearson_correlation"] > 0.8
    assert metrics["spearman_correlation"] > 0.8
    assert metrics["r2"] > 0.7


def test_training_and_evaluation_pipeline(tmp_path: Path):
    # Create synthetic synchronized dataset for testing pipeline integrity
    num_windows = 40
    seq_len = 8
    feat_dim = 4233
    eeg_samples = 128

    facial = np.random.randn(num_windows, seq_len, feat_dim).astype(np.float32)
    eeg = np.random.randn(num_windows, eeg_samples).astype(np.float32)
    # 4 distinct subjects
    subjects = np.array(["subj_01"] * 10 + ["subj_02"] * 10 + ["subj_03"] * 10 + ["subj_04"] * 10)

    dataset_file = tmp_path / "test_dataset.npz"
    np.savez(dataset_file, facial=facial, eeg=eeg, subjects=subjects)

    config = DEFAULT_CONFIG.copy()
    config["sequence_length"] = seq_len
    config["eeg_window_samples"] = eeg_samples
    config["training"] = dict(DEFAULT_CONFIG["training"], batch_size=4, epochs=2, patience=2)
    config["model"] = dict(DEFAULT_CONFIG["model"], feature_dim=feat_dim, hidden_dim=32, attention_heads=2, transformer_layers=1)

    checkpoint_file = tmp_path / "test_best.pt"
    train_res = train_model(config, dataset_file, model_name="transformer", output_path=checkpoint_file)
    assert checkpoint_file.exists()
    assert train_res["best_epoch"] >= 1

    # Test evaluation
    results_dir = tmp_path / "results"
    report = evaluate_checkpoint(config, dataset_file, checkpoint_file, output_dir=results_dir)
    assert "overall_metrics" in report
    assert "per_subject_metrics" in report
    assert (results_dir / "metrics.json").exists()
    assert (results_dir / "prediction_vs_ground_truth.png").exists()
    assert (results_dir / "residuals.png").exists()
    assert (results_dir / "band_powers.png").exists()
