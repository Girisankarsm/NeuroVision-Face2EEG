from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from neurovision.config import DEFAULT_CONFIG
from neurovision.training.train import train_model


def build_default_checkpoint(
    output_path: str | Path = "neurovision/models/checkpoints/best.pt",
    dataset_path: str | Path = "neurovision/data/synchronized/default_research_dataset.npz",
    model_name: str = "multimodal",
) -> None:
    output_path = Path(output_path)
    dataset_path = Path(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating physiologically grounded calibration dataset...")
    np.random.seed(42)
    torch.manual_seed(42)

    num_windows = 240
    seq_len = 64
    feat_dim = 4233
    eeg_samples = 128
    sample_rate = 256.0

    # 4 distinct subjects to strictly preserve subject-independent validation
    subjects = np.array(
        ["sub-01"] * 60 + ["sub-02"] * 60 + ["sub-03"] * 60 + ["sub-04"] * 60
    )

    facial_data = np.zeros((num_windows, seq_len, feat_dim), dtype=np.float32)
    eeg_data = np.zeros((num_windows, eeg_samples), dtype=np.float32)

    t_eeg = np.linspace(0, eeg_samples / sample_rate, eeg_samples, endpoint=False)

    for i in range(num_windows):
        # Generate temporal kinematics in facial features
        movement_level = np.random.uniform(0.05, 0.95)
        freq_mod = np.random.uniform(0.8, 1.2)

        # Facial spatial + kinematic + blendshapes features
        noise = np.random.normal(0, 0.02, (seq_len, feat_dim)).astype(np.float32)
        facial_data[i] = noise
        # Dynamics velocity channels (dims 1415 to 2819)
        facial_data[i, :, 1415:1500] += movement_level * 0.1

        # Synthesize physiologically correlated EEG oscillations:
        # High movement -> higher beta (18Hz) & gamma (40Hz)
        # Low movement -> dominant alpha (10Hz) & theta (6Hz)
        alpha_amp = max(0.1, 1.2 - movement_level)
        beta_amp = 0.2 + movement_level * 0.8
        theta_amp = 0.3 + (1.0 - movement_level) * 0.4
        gamma_amp = 0.1 + movement_level * 0.5

        sig = (
            alpha_amp * np.sin(2 * np.pi * 10.0 * freq_mod * t_eeg)
            + beta_amp * np.sin(2 * np.pi * 20.0 * freq_mod * t_eeg)
            + theta_amp * np.sin(2 * np.pi * 6.0 * freq_mod * t_eeg)
            + gamma_amp * np.sin(2 * np.pi * 40.0 * freq_mod * t_eeg)
            + np.random.normal(0, 0.08, eeg_samples)
        )
        # Standardize
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        eeg_data[i] = sig.astype(np.float32)

    np.savez_compressed(
        dataset_path,
        facial=facial_data,
        eeg=eeg_data,
        subjects=subjects,
    )
    print(f"Saved dataset ({num_windows} windows across 4 subjects) to {dataset_path}")

    config = DEFAULT_CONFIG.copy()
    config["sequence_length"] = seq_len
    config["eeg_window_samples"] = eeg_samples
    config["model"] = dict(
        DEFAULT_CONFIG["model"],
        feature_dim=feat_dim,
        hidden_dim=256,
        dropout=0.10,
    )
    config["training"] = dict(
        DEFAULT_CONFIG["training"],
        batch_size=16,
        epochs=30,
        learning_rate=0.0005,
        patience=8,
    )

    print(f"Training default {model_name} model checkpoint...")
    train_model(
        config=config,
        dataset_path=dataset_path,
        model_name=model_name,
        output_path=output_path,
        seed=42,
    )
    print(f"Successfully generated trained default model checkpoint at: {output_path}")


if __name__ == "__main__":
    build_default_checkpoint()
