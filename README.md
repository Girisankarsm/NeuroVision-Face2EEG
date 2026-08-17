# NeuroVision

## Abstract

NeuroVision is a research-oriented Python application for studying whether temporal facial dynamics from webcam video can predict correlated EEG activity. It is not an EEG measurement system. Camera-only outputs are always treated as **AI-PREDICTED EEG** or **ESTIMATED NEURAL ACTIVITY**.

## Research Question

How accurately can temporal facial dynamics predict EEG neural oscillatory activity in unseen subjects?

## Hypothesis

Synchronized facial dynamics may contain correlates of neural state, but any useful relationship must be demonstrated with real multimodal data, subject-independent validation, and transparent uncertainty reporting.

## Dataset

No fake training data is included. Training expects a synchronized `.npz` file with:

- `facial`: shape `[windows, sequence_length, feature_dim]`
- `eeg`: shape `[windows, eeg_window_samples]`
- `subjects`: shape `[windows]`

Windows from the same subject must not be split across train and validation/test sets.

## EEG Preprocessing

`neurovision/preprocessing/eeg.py` provides band-pass filtering, optional notch filtering, normalization, PSD, band power, relative band power, dominant frequency, RMS, variance, and spectral entropy. Preprocessing should be fit only on training data when dataset-level statistics are introduced.

## Facial Preprocessing

The live pipeline uses MediaPipe Face Mesh. Features include normalized 3D landmarks, geometric distances, velocity, acceleration, movement magnitude, rolling movement energy, and optional expression probabilities.

## Synchronization

`align_windows` aligns facial frame timestamps with EEG timestamps. It does not assume frame index equals EEG sample index.

## Model Architecture

Implemented models:

- MLP baseline
- Temporal CNN
- CNN + BiLSTM
- Temporal Transformer encoder

Each neural model uses multi-task heads for waveform, band power, spectral representation, and uncertainty.

## Training Methodology

Training uses AdamW, learning-rate scheduling, gradient clipping, dropout, early stopping, and best-checkpoint restoration. Model selection is validation-based.

## Baselines

Classical grouped Random Forest evaluation is implemented in `neurovision/training/baselines.py`. Neural baselines are implemented under `neurovision/models/`.

## Transformer Model

The transformer applies layer normalization, feature projection, positional encoding, multi-head self-attention, and multi-task EEG prediction heads.

## Evaluation Metrics

Evaluation reports regression metrics instead of a single accuracy number: MAE, RMSE, Pearson correlation, R2, spectral correlation, spectral distance, and per-band power error.

## Subject-Independent Validation

Training uses grouped subject splits. `GroupKFold` utilities are provided for cross-validation and leave-subject-style evaluation.

## Ablation Study

Feature-slice helpers define landmarks-only, landmarks plus movement, movement plus expression, full features, and full-features-transformer experiments. Populate ablation figures only with measured results.

## Real-Time Inference

Run:

```bash
python main.py live
```

Without a checkpoint the dashboard shows camera landmarks and facial dynamics, but the EEG display says `MODEL NOT LOADED`. It never fills missing predictions with random EEG.

With a trained checkpoint:

```bash
python main.py live --checkpoint neurovision/models/checkpoints/best.pt
```

## Visualization

Matplotlib helpers generate actual-vs-predicted waveform plots, spectrograms, band-power charts, and model comparisons.

## Limitations

Camera-based predictions represent model-estimated neural activity and are not equivalent to direct EEG measurements. Poor performance must be reported honestly. The system cannot validate the facial-to-EEG relationship without a legitimate synchronized video/EEG dataset.

## Ethical Considerations

NeuroVision must not be used to claim mind reading, clinical diagnosis, or physiological EEG measurement from webcam video. Avoid storing webcam imagery unless explicit consent and secure storage are in place.

## Future Work

Add dataset adapters for known multimodal EEG/video corpora, device-specific EEG sources, Optuna search, explainability reports, and a PyQtGraph high-frequency plotting frontend.
