# NeuroVision — Real-Time Research Instrument & AI-Predicted EEG Modeling

## Abstract

NeuroVision is a scientific research instrument and machine learning pipeline designed to study whether temporal facial dynamics captured from standard webcam video can predict correlated electroencephalogram (EEG) neural oscillations.

> [!IMPORTANT]
> **Scientific Integrity & Ethical Disclaimer**:
> NeuroVision is a research instrument, **not a medical device or physiological EEG replacement**. Camera-only outputs represent AI-predicted statistical estimates and require rigorous experimental validation on synchronized multimodal corpora. When no trained model checkpoint is loaded, the application displays live facial dynamics while explicitly showing `EEG MODEL NOT LOADED`—it **never fabricates or simulates fake neural activity**.

---

## Key Capabilities

1. **Real-Time Facial Dynamics & Geometry**:
   - 468-point 3D facial landmarks via MediaPipe Face Mesh.
   - **Eye Aspect Ratio (EAR)** & **Blink Rate** calculation (blinks/min over a 60s rolling window).
   - **Mouth Aspect Ratio (MAR)** & Lip aperture dynamics.
   - **3D Head Pose Estimation**: Perspective-n-Point (`cv2.solvePnP`) extracting Euler angles (Pitch, Yaw, Roll) and 3D orientation projection axes.
   - **Kinematic Derivatives**: Real-time velocity (1st derivative), acceleration (2nd derivative), movement magnitude, and smoothed kinetic energy.
   - **Blendshape / Action Unit Intensities**: Action unit feature intensities (strictly labeled as physiological action units, not emotional states).

2. **Temporal Feature Buffer**:
   - Sliding window buffer ($T \times D$, e.g. $64 \times 4233$).
   - Per-frame timestamp tracking, jitter estimation, dropped frame detection, and fill telemetry (`BUFFER 64/64 READY`).

3. **Multi-Model ML Architectures**:
   - `mlp`: Linear baseline with LayerNorm and temporal pooling.
   - `temporal_cnn`: Multi-scale 1D dilated residual convolutional network.
   - `cnn_lstm`: Spatial Conv1D front-end with 2-layer Bidirectional LSTM.
   - `transformer`: Pre-LN Transformer encoder with multi-head self-attention and positional encoding.
   - `multimodal`: Tri-stream multimodal fusion network disentangling spatial landmarks ($1415$), kinematics ($2810$), and blendshapes ($8$) with cross-modal fusion attention.
   - `MultiTaskEEGHeads`: Predicts waveforms ($128$ samples), frequency band powers ($5$ bands), spectral distribution ($64$ bins), and heteroscedastic predictive uncertainty.

4. **Calibrated Uncertainty Estimation**:
   - Monte Carlo Dropout inference for epistemic predictive variance.
   - Explicit confidence reporting (`Confidence: N/A` when uncalibrated or checkpoint missing).
   - Clear distinction: **Model confidence ≠ biological certainty**.

5. **Research-Grade Scientific Monitoring Dashboard**:
   - Dark scientific workstation UI (1600x960) with subtle mesh overlays, 3D pose vectors, and corner-bracket tracking.
   - Dedicated **Live Statistics Panel** showing measured FPS, latency breakdown, buffer length, session duration, and face count.
   - Real predicted neural oscillations and frequency band power sparklines (Delta, Theta, Alpha, Beta, Gamma).
   - UI Modes:
     * `demo`: Clean focus on live video, landmarks, and key dynamics.
     * `research`: Comprehensive view with all telemetry, kinematics, buffer metrics, and model status.
     * `validation`: Real-time predicted vs ground-truth comparison (MAE, RMSE, Pearson $r$, residual error) when synchronized data is provided.

6. **Rigorous Training & Evaluation Pipeline**:
   - **Subject-Independent Splitting**: Strict `GroupShuffleSplit` and `GroupKFold` preventing data leakage between train and validation subjects.
   - **Scientific Metrics**: MAE, RMSE, Pearson $r$, Spearman $\rho$, $R^2$, spectral correlation, and band power MAE.
   - Automated generation of publication-ready figures in `results/`:
     * `prediction_vs_ground_truth.png`
     * `residuals.png`
     * `loss_curves.png`
     * `per_subject_metrics.png`
     * `band_powers.png`
     * `metrics.json` and `metrics_summary.csv`.

---

## Quickstart & CLI Commands

### Streamlit Live tab

Launch the webcam interface with `streamlit run app.py`. The Live tab works
without a camera or model: it shows an actionable camera message and the
`EEG MODEL NOT LOADED` state. Facial measures are live landmark measurements;
EEG output, when a compatible checkpoint is available, must be labeled
**predicted, not measured**.

Use the sidebar to select the camera, mirror the image, show landmarks, or
calibrate for ten seconds while holding a neutral expression. Enable **Record
feature session** to save timestamped CSV and NPZ files under
`data/recordings/`; select **Also save raw video** only when that is intended.
An event marker can be added during a recording to support later synchronization
with an EEG device. Advanced displays MAR, action unit intensities, session
timing, and processing latency. EAR can be less accurate with glasses.

![Streamlit Live tab screenshot placeholder](docs/live-tab-screenshot.png)

### 1. Launch Live Research Dashboard
```bash
# Research mode (default)
python3 main.py live --mode research

# Demo mode
python3 main.py live --mode demo

# With a trained checkpoint
python3 main.py live --checkpoint neurovision/models/checkpoints/best.pt --mode research
```

### 2. Train a Model on Synchronized Data
```bash
python3 main.py train \
  --dataset neurovision/data/synchronized/dataset.npz \
  --model transformer \
  --out neurovision/models/checkpoints/best.pt
```

### 3. Evaluate a Checkpoint
```bash
python3 main.py evaluate \
  --dataset neurovision/data/synchronized/test.npz \
  --checkpoint neurovision/models/checkpoints/best.pt \
  --out-dir results
```

### 4. Run Subject-Independent Cross-Validation
```bash
python3 main.py cv \
  --dataset neurovision/data/synchronized/dataset.npz \
  --model transformer \
  --folds 5 \
  --out-dir results/cv
```

### 5. Evaluate Baseline Models
```bash
python3 main.py baseline \
  --dataset neurovision/data/synchronized/dataset.npz \
  --folds 5
```

---

## Dataset Format

Training requires a synchronized `.npz` archive containing:
- `facial`: shape `[N_windows, sequence_length, 4233]`
- `eeg`: shape `[N_windows, eeg_window_samples]`
- `subjects`: shape `[N_windows]` (subject identifiers)

Windows from the same subject are strictly isolated to avoid subject leakage.

---

## Testing & Verification

Run the comprehensive unit and integration test suite:

```bash
# Verify compilation
python3 -m compileall main.py neurovision

# Run all tests
python3 -m pytest -v
```


<!-- RESULTS_START -->
## Results (Auto-Generated)

### Baseline Models
| model              |      mae |     rmse |         r2 |
|:-------------------|---------:|---------:|-----------:|
| Ridge (Blink Only) | 0.246669 | 0.354735 |  0.143473  |
| Training Mean      | 0.247105 | 0.354818 |  0.142308  |
| SVR                | 0.258451 | 0.370907 |  0.0636294 |
| Gradient Boosting  | 0.26218  | 0.380234 |  0.0244788 |
| Ridge (Compact)    | 0.274594 | 0.392293 | -0.0623854 |

### Deep Models (Cross-Validation)
| model        |     mae |    rmse |        r2 |
|:-------------|--------:|--------:|----------:|
| temporal_cnn | 6.93462 | 8.83438 | -0.321529 |

### Controls & Rigor
- **Permutation Test p-value**: 0.9701492537313433
- **Blink Ablation R² Drop**: -0.004943394660949707

### Scientific Interpretation
- **Are we predicting real EEG?** A p-value > 0.05 indicates the model is NOT learning a meaningful signal better than random chance.
- **Is it just blinks?** If the blink ablation drop is large (e.g. > 0.05), the model is heavily relying on ocular artifacts rather than neural activity.

<!-- RESULTS_END -->
