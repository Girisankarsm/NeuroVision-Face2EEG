# NeuroVision (Face2EEG)

NeuroVision is a research pipeline for studying whether facial movement features from a standard camera are statistically associated with synchronized EEG recordings. It includes live facial-feature tracking, dataset preparation, model training, and evaluation tools.

> **Research use only.** NeuroVision is not a medical device and does not measure EEG through a camera. Any EEG output is a model prediction, not a measurement. Model outputs require validation on synchronized data. When no model is loaded, the application reports `EEG MODEL NOT LOADED` and does not display predicted EEG.

## Features

- **Live facial tracking:** MediaPipe landmarks, eye aspect ratio (EAR), blink rate, mouth aspect ratio (MAR), head pose, movement, and action-unit intensities.
- **Streamlit interface:** live metrics, tracking quality hints, neutral-face calibration, feature recording, and optional raw-video recording.
- **Temporal data pipeline:** timestamp alignment, feature windows, dataset normalization, and subject-aware splits.
- **Modeling and evaluation:** MLP, temporal CNN, CNN-LSTM, Transformer, and multimodal architectures; cross-validation, statistical baselines, and evaluation reports.
- **Prediction uncertainty:** Monte Carlo Dropout and model uncertainty estimates where supported.

## Getting started

Use Python 3.10 or later. Install the project dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the Streamlit interface:

```bash
streamlit run app.py
```

The Live tab can be opened without a camera or checkpoint. Choose a camera index in the sidebar and start tracking when a camera is available. The Results and About tabs provide experiment outputs and research limitations.

## Live session workflow

1. Select a camera index and configure mirroring and the landmark overlay.
2. Start tracking. The main view shows blink rate, EAR, head pose, FPS, and blink count; additional metrics are available under **Advanced**.
3. Optionally calibrate for ten seconds while maintaining a neutral expression.
4. Enable **Record feature session** to write timestamped feature data to `data/recordings/` as CSV and NPZ files. Select **Also save raw video** only when video capture is needed.
5. Add event markers during recording to support later alignment with an EEG device.

Glass reflections and frame quality can affect facial measurements, including EAR. Recorded features are camera-derived and are not EEG.

![Streamlit Live tab screenshot placeholder](docs/live-tab-screenshot.png)

## Command-line workflows

Start the desktop live dashboard:

```bash
python main.py live --mode research
python main.py live --mode demo
python main.py live --checkpoint neurovision/models/checkpoints/best.pt --mode research
```

Train a model using a synchronized dataset:

```bash
python main.py train \
  --dataset neurovision/data/synchronized/dataset.npz \
  --model transformer \
  --out neurovision/models/checkpoints/best.pt
```

Evaluate a checkpoint:

```bash
python main.py evaluate \
  --dataset neurovision/data/synchronized/test.npz \
  --checkpoint neurovision/models/checkpoints/best.pt \
  --out-dir results
```

Run subject-independent cross-validation or baseline evaluation:

```bash
python main.py cv \
  --dataset neurovision/data/synchronized/dataset.npz \
  --model transformer \
  --folds 5 \
  --out-dir results/cv
```

```bash
python main.py baseline \
  --dataset neurovision/data/synchronized/dataset.npz \
  --folds 5
```

## Dataset format

Training and evaluation datasets use NumPy `.npz` archives with these arrays:

| Array | Shape | Description |
| --- | --- | --- |
| `facial` | `[N, sequence_length, 4233]` | Windowed facial feature sequences |
| `eeg` | `[N, eeg_window_samples]` | Synchronized EEG windows |
| `subjects` | `[N]` | Subject identifiers used for grouped splits |

Keep windows from each subject within a single split to avoid subject leakage. Raw or synchronized research data are not included in this repository.

## Tests

Run the test suite and bytecode compilation with:

```bash
python -m pytest -v
python -m compileall -q main.py neurovision
```

Tests use synthetic inputs and do not require camera hardware.

## Evaluation snapshot

The generated results below are a project snapshot, not evidence of validated EEG inference. In this snapshot, the permutation-test result does not show a statistically significant relationship. See `results/` for the underlying tables and control outputs.

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
