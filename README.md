# NeuroVision (Face2EEG)

NeuroVision is a research pipeline for studying whether facial movement features from a standard camera are statistically associated with synchronized EEG recordings. It includes live facial-feature tracking, dataset preparation, model training, and evaluation tools.

> **Research use only.** NeuroVision is not a medical device and does not measure EEG through a camera. Any EEG output is a model prediction, not a measurement. Model outputs require validation on synchronized data. When no model is loaded, the application reports `EEG MODEL NOT LOADED` and does not display predicted EEG.

## Features

- **Live facial tracking:** MediaPipe landmarks, eye aspect ratio (EAR), blink rate, mouth aspect ratio (MAR), head pose, movement, and action-unit intensities.
- **Streamlit interface:** live facial metrics, tracking quality hints, neutral-face calibration, feature recording, optional raw-video recording, and checkpoint-backed EEG predictions.
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
2. Start tracking. The main view shows blink rate, EAR, head pose, FPS, and blink count; facial signals are plotted against seconds since tracking started. Live processing and optional video recording target 18 FPS; actual rates depend on camera and hardware.
3. Optionally calibrate for ten seconds while maintaining a neutral expression. The measured open-eye EAR baseline sets the blink threshold to 65% of baseline.
4. Enable **Record feature session** to write timestamped feature data to `data/recordings/` as CSV and NPZ files. Select **Also save raw video** only when video capture is needed.
5. Add event markers during recording to support later alignment with an EEG device.

Glass reflections and frame quality can affect facial measurements, including EAR. Recorded features are camera-derived and are not EEG.

## Supervised EEG predictions

The Streamlit Live tab loads `neurovision/models/checkpoints/best.pt` by default. Use the **EEG checkpoint** field in the sidebar to select another trained checkpoint. The app reads the sequence length and feature dimension from the checkpoint, buffers contiguous face-tracked frames, and predicts after a complete window is available. Inference is rate-limited to at most twice per second. The default checkpoint expects 64 frames of 4,233 full facial features; compact checkpoints expecting 28 features use the compact extractor instead.

The Live view shows predicted alpha power. Under **Advanced**, it shows the predicted band-power distribution and waveform. These are model outputs, not EEG measurements; they require validation against synchronized recordings and may reflect facial artifacts. `MODEL NOT LOADED` means the selected checkpoint could not be loaded; the status detail provides the reason. The Results tab displays saved results only; train checkpoints from the command line:

```bash
python main.py train \
  --dataset neurovision/data/synchronized/dataset.npz \
  --model transformer \
  --out neurovision/models/checkpoints/best.pt
```

The default config and this example use compact 28-feature inputs. The training config's `model.feature_dim` must match the dataset's per-frame facial feature width. A checkpoint trained on compact 28-feature inputs and one trained on 4,233 full features are not interchangeable.

## Unsupervised analysis

The `cluster` command summarizes compact facial features over 2–4 second windows, then discovers facial-state patterns with a Gaussian Mixture Model (GMM). GMM component count (`k=2..8`) and diagonal/full covariance are selected by BIC. K-Means is the baseline, selecting `k` by training-fold silhouette with `n_init=20`. Suggested names such as “frequent blinking” describe facial patterns only and require manual review.

Every evaluation fold is a subject-independent `GroupKFold`. The per-subject normalizer, 95%-variance PCA, GMM/K-Means fits, and `k` selection use training subjects only. Held-out subjects are transformed with training-global statistics and assigned with `predict` or `predict_proba`. Cluster-versus-EEG analyses use z-scored log10 alpha, theta, and beta power, Kruskal-Wallis tests with Holm correction, within-subject label permutations, and subject-level bootstrap intervals. Window-level Kruskal-Wallis tests do not model repeated windows within subjects, so compare them with the subject-stratified permutation control. Regression compares compact features with GMM membership probabilities using identical folds.

These are exploratory association tests, not evidence that clusters are neural states. Blinks and facial muscle activity (including EMG) can affect EEG band power. The pipeline repeats the analysis after removing blink features and after excluding blink/movement-dominated facial clusters, but a surviving association still does not prove a neural origin. The t-SNE, profile, and timeline figures are descriptive; the archive has no session timestamps, so its timeline uses input window order.

Run the analysis with fixed defaults (seed 42, 5 subject folds, 1,000 permutations, 5 subject bootstraps):

```bash
python main.py cluster \
  --dataset neurovision/data/synchronized/dataset.npz \
  --out results/clustering \
  --folds 5 --seed 42 --permutations 1000 --bootstrap 5
```

The frame count divided by `--fps` must describe a 2–4 second window; the default is 18 FPS. Outputs include `cluster_results.json`, `cluster_table.csv`, and figures in `results/clustering/`. To save a final pipeline fitted on all supplied subjects for the optional Live estimate, add `--save-model`; this full-data artifact is for deployment, not held-out evaluation. The experiment command can include clustering with `--with-clustering`.

## Status

- Clustering code has been exercised on `synthetic_data.npz` and on planted three-state recovery tests, all labeled **SYNTHETIC SANITY CHECK**. These runs validate implementation behavior only.
- The checked-in EEG checkpoint loads in Streamlit and passes a **SYNTHETIC SANITY CHECK** for output shape/finite values. No prediction from synthetic input is presented as a research result.
- No MAHNOB-HCI or user-recorded dataset has been analyzed by the clustering pipeline yet. Do not interpret the checked-in supervised evaluation snapshot as evidence for clustering results.

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
| `facial` | `[N, sequence_length, 28]` | Compact per-frame facial features (legacy full-feature archives may use 4233) |
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
