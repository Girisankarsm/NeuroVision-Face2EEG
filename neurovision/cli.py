from __future__ import annotations

import argparse
from pathlib import Path

from neurovision.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neurovision",
        description="NeuroVision: Real-time facial-dynamics research instrument & AI-predicted EEG oscillation modeling.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration YAML file")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Live dashboard command
    live = sub.add_parser("live", help="Launch the real-time webcam dashboard")
    live.add_argument("--checkpoint", default="neurovision/models/checkpoints/best.pt", help="Path to trained PyTorch model checkpoint (.pt)")
    live.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    live.add_argument("--mode", default="research", choices=["demo", "research", "validation"], help="UI/UX dashboard mode")
    live.add_argument("--validation-dataset", default=None, help="Optional synchronized dataset for validation mode")

    # 2. Train command
    train = sub.add_parser("train", help="Train a model on synchronized facial-EEG data")
    train.add_argument("--dataset", required=True, help="Path to synchronized .npz dataset")
    train.add_argument("--model", default="transformer", choices=["mlp", "temporal_cnn", "cnn_lstm", "transformer", "multimodal"], help="Model architecture")
    train.add_argument("--out", default="neurovision/models/checkpoints/best.pt", help="Output path for best checkpoint")
    train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # 3. Evaluate command
    evaluate = sub.add_parser("evaluate", help="Evaluate a saved model checkpoint against a dataset")
    evaluate.add_argument("--dataset", required=True, help="Path to test dataset .npz")
    evaluate.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    evaluate.add_argument("--out-dir", default="results", help="Directory to save evaluation plots and metrics")

    # 4. Cross-validation command
    cv = sub.add_parser("cv", help="Run subject-independent K-Fold cross validation")
    cv.add_argument("--dataset", required=True, help="Path to dataset .npz")
    cv.add_argument("--model", default="transformer", choices=["mlp", "temporal_cnn", "cnn_lstm", "transformer", "multimodal"], help="Model architecture")
    cv.add_argument("--folds", type=int, default=5, help="Number of subject folds")
    cv.add_argument("--epochs", type=int, default=20, help="Epochs per fold")
    cv.add_argument("--out-dir", default="results/cv", help="Output directory for CV logs and table")

    # 5. Baseline models command
    baseline = sub.add_parser("baseline", help="Evaluate statistical and machine learning baseline models")
    baseline.add_argument("--dataset", required=True, help="Path to dataset .npz")
    baseline.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")

    # 6. Experiment command
    experiment = sub.add_parser("experiment", help="Run full pipeline: baselines, deep models, controls, and report")
    experiment.add_argument("--dataset", required=True, help="Path to synchronized dataset .npz")
    experiment.add_argument("--out-dir", default="results", help="Directory to save experiment results")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "live":
        from neurovision.realtime.dashboard import run_dashboard

        val_eeg = None
        if args.validation_dataset:
            import numpy as np

            arr = np.load(args.validation_dataset)
            if "eeg" in arr:
                val_eeg = arr["eeg"]

        run_dashboard(
            config=config,
            checkpoint=args.checkpoint,
            camera_index=args.camera,
            mode=args.mode,
            validation_eeg=val_eeg,
        )
    elif args.command == "train":
        from neurovision.training.train import train_model

        train_model(
            config=config,
            dataset_path=Path(args.dataset),
            model_name=args.model,
            output_path=Path(args.out),
            seed=args.seed,
        )
    elif args.command == "evaluate":
        from neurovision.training.evaluate import evaluate_checkpoint

        evaluate_checkpoint(
            config=config,
            dataset_path=Path(args.dataset),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.out_dir),
        )
    elif args.command == "cv":
        from neurovision.training.cross_validation import run_cross_validation

        run_cross_validation(
            config=config,
            dataset_path=Path(args.dataset),
            model_name=args.model,
            folds=args.folds,
            epochs=args.epochs,
            output_dir=Path(args.out_dir),
        )
    elif args.command == "baseline":
        from neurovision.training.baselines import evaluate_baselines

        evaluate_baselines(
            dataset_path=Path(args.dataset),
            folds=args.folds,
            sample_rate=float(config.get("eeg_sample_rate", 256.0)),
        )
    elif args.command == "experiment":
        from neurovision.training.experiment import run_full_experiment
        from neurovision.training.report import write_report_to_readme

        run_full_experiment(dataset_path=Path(args.dataset), output_dir=Path(args.out_dir))
        write_report_to_readme(results_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
