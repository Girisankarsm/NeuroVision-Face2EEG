from __future__ import annotations

import argparse
from pathlib import Path

from neurovision.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neurovision",
        description="Research tooling for facial-dynamics to AI-predicted EEG experiments.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="Run webcam dashboard")
    live.add_argument("--checkpoint", default=None)
    live.add_argument("--camera", type=int, default=0)

    train = sub.add_parser("train", help="Train a model from synchronized feature files")
    train.add_argument("--dataset", required=True)
    train.add_argument("--model", default="transformer", choices=["mlp", "temporal_cnn", "cnn_lstm", "transformer"])
    train.add_argument("--out", default="neurovision/models/checkpoints/best.pt")

    evaluate = sub.add_parser("evaluate", help="Evaluate a saved model")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "live":
        from neurovision.realtime.dashboard import run_dashboard

        run_dashboard(config=config, checkpoint=args.checkpoint, camera_index=args.camera)
    elif args.command == "train":
        from neurovision.training.train import train_model

        train_model(config=config, dataset_path=Path(args.dataset), model_name=args.model, output_path=Path(args.out))
    elif args.command == "evaluate":
        from neurovision.training.evaluate import evaluate_checkpoint

        metrics = evaluate_checkpoint(config=config, dataset_path=Path(args.dataset), checkpoint_path=Path(args.checkpoint))
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
