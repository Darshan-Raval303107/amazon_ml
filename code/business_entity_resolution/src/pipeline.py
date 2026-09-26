"""
Pipeline orchestrator module for Business Entity Resolution.

CLI and programmatic entry point for running training, inference,
and validation stages across the entity resolution workflow.
"""

import argparse
from pathlib import Path
import sys

from .config import ModelConfig, PathConfig


def run_training_pipeline(args: argparse.Namespace) -> None:
    """
    Execute full training workflow:
    1. Load training datasets (source 1, 2, 3 and ground truth)
    2. Normalize text fields
    3. Run candidate generation / blocking
    4. Compute traditional features and DL multilingual embeddings
    5. Train CatBoostClassifier
    6. Perform validation and threshold tuning on macro F_0.5
    7. Save model checkpoint and threshold
    """
    from .train import run_training_pipeline as _train
    _train()


def run_prediction_pipeline(args: argparse.Namespace) -> None:
    """
    Execute inference workflow on test data:
    1. Load test datasets
    2. Normalize text fields
    3. Generate candidate pairs -> save output/candidate_pairs.tsv
    4. Extract pairwise features & multilingual embeddings
    5. Score candidates with CatBoost model
    6. Filter using optimal threshold -> save output/matching_results.tsv
    """
    from .predict import Predictor, PredictionConfig
    cfg = PredictionConfig()
    predictor = Predictor(cfg)
    predictor.run()


def main() -> None:
    """CLI parser for entity resolution execution modes."""
    parser = argparse.ArgumentParser(
        description="End-to-End Business Entity Resolution Pipeline"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Pipeline execution mode")

    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Run model training and validation")
    train_parser.add_argument(
        "--data-dir", type=Path, default=PathConfig().train_dir, help="Directory containing training TSV files"
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=Path, default=PathConfig().artifacts_dir, help="Directory to save model artifacts"
    )

    # Prediction subcommand
    predict_parser = subparsers.add_parser("predict", help="Run inference on test data")
    predict_parser.add_argument(
        "--test-dir", type=Path, default=PathConfig().test_dir, help="Directory containing test TSV files"
    )
    predict_parser.add_argument(
        "--output-dir", type=Path, default=PathConfig().output_dir, help="Directory to save submission TSVs"
    )
    predict_parser.add_argument(
        "--checkpoint-dir", type=Path, default=PathConfig().artifacts_dir, help="Directory containing model artifacts"
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline(args)
    elif args.mode == "predict":
        run_prediction_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
