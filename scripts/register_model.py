#!/usr/bin/env python3
"""
Register a new model version with the platform.

Usage:
  python scripts/register_model.py --model-path model.joblib --name fraud_detector --version v1 --type classification --features amount,merchant_id,time_of_day --target is_fraud [--metrics '{"accuracy": 0.94}']

Model is registered as candidate by default. Use platform API or promoter to activate.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from ml_platform.registry import ModelRegistry
from ml_platform.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a model with the ML Reliability Platform")
    parser.add_argument("--model-path", required=True, help="Path to pickled model (joblib)")
    parser.add_argument("--name", "--model-name", dest="model_name", required=True, help="Logical model name")
    parser.add_argument("--version", default="v1", help="Version string")
    parser.add_argument("--type", dest="model_type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--features", required=True, help="Comma-separated feature names")
    parser.add_argument("--target", required=True, help="Target variable name")
    parser.add_argument("--metrics", default="{}", help="JSON dict of performance metrics")
    parser.add_argument("--activate", action="store_true", help="Set this version as active after registration")
    args = parser.parse_args()

    load_config()
    registry = ModelRegistry()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return 1

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return 1

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError:
        metrics = {}

    metadata = {
        "model_name": args.model_name,
        "version": args.version,
        "model_type": args.model_type,
        "features": features,
        "target": args.target,
        "performance_metrics": metrics,
        "status": "candidate",
    }

    try:
        model_id = registry.register_model(model, metadata)
        logger.info("Registered model_id: %s", model_id)
        if args.activate:
            registry.update_model_status(model_id, "active")
            logger.info("Model set as active")
        print(model_id)
        return 0
    except Exception as e:
        logger.exception("Registration failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
