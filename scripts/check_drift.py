#!/usr/bin/env python3
"""
Run drift detection between reference and current data.

Usage:
  python scripts/check_drift.py --reference reference.csv --current current.csv [--model-id my_model] [--threshold 0.15]

Outputs JSON with drift_detected, drift_score, feature_drifts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from monitoring.drift_detector import DriftDetector
from ml_platform.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data drift between reference and current datasets")
    parser.add_argument("--reference", "-r", required=True, help="Path to reference CSV")
    parser.add_argument("--current", "-c", required=True, help="Path to current CSV")
    parser.add_argument("--model-id", default="", help="Model ID for report")
    parser.add_argument("--threshold", type=float, default=0.15, help="Drift threshold")
    parser.add_argument("--output", "-o", help="Write report JSON to file")
    args = parser.parse_args()

    load_config()
    detector = DriftDetector(drift_threshold=args.threshold)

    try:
        reference_data = pd.read_csv(args.reference)
        current_data = pd.read_csv(args.current)
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        return 1

    report = detector.compute_data_drift(reference_data, current_data, model_id=args.model_id)
    out = report.model_dump(mode="json")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if not report.drift_detected else 2  # exit 2 = drift detected


if __name__ == "__main__":
    sys.exit(main())
