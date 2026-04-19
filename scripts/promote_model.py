#!/usr/bin/env python3
"""
Promote a candidate model to active after evaluation.

Usage:
  python scripts/promote_model.py --model-id <candidate_model_id>

Requires that the candidate is better than current active (e.g. after compare_models).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_platform.registry import ModelRegistry
from ml_platform.exceptions import ModelNotFoundError, PromotionError
from lifecycle.promoter import ModelPromoter
from ml_platform.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a candidate model to active")
    parser.add_argument("--model-id", required=True, help="Model ID of the candidate to promote")
    args = parser.parse_args()

    load_config()
    registry = ModelRegistry()
    promoter = ModelPromoter(registry=registry)

    try:
        meta = registry.get_model_metadata(args.model_id)
    except Exception as e:
        logger.error("Model not found: %s", e)
        return 1

    if meta.status == "active":
        logger.info("Model is already active")
        return 0

    if meta.status != "candidate":
        logger.error("Model status is '%s'; only candidate can be promoted", meta.status)
        return 1

    try:
        promoter.promote_to_active(args.model_id, {"candidate_is_better": True})
        logger.info("Promoted model_id %s to active", args.model_id)
        return 0
    except PromotionError as e:
        logger.error("Promotion failed: %s", e.message)
        return 1
    except Exception as e:
        logger.exception("Promotion failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
