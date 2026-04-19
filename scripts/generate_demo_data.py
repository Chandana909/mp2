"""
Generate rich demo data for the ML Reliability Platform.

Creates 2 model families with multiple versions, realistic metrics,
drift reference data, and production data (with injected drift) so the
full platform can be demonstrated end-to-end without real training pipelines.

Usage:
    python scripts/generate_demo_data.py
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml_platform.config import load_config
from ml_platform.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Model family specs ──────────────────────────────────────────────────────
MODELS = [
    {
        "model_name": "customer_churn",
        "features": ["age", "tenure_months", "monthly_charge", "total_charge", "support_calls"],
        "target": "churn",
        "model_type": "binary_classification",
        "versions": [
            # (version, n_estimators, max_depth, status, description)
            ("v1", 10, 3, "retired",   "Initial baseline · logistic regression era"),
            ("v2", 30, 5, "retired",   "Feature engineering · added support_calls"),
            ("v3", 60, 7, "retired",   "Hyperparameter tuned · GridSearchCV"),
            ("v4", 100, 9, "active",   "Production model · deployed 2026-03-01"),
            ("v5", 50,  4, "candidate","New candidate · early dropout regularisation"),
        ],
    },
    {
        "model_name": "fraud_detector",
        "features": ["transaction_amount", "merchant_category", "time_of_day",
                     "card_present", "user_age"],
        "target": "is_fraud",
        "model_type": "binary_classification",
        "versions": [
            ("v1", 20,  4, "retired",   "Baseline RF · false-positive heavy"),
            ("v2", 80,  8, "active",    "Production · threshold calibrated 0.45"),
            ("v3", 120, None, "candidate", "GBM variant · lower recall tradeoff"),
        ],
    },
]

# ── Base dates for versioning timeline ─────────────────────────────────────
BASE_DATE = datetime(2026, 1, 15)
DATE_STEP = timedelta(days=18)


def train_model(version: str, n_estimators: int, max_depth, X_train, y_train,
                X_test, y_test, model_name: str, family_idx: int):
    """Train a RandomForest with progression-based randomness."""
    seed = (family_idx * 100) + int(version.lstrip("v"))
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def main():
    logger.info("=" * 62)
    logger.info("  🚀  ML RELIABILITY PLATFORM — DEMO DATA GENERATOR")
    logger.info("=" * 62)
    logger.info("")

    load_config()
    registry = ModelRegistry()

    for family_idx, spec in enumerate(MODELS):
        model_name = spec["model_name"]
        features = spec["features"]
        n_features = len(features)
        logger.info("▶  Model family: %s (%d versions)", model_name, len(spec["versions"]))

        # Retire any existing models in this family to start clean
        try:
            existing = registry.list_models(model_name=model_name)
            for m in existing:
                registry.update_model_status(m.model_id, "retired")
            if existing:
                logger.info("   ↩  Retired %d existing versions", len(existing))
        except Exception:
            pass

        # Generate synthetic data
        X, y = make_classification(
            n_samples=4000,
            n_features=n_features,
            n_informative=max(2, n_features - 1),
            n_redundant=1,
            random_state=family_idx * 7 + 42,
            class_sep=0.9,
            weights=[0.65, 0.35],   # Realistic class imbalance
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        last_id = None
        last_status = None

        for version_idx, (version, n_est, max_d, status, description) in enumerate(
            spec["versions"]
        ):
            existing_metas = registry.list_models(model_name=model_name)
            if any(m.version == version for m in existing_metas):
                logger.info("   ⭐  %s %s | ALREADY EXISTS (Skipping setup)", model_name, version)
                continue
                
            model = train_model(
                version=version,
                n_estimators=n_est,
                max_depth=max_d,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                family_idx=family_idx,
            )
            metrics = compute_metrics(model, X_test, y_test)

            # Compute training date on the timeline
            training_date = BASE_DATE + DATE_STEP * version_idx

            model_id = registry.register_model(
                model=model,
                metadata={
                    "model_name": model_name,
                    "version": version,
                    "model_type": spec["model_type"],
                    "features": features,
                    "target": spec["target"],
                    "performance_metrics": metrics,
                    "training_date": training_date.isoformat(),
                    "status": "candidate",   # Always register as candidate first
                    "description": description,
                },
            )
            logger.info(
                "   ✅  %s %s | acc=%.3f f1=%.3f | %s",
                model_name, version,
                metrics["accuracy"], metrics["f1"],
                description,
            )
            last_id = model_id
            last_status = status

        # Now set the correct statuses (only one can be active)
        all_versions = registry.list_models(model_name=model_name)
        # Sort by version number
        all_versions.sort(key=lambda m: m.version)

        for m in all_versions:
            # Find the intended status from spec
            intended = next(
                (s for v, _, _, s, _ in spec["versions"] if v == m.version), "retired"
            )
            if intended == "active":
                registry.update_model_status(m.model_id, "active")
            elif intended == "candidate":
                registry.update_model_status(m.model_id, "candidate")
            # retired remains as-is (already default)

        logger.info("   📌  Status set: active & candidates configured")
        logger.info("")

    logger.info("=" * 62)
    logger.info("  ✅  DEMO DATA GENERATION COMPLETE")
    logger.info("")
    logger.info("  Models created:")
    for spec in MODELS:
        logger.info("    • %s (%d versions)", spec["model_name"], len(spec["versions"]))
    logger.info("")
    logger.info("  Now run: python run.py")
    logger.info("  Open:    http://localhost:8000")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
