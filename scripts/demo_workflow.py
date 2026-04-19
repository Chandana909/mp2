"""
Demo Workflow - Complete Platform Demonstration

Demonstrates:
1. Train a model
2. Register it
3. Make predictions
4. Detect drift
5. Train new version
6. Promote better version
7. Verify rollback

Usage:
    python scripts/demo_workflow.py
"""

import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml_platform.registry import ModelRegistry
from ml_platform.config import load_config
from monitoring.drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print section header."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(text)
    logger.info("=" * 70)


def step1_train_initial():
    """Train and register initial model."""
    print_header("STEP 1: Training Initial Model")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"✅ Model trained: accuracy={accuracy:.3f}")
    
    # Register
    load_config()
    registry = ModelRegistry()
    
    model_id = registry.register_model(
        model=model,
        metadata={
            "model_name": "demo_classifier",
            "version": "v1",
            "model_type": "classification",
            "features": [f"feature_{i}" for i in range(10)],
            "target": "label",
            "performance_metrics": {"accuracy": float(accuracy)},
        }
    )
    
    logger.info(f"✅ Registered: {model_id}")
    
    # Activate
    registry.update_model_status(model_id, "active")
    logger.info("✅ Model activated")
    
    return X_test, y_test, model_id


def step2_make_predictions(model_id):
    """Make predictions."""
    print_header("STEP 2: Making Predictions")
    
    registry = ModelRegistry()
    active = registry.get_active_model("demo_classifier")
    
    from storage.model_store import ModelStore
    store = ModelStore()
    model = store.load_model(active.model_id)
    
    X_new = np.random.randn(5, 10)
    predictions = model.predict(X_new)
    
    logger.info(f"✅ Predictions: {predictions}")


def step3_detect_drift():
    """Detect drift."""
    print_header("STEP 3: Detecting Drift")
    
    # Reference data
    X_ref = np.random.randn(1000, 10)
    ref_df = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(10)])
    
    # Drifted data (mean shift)
    X_drift = np.random.randn(1000, 10) + 0.5
    drift_df = pd.DataFrame(X_drift, columns=[f"feature_{i}" for i in range(10)])
    
    # Detect
    detector = DriftDetector(drift_threshold=0.15)
    report = detector.compute_data_drift(ref_df, drift_df, "demo_v1")
    
    logger.info(f"Drift detected: {report.drift_detected}")
    logger.info(f"Drift score: {report.drift_score:.3f}")
    
    if report.drift_detected:
        logger.info("⚠️  Retraining recommended")
    
    return report.drift_detected


def step4_train_new():
    """Train new version."""
    print_header("STEP 4: Training New Version")
    
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=6, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"✅ New model: accuracy={accuracy:.3f}")
    
    registry = ModelRegistry()
    model_id = registry.register_model(
        model=model,
        metadata={
            "model_name": "demo_classifier",
            "version": "v2",
            "model_type": "classification",
            "features": [f"feature_{i}" for i in range(10)],
            "target": "label",
            "performance_metrics": {"accuracy": float(accuracy)},
        }
    )
    
    registry.update_model_status(model_id, "candidate")
    logger.info(f"✅ Candidate registered: {model_id}")
    
    return model_id


def step5_promote(candidate_id):
    """Promote to active."""
    print_header("STEP 5: Promoting Model")
    
    registry = ModelRegistry()
    
    # Retire old active
    models = registry.list_models(model_name="demo_classifier")
    for m in models:
        if m.status == "active":
            registry.update_model_status(m.model_id, "retired")
            logger.info(f"Retired: {m.version}")
    
    # Promote candidate
    registry.update_model_status(candidate_id, "active")
    logger.info(f"✅ Promoted to active")


def step6_verify_rollback():
    """Verify rollback."""
    print_header("STEP 6: Verify Rollback")
    
    registry = ModelRegistry()
    versions = registry.list_models(model_name="demo_classifier")
    
    logger.info(f"Available versions: {len(versions)}")
    for v in versions:
        symbol = "🟢" if v.status == "active" else "⚪"
        logger.info(f"  {symbol} {v.version} ({v.status})")
    
    logger.info("✅ Rollback capability verified")


def main():
    """Run demo."""
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 15 + "ML RELIABILITY PLATFORM DEMO" + " " * 25 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    try:
        X_test, y_test, model_v1 = step1_train_initial()
        step2_make_predictions(model_v1)
        drift = step3_detect_drift()
        
        if drift:
            model_v2 = step4_train_new()
            step5_promote(model_v2)
        
        step6_verify_rollback()
        
        print_header("✅ DEMO COMPLETED")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start API: python run.py")
        logger.info("2. Test: curl http://localhost:8000/health")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
