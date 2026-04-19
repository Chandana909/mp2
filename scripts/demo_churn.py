"""
Customer Churn Demo

Demonstrates the platform by creating 5 versions of a customer churn model,
registering them all, setting one as active, and making a prediction programmatically.

Usage:
    python scripts/demo_churn.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from ml_platform.registry import ModelRegistry
from ml_platform.config import load_config
from storage.model_store import ModelStore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("🚀 CUSTOMER CHURN PREDICTION - 5 VERSION DEMO")
    logger.info("="*60)
    
    # 1. Generate Synthetic Customer Data (Features: Age, Tenure, MonthlyCharge, TotalCharge, SupportCalls)
    X, y = make_classification(
        n_samples=2000, 
        n_features=5, 
        n_informative=4, 
        n_redundant=1,
        random_state=42,
        class_sep=0.8
    )
    
    feature_names = ["age", "tenure_months", "monthly_charge", "total_charge", "support_calls"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    load_config()
    registry = ModelRegistry()
    
    model_name = "customer_churn"
    
    # Retire any existing active models for this name to start fresh
    try:
        existing = registry.list_models(model_name=model_name)
        for m in existing:
            registry.update_model_status(m.model_id, "retired")
    except Exception:
        pass
        
    last_registered_id = None
    
    # 2. Train 5 Different Versions
    logger.info("\n⚙️ Training 5 versions of the Churn Model...")
    
    for version in range(1, 6):
        # We simulate "improvements" by increasing complexity
        n_trees = 10 * version
        model = RandomForestClassifier(n_estimators=n_trees, max_depth=version+2, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        v_str = f"v{version}"
        model_id = registry.register_model(
            model=model,
            metadata={
                "model_name": model_name,
                "version": v_str,
                "model_type": "binary_classification",
                "features": feature_names,
                "target": "churn",
                "performance_metrics": {
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec)
                }
            }
        )
        
        last_registered_id = model_id
        logger.info(f"✅ Registered {model_name} {v_str} | ID: {model_id} | Accuracy: {acc:.3f}")
        
    # 3. Promote v5 to Active
    logger.info(f"\n🚀 Promoting v5 (ID: {last_registered_id}) to ACTIVE status...")
    registry.update_model_status(last_registered_id, "active")
    
    # 4. Make a sample prediction using the active model
    logger.info("\n🔮 Making a programmatic Customer Churn Prediction...")
    
    active_meta = registry.get_active_metadata(model_name)
    store = ModelStore()
    active_model = store.load_model(active_meta.model_id)
    
    # Sample new customer profile:
    # age=35, tenure=2, monthly=85.0, total=170.0, support=4
    sample_customer = np.array([[35.0, 2.0, 85.0, 170.0, 4.0]])
    prediction = active_model.predict(sample_customer)[0]
    prob = active_model.predict_proba(sample_customer)[0]
    
    logger.info(f"Active Model ID: {active_meta.model_id} ({active_meta.version})")
    logger.info(f"Customer Data: Age=35, Tenure=2m, Monthly=$85, Total=$170, SupportCalls=4")
    logger.info(f"Prediction: {'Will Churn 🔴' if prediction == 1 else 'Will Stay 🟢'}")
    logger.info(f"Confidence: {max(prob)*100:.1f}%")

    logger.info("\n" + "="*60)
    logger.info("✅ Demo complete! You can now test this dynamically via the API.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
