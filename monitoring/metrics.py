"""
Compute monitoring metrics.

Feature statistics (mean, std, min, max for numeric; value counts for categorical),
prediction statistics, and model performance (accuracy/F1/ROC for classification;
RMSE/MAE/R2 for regression).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute per-feature statistics: mean, std, min, max for numerical;
    value_counts for categorical.

    Returns:
        Dict mapping feature name to stats dict.
    """
    result: Dict[str, Any] = {}
    for col in data.columns:
        s = data[col]
        if pd.api.types.is_numeric_dtype(s):
            result[col] = {
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() and s.notna().sum() > 1 else None,
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
                "count": int(s.notna().sum()),
            }
        else:
            result[col] = {
                "value_counts": s.value_counts().head(20).to_dict(),
                "count": int(s.notna().sum()),
            }
    return result


def compute_prediction_statistics(
    predictions: np.ndarray,
    model_type: str,
) -> Dict[str, Any]:
    """
    Compute distribution of predictions: class balance for classification,
    range for regression.
    """
    pred = np.asarray(predictions).ravel()
    if model_type == "classification":
        unique, counts = np.unique(pred, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        return {
            "distribution": dist,
            "num_classes": len(unique),
            "count": len(pred),
        }
    return {
        "mean": float(np.mean(pred)),
        "std": float(np.std(pred)) if len(pred) > 1 else 0.0,
        "min": float(np.min(pred)),
        "max": float(np.max(pred)),
        "count": len(pred),
    }


def track_model_performance(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_type: str,
) -> Dict[str, float]:
    """
    Compute performance metrics: accuracy, F1, ROC-AUC for classification;
    RMSE, MAE, R² for regression.
    """
    y_pred = np.asarray(predictions).ravel()
    y_true = np.asarray(actuals).ravel()
    if model_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            roc_auc_score = None
        out: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if roc_auc_score is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            except Exception:
                out["roc_auc"] = 0.0
        return out
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
