"""
Model evaluation and comparison.

Evaluates a single model on test data and compares candidate vs active
with configurable criteria for promotion.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from platform.models import ModelMetadata
from platform.registry import ModelRegistry

logger = logging.getLogger(__name__)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        metrics["f1_score"] = 0.0
    if y_proba is not None and roc_auc_score is not None:
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


class ModelEvaluator:
    """
    Evaluates models on test data and compares candidate vs active.

    Returns standardized metric dicts and recommendation for promotion.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()

    def evaluate_model(
        self,
        model: Any,
        test_data: pd.DataFrame,
        model_type: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute performance metrics on test_data.

        Args:
            model: Trained model.
            test_data: DataFrame with features and target.
            model_type: "classification" or "regression".
            target_column: Name of target column in test_data.
            feature_columns: Optional list of feature columns; if None, use all except target.

        Returns:
            Dict of metric name -> value.
        """
        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not in test_data")
        if feature_columns is None:
            feature_columns = [c for c in test_data.columns if c != target_column]
        X = test_data[feature_columns]
        y_true = test_data[target_column].values
        y_pred = model.predict(X)
        y_proba = None
        if model_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                pass
        if model_type == "classification":
            return _classification_metrics(y_true, y_pred, y_proba)
        return _regression_metrics(y_true, y_pred)

    def compare_models(
        self,
        candidate_id: str,
        active_id: str,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        criteria: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate both models on the same test set and recommend promotion.

        criteria: list of dicts with "metric" and "improvement_threshold".
                  For accuracy/f1/r2 higher is better; for rmse/mae lower is better.
                  Default: [{"metric": "accuracy", "improvement_threshold": 0.02}].

        Returns:
            Dict with candidate_metrics, active_metrics, candidate_is_better, recommendation.
        """
        if criteria is None:
            criteria = [{"metric": "accuracy", "improvement_threshold": 0.02}]
        candidate = self._registry.get_model(candidate_id)
        active = self._registry.get_model(active_id)
        c_meta = self._registry.get_model_metadata(candidate_id)
        a_meta = self._registry.get_model_metadata(active_id)
        model_type = c_meta.model_type
        candidate_metrics = self.evaluate_model(
            candidate, test_data, model_type, target_column, feature_columns
        )
        active_metrics = self.evaluate_model(
            active, test_data, model_type, target_column, feature_columns
        )
        higher_is_better = {"accuracy", "f1_score", "roc_auc", "r2"}
        candidate_is_better = True
        for c in criteria:
            metric_name = c.get("metric", "accuracy")
            threshold = float(c.get("improvement_threshold", 0.0))
            c_val = candidate_metrics.get(metric_name)
            a_val = active_metrics.get(metric_name)
            if c_val is None or a_val is None:
                candidate_is_better = False
                break
            if metric_name in higher_is_better:
                if c_val - a_val < threshold:
                    candidate_is_better = False
                    break
            else:
                if a_val - c_val < threshold:
                    candidate_is_better = False
                    break
        recommendation = "promote" if candidate_is_better else "do_not_promote"
        return {
            "candidate_metrics": candidate_metrics,
            "active_metrics": active_metrics,
            "candidate_is_better": candidate_is_better,
            "recommendation": recommendation,
        }
