"""
Monitoring & drift detection module.

Provides data drift detection, metrics computation, and alert logic.
"""

from monitoring.drift_detector import DriftDetector
from monitoring.metrics import (
    compute_feature_statistics,
    compute_prediction_statistics,
    track_model_performance,
)
from monitoring.alerts import AlertManager

__all__ = [
    "DriftDetector",
    "compute_feature_statistics",
    "compute_prediction_statistics",
    "track_model_performance",
    "AlertManager",
]
