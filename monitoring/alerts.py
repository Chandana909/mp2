"""
Alert triggering logic.

Triggers retrain when drift exceeds threshold; can be extended for
notifications (email, Slack, etc.).
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from ml_platform.schema import DriftReport

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Evaluates drift reports and performance against config thresholds.
    Calls optional handlers (e.g. trigger retrain, send notification).
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        on_drift: Optional[Callable[[DriftReport], None]] = None,
    ):
        self.drift_threshold = drift_threshold
        self._on_drift = on_drift or (lambda _: None)

    def check_drift(self, report: DriftReport) -> bool:
        """
        If drift_detected, log and call on_drift handler.
        Returns True if drift was detected.
        """
        if not report.drift_detected:
            return False
        logger.warning(
            "Drift detected",
            extra={
                "model_id": report.model_id,
                "drift_score": report.drift_score,
                "feature_drifts": report.feature_drifts,
            },
        )
        try:
            self._on_drift(report)
        except Exception as e:
            logger.exception("Drift handler failed: %s", e)
        return True

    def trigger_retrain(self, model_name: str, report: DriftReport) -> None:
        """
        Placeholder: in production would enqueue retrain job or call ModelTrainer.retrain_active_model.
        """
        logger.info(
            "Retrain triggered (stub)",
            extra={"model_name": model_name, "model_id": report.model_id, "drift_score": report.drift_score},
        )
