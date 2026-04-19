"""
Rollback to previous model version.

Supports manual rollback by version and optional auto-rollback
when performance drops below threshold.
"""

import logging
from typing import Optional

from ml_platform.exceptions import ModelNotFoundError, RollbackError
from ml_platform.models import ModelMetadata
from ml_platform.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RollbackHandler:
    """
    Handles rollback of active model to a previous version.

    Updates registry status: target version -> active, current active -> retired.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()

    def rollback_to_version(self, model_name: str, target_version: str) -> None:
        """
        Set the given version as active and retire the current active version.

        Raises:
            ModelNotFoundError: If model_name or target_version not found.
            RollbackError: If target is already active or update fails.
        """
        versions = self._registry.list_models(model_name=model_name)
        if not versions:
            raise ModelNotFoundError(f"No models found for: {model_name}", model_name=model_name)
        target_meta: Optional[ModelMetadata] = None
        active_meta: Optional[ModelMetadata] = None
        for m in versions:
            if m.version == target_version:
                target_meta = m
            if m.status == "active":
                active_meta = m
        if target_meta is None:
            raise ModelNotFoundError(
                f"Version not found: {model_name}@{target_version}",
                model_name=model_name,
            )
        if target_meta.status == "active":
            raise RollbackError(f"Version {target_version} is already active")
        # Retire current active
        if active_meta and active_meta.model_id != target_meta.model_id:
            self._registry.update_model_status(active_meta.model_id, "retired")
        # Activate target
        self._registry.update_model_status(target_meta.model_id, "active")
        logger.info(
            "Rollback completed",
            extra={"model_name": model_name, "target_version": target_version},
        )

    def auto_rollback_on_performance_drop(
        self,
        model_name: str,
        performance_threshold: float,
        current_metric: Optional[float] = None,
        metric_name: str = "accuracy",
    ) -> bool:
        """
        If current active model performance is below threshold, rollback to last good version.

        Requires current_metric to be passed (from monitoring). If current_metric is None,
        no rollback is performed. For accuracy/F1, higher is better; for RMSE, lower is better
        so pass negative RMSE or use a wrapper. This method treats higher metric as better.

        Returns:
            True if rollback was performed, False otherwise.
        """
        if current_metric is None:
            return False
        if current_metric >= performance_threshold:
            return False
        versions = self._registry.list_models(model_name=model_name)
        active = next((m for m in versions if m.status == "active"), None)
        if not active:
            return False
        # Find another version (e.g. retired) with better metrics to rollback to
        best_other: Optional[ModelMetadata] = None
        best_val: Optional[float] = None
        for m in versions:
            if m.model_id == active.model_id:
                continue
            val = (m.performance_metrics or {}).get(metric_name)
            if val is None:
                continue
            if best_val is None or val > best_val:
                best_val = val
                best_other = m
        if best_other is None or best_val is None or best_val < performance_threshold:
            return False
        try:
            self.rollback_to_version(model_name, best_other.version)
            return True
        except Exception as e:
            logger.exception("Auto rollback failed: %s", e)
            return False
