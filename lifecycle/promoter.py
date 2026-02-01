"""
Promote candidate models to active.

Checks promotion criteria, updates status (active -> retired, candidate -> active),
and supports optional safe promotion with A/B traffic split.
"""

import logging
from typing import Any, Dict, Optional

from platform.exceptions import PromotionError
from platform.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Promotes a candidate model to active after criteria check.

    Updates registry: current active -> retired, candidate -> active.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()

    def promote_to_active(self, candidate_id: str, evaluation_report: Dict[str, Any]) -> bool:
        """
        Promote candidate to active if evaluation_report indicates candidate is better.

        evaluation_report should contain "candidate_is_better" (bool) from evaluator.compare_models.

        Returns:
            True if promotion was performed, False if criteria not met.
        """
        if not evaluation_report.get("candidate_is_better", False):
            raise PromotionError("Promotion criteria not met: candidate is not better than active")
        meta = self._registry.get_model_metadata(candidate_id)
        if meta.status == "active":
            return True
        if meta.status != "candidate":
            raise PromotionError(f"Model status is '{meta.status}', not candidate")
        # Retire current active for this model_name
        versions = self._registry.list_models(model_name=meta.model_name)
        for m in versions:
            if m.status == "active" and m.model_id != candidate_id:
                self._registry.update_model_status(m.model_id, "retired")
        self._registry.update_model_status(candidate_id, "active")
        logger.info("Model promoted to active", extra={"model_id": candidate_id, "model_name": meta.model_name})
        return True

    def safe_promote_with_ab_test(
        self,
        candidate_id: str,
        traffic_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Placeholder for gradual traffic rollout.

        In a full implementation: route traffic_split of requests to candidate,
        monitor performance, then full promote if stable. Here we only document
        the contract and return a stub.
        """
        meta = self._registry.get_model_metadata(candidate_id)
        return {
            "status": "stub",
            "message": "A/B promotion requires traffic routing and metrics; integrate with load balancer",
            "candidate_id": candidate_id,
            "model_name": meta.model_name,
            "traffic_split": traffic_split,
        }
