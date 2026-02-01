"""
Audit logging for predictions.

Logs every prediction request and response to storage (JSONL) with
timestamp, model_version, input_features, prediction, confidence.
Queryable by model_id and date range.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from platform.schema import PredictionRequest, PredictionResponse
from storage.audit_store import AuditStore
from platform.config import get_storage_paths

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logs prediction requests and responses for audit trail.

    Writes to JSONL (one JSON per line) under config audit_logs_path.
    Supports query by model_id, start_date, end_date.
    """

    def __init__(self, audit_store: Optional[AuditStore] = None):
        _, _, audit_path = get_storage_paths()
        self._store = audit_store or AuditStore(audit_path)

    def log_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store one audit entry: timestamp, model info, inputs, outputs.

        request and response can be Pydantic models or dict-like.
        """
        now = datetime.utcnow()
        if hasattr(request, "model_dump"):
            req_dict = request.model_dump()
        elif hasattr(request, "dict"):
            req_dict = request.dict()
        else:
            req_dict = dict(request)
        if hasattr(response, "model_dump"):
            resp_dict = response.model_dump()
        elif hasattr(response, "dict"):
            resp_dict = response.dict()
        else:
            resp_dict = dict(response)
        entry = {
            "timestamp": now.isoformat() + "Z",
            "model_id": resp_dict.get("model_id"),
            "model_version": resp_dict.get("model_version"),
            "model_name": req_dict.get("model_name"),
            "input_features": req_dict.get("features"),
            "prediction": resp_dict.get("prediction"),
            "confidence": resp_dict.get("confidence"),
            "request_id": extra.get("request_id") if extra else None,
        }
        if extra:
            for k, v in extra.items():
                if k != "request_id" and k not in entry:
                    entry[k] = v
        self._store.save_audit_log(entry)
        logger.debug("Audit log written", extra={"model_id": entry.get("model_id")})

    def get_audit_logs(
        self,
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> List[dict]:
        """Query audit logs with filters. Returns list of log entries."""
        return self._store.query_audit_logs(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
