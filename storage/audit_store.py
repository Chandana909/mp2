"""
Audit log storage.

Appends prediction audit entries to JSONL files under config storage.audit_logs_path.
Query by model_id, start_date, end_date. Supports daily rotation by filename.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ml_platform.config import get_storage_paths
from ml_platform.exceptions import StorageError

logger = logging.getLogger(__name__)


def _audit_log_dir() -> Path:
    _, _, audit_path = get_storage_paths()
    return Path(audit_path)


def _today_log_path() -> Path:
    """Path for today's audit log (one file per day)."""
    root = _audit_log_dir()
    root.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    return root / f"audit_{date_str}.jsonl"


class AuditStore:
    """Append-only audit log (JSONL) with query support."""

    def __init__(self, audit_logs_path: Optional[Path] = None):
        if audit_logs_path is None:
            audit_logs_path = _audit_log_dir()
        self._root = Path(audit_logs_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def save_audit_log(self, log_entry: dict) -> None:
        """
        Append one JSON line to the current day's audit file.

        Raises:
            StorageError: On write failure.
        """
        path = self._root / f"audit_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Appending audit log to %s", path)
        try:
            line = json.dumps(log_entry, default=str) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error("Failed to write audit log: %s", e, exc_info=True)
            raise StorageError(f"Failed to write audit log: {e}") from e

    def query_audit_logs(
        self,
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> List[dict]:
        """
        Read audit logs with optional filters.

        Scans JSONL files in audit_logs_path for the date range and model_id.
        Returns entries in chronological order, capped by limit.
        """
        entries: List[dict] = []
        start_ts = start_date.timestamp() if start_date else 0
        end_ts = end_date.timestamp() if end_date else float("inf")

        pattern = "audit_*.jsonl"
        files = sorted(self._root.glob(pattern))
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ts = obj.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                ts = dt.timestamp()
                            except ValueError:
                                ts = 0
                        elif hasattr(ts, "timestamp"):
                            ts = ts.timestamp()
                        else:
                            ts = 0
                        if ts < start_ts or ts > end_ts:
                            continue
                        if model_id is not None and obj.get("model_id") != model_id:
                            continue
                        entries.append(obj)
                        if len(entries) >= limit:
                            return entries
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning("Error reading audit file %s: %s", path, e)
        return entries
