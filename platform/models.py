"""
Data structures for the registry (no external dependencies on storage).

ModelMetadata is defined here to avoid circular imports between
platform.registry and storage.metadata_store.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    """Metadata for a registered model version."""

    model_id: str
    model_name: str
    version: str
    model_type: str  # "classification" | "regression"
    features: List[str]
    target: str
    training_date: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None
    status: str = "candidate"  # "active" | "candidate" | "retired"
    artifact_path: str = ""

    def __post_init__(self) -> None:
        if self.performance_metrics is None:
            self.performance_metrics = {}

    def to_dict(self) -> dict:
        """Serialize for JSON/storage."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "features": self.features,
            "target": self.target,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "performance_metrics": self.performance_metrics,
            "status": self.status,
            "artifact_path": self.artifact_path,
        }
