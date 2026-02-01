"""
Model metadata storage.

Saves and queries model metadata as JSON files under config storage.metadata_path.
Each model version is one JSON file keyed by model_id.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from platform.config import get_storage_paths
from platform.exceptions import StorageError
from platform.models import ModelMetadata

logger = logging.getLogger(__name__)


def _metadata_to_dict(m: ModelMetadata) -> dict:
    """Serialize ModelMetadata to JSON-safe dict."""
    return {
        "model_id": m.model_id,
        "model_name": m.model_name,
        "version": m.version,
        "model_type": m.model_type,
        "features": m.features,
        "target": m.target,
        "training_date": m.training_date.isoformat() if m.training_date else None,
        "performance_metrics": m.performance_metrics,
        "status": m.status,
        "artifact_path": m.artifact_path,
    }


def _dict_to_metadata(d: dict) -> ModelMetadata:
    """Deserialize dict to ModelMetadata."""
    training_date = d.get("training_date")
    if isinstance(training_date, str):
        try:
            training_date = datetime.fromisoformat(training_date.replace("Z", "+00:00"))
        except ValueError:
            training_date = None
    return ModelMetadata(
        model_id=d["model_id"],
        model_name=d["model_name"],
        version=d["version"],
        model_type=d["model_type"],
        features=d.get("features", []),
        target=d.get("target", ""),
        training_date=training_date,
        performance_metrics=d.get("performance_metrics", {}),
        status=d.get("status", "candidate"),
        artifact_path=d.get("artifact_path", ""),
    )


def _safe_id_path(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_") + ".json"


class MetadataStore:
    """Persist and query model metadata (JSON files)."""

    def __init__(self, metadata_path: Path | None = None):
        if metadata_path is None:
            _, metadata_path, _ = get_storage_paths()
        self._root = Path(metadata_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: ModelMetadata) -> None:
        """
        Save model metadata to a JSON file.

        Raises:
            StorageError: On write failure.
        """
        path = self._root / _safe_id_path(metadata.model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_metadata_to_dict(metadata), f, indent=2)
            logger.debug("Metadata saved", extra={"model_id": metadata.model_id})
        except Exception as e:
            logger.exception("Failed to save metadata")
            raise StorageError(f"Failed to save metadata: {e}") from e

    def load_metadata(self, model_id: str) -> ModelMetadata:
        """
        Load metadata for a model_id.

        Raises:
            StorageError: If file not found or invalid.
        """
        path = self._root / _safe_id_path(model_id)
        if not path.exists():
            raise StorageError(f"Metadata not found: {model_id}", details={"path": str(path)})
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return _dict_to_metadata(d)
        except Exception as e:
            logger.exception("Failed to load metadata")
            raise StorageError(f"Failed to load metadata: {e}") from e

    def query_metadata(self, filters: dict) -> List[ModelMetadata]:
        """
        List metadata files and filter by model_name, status, etc.

        Supported filters: model_name (str), status (str), model_type (str).
        """
        results: List[ModelMetadata] = []
        model_name_filter = filters.get("model_name")
        status_filter = filters.get("status")
        model_type_filter = filters.get("model_type")

        for path in self._root.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if model_name_filter is not None and d.get("model_name") != model_name_filter:
                    continue
                if status_filter is not None and d.get("status") != status_filter:
                    continue
                if model_type_filter is not None and d.get("model_type") != model_type_filter:
                    continue
                results.append(_dict_to_metadata(d))
            except Exception as e:
                logger.warning("Skip invalid metadata file %s: %s", path, e)
        return results

    def list_all_model_ids(self) -> List[str]:
        """Return all model_ids that have metadata (without loading full metadata)."""
        ids = []
        for path in self._root.glob("*.json"):
            stem = path.stem
            ids.append(stem)
        return ids

    def delete_metadata(self, model_id: str) -> None:
        """Remove metadata file if present."""
        path = self._root / _safe_id_path(model_id)
        if path.exists():
            path.unlink()
            logger.info("Metadata deleted", extra={"model_id": model_id})
