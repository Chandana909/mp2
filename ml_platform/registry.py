"""
Central model registry managing all model versions.

Provides registration, versioning, status (active/candidate/retired),
and lookup by model_id or active model name. Thread-safe operations.
"""

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ml_platform.config import get_storage_paths
from ml_platform.exceptions import ModelNotFoundError, DuplicateModelError, StorageError
from ml_platform.models import ModelMetadata
from storage.model_store import ModelStore
from storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for model versions.

    Stores metadata as JSON in storage/metadata/ and model artifacts via joblib in models/.
    Maintains index of active models per model_name. Operations are thread-safe.
    """

    def __init__(
        self,
        model_store: Optional[ModelStore] = None,
        metadata_store: Optional[MetadataStore] = None,
    ):
        models_path, metadata_path, _ = get_storage_paths()
        self._model_store = model_store or ModelStore(models_path)
        self._metadata_store = metadata_store or MetadataStore(metadata_path)
        self._lock = threading.RLock()

    def register_model(self, model: Any, metadata: dict) -> str:
        """
        Register a new model version. Persists artifact and metadata.

        Args:
            model: Trained model object (sklearn-compatible).
            metadata: Dict with model_name, version, model_type, features, target,
                      optional performance_metrics, training_date.

        Returns:
            model_id: Unique identifier for this version.

        Raises:
            DuplicateModelError: If same model_name + version already exists.
            StorageError: On save failure.
        """
        with self._lock:
            model_name = metadata.get("model_name") or ""
            version = metadata.get("version") or "v1"
            existing = self._metadata_store.query_metadata({"model_name": model_name})
            for m in existing:
                if m.version == version:
                    raise DuplicateModelError(
                        f"Model version already exists: {model_name} {version}",
                        model_name=model_name,
                        version=version,
                    )
            model_id = f"{model_name}_{version}_{uuid.uuid4().hex[:8]}"
            training_date = metadata.get("training_date")
            if isinstance(training_date, str):
                try:
                    training_date = datetime.fromisoformat(training_date.replace("Z", "+00:00"))
                except ValueError:
                    training_date = datetime.utcnow()
            elif training_date is None:
                training_date = datetime.utcnow()
            meta = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                model_type=metadata.get("model_type", "classification"),
                features=metadata.get("features", []),
                target=metadata.get("target", ""),
                training_date=training_date,
                performance_metrics=metadata.get("performance_metrics") or {},
                status=metadata.get("status", "candidate"),
                artifact_path="",
            )
            artifact_path = self._model_store.save_model(model, model_id)
            meta.artifact_path = artifact_path
            self._metadata_store.save_metadata(meta)
            logger.info("Model registered", extra={"model_id": model_id, "model_name": model_name, "version": version})
            return model_id

    def get_model(self, model_id: str) -> Any:
        """
        Load model object by model_id.

        Raises:
            ModelNotFoundError: If model_id not in registry or artifact missing.
        """
        with self._lock:
            try:
                self._metadata_store.load_metadata(model_id)
            except StorageError as e:
                raise ModelNotFoundError(f"Model not found: {model_id}", model_id=model_id) from e
            try:
                return self._model_store.load_model(model_id)
            except StorageError as e:
                raise ModelNotFoundError(f"Model artifact not found: {model_id}", model_id=model_id) from e

    def get_active_model(self, model_name: str) -> Any:
        """
        Load the currently active model for model_name.

        Raises:
            ModelNotFoundError: If no active model for model_name.
        """
        with self._lock:
            meta = self.get_active_metadata(model_name)
            return self._model_store.load_model(meta.model_id)

    def get_active_metadata(self, model_name: str) -> ModelMetadata:
        """Return metadata for the active model of model_name."""
        with self._lock:
            candidates = self._metadata_store.query_metadata({"model_name": model_name, "status": "active"})
            if not candidates:
                raise ModelNotFoundError(f"No active model for: {model_name}", model_name=model_name)
            return candidates[0]

    def list_models(self, model_name: Optional[str] = None) -> List[ModelMetadata]:
        """List all registered models, optionally filtered by model_name."""
        with self._lock:
            filters = {}
            if model_name is not None:
                filters["model_name"] = model_name
            if filters:
                return self._metadata_store.query_metadata(filters)
            return self._metadata_store.query_metadata({})

    def update_model_status(self, model_id: str, status: str) -> None:
        """
        Set model status to active, candidate, or retired.

        When setting one version to active, other versions of the same model_name
        that were active are set to retired.
        """
        with self._lock:
            meta = self._metadata_store.load_metadata(model_id)
            if status == "active":
                # Retire other active versions of same model_name
                others = self._metadata_store.query_metadata({"model_name": meta.model_name})
                for m in others:
                    if m.model_id != model_id and m.status == "active":
                        m.status = "retired"
                        self._metadata_store.save_metadata(m)
            meta.status = status
            self._metadata_store.save_metadata(meta)
            logger.info("Model status updated", extra={"model_id": model_id, "status": status})

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Return metadata for model_id."""
        with self._lock:
            return self._metadata_store.load_metadata(model_id)
