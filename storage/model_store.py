"""
Model artifact storage.

Saves and loads model artifacts using joblib.
Paths are under config storage.models_path; model_id used for filenames.
"""

import logging
from pathlib import Path
from typing import Any

import joblib

from platform.config import get_storage_paths
from platform.exceptions import StorageError

logger = logging.getLogger(__name__)


def _safe_model_path(model_id: str) -> str:
    """Return filesystem-safe path segment from model_id."""
    return model_id.replace("/", "_").replace("\\", "_")


class ModelStore:
    """Persist and load model artifacts (joblib)."""

    def __init__(self, models_path: Path | None = None):
        if models_path is None:
            models_path, _, _ = get_storage_paths()
        self._root = Path(models_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, model_id: str, path: str | None = None) -> str:
        """
        Save model to disk using joblib.

        Args:
            model: Trained model object (sklearn-compatible).
            model_id: Unique model identifier (used in filename).
            path: Optional subpath under models_path. If None, uses model_id.

        Returns:
            Absolute path where model was saved.

        Raises:
            StorageError: On write failure.
        """
        segment = path or _safe_model_path(model_id)
        full_path = self._root / f"{segment}.joblib"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, full_path)
            logger.info("Model saved", extra={"model_id": model_id, "path": str(full_path)})
            return str(full_path)
        except Exception as e:
            logger.exception("Failed to save model")
            raise StorageError(f"Failed to save model: {e}") from e

    def load_model(self, model_id: str) -> Any:
        """
        Load model from disk.

        Args:
            model_id: Unique model identifier (used to find .joblib file).

        Returns:
            Loaded model object.

        Raises:
            StorageError: If file not found or load fails.
        """
        segment = _safe_model_path(model_id)
        full_path = self._root / f"{segment}.joblib"
        if not full_path.exists():
            raise StorageError(f"Model artifact not found: {model_id}", details={"path": str(full_path)})
        try:
            return joblib.load(full_path)
        except Exception as e:
            logger.exception("Failed to load model")
            raise StorageError(f"Failed to load model: {e}") from e

    def exists(self, model_id: str) -> bool:
        """Return True if artifact for model_id exists."""
        segment = _safe_model_path(model_id)
        return (self._root / f"{segment}.joblib").exists()

    def delete(self, model_id: str) -> None:
        """Remove model artifact if present. No-op if not found."""
        segment = _safe_model_path(model_id)
        path = self._root / f"{segment}.joblib"
        if path.exists():
            path.unlink()
            logger.info("Model artifact deleted", extra={"model_id": model_id})
