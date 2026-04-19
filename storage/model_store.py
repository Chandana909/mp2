"""
Model artifact storage.

Saves and loads model artifacts using joblib.
Paths are under config storage.models_path; model_id used for filenames.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import threading
from collections import OrderedDict
from ml_platform.config import get_storage_paths
from ml_platform.exceptions import StorageError

logger = logging.getLogger(__name__)


def _safe_model_path(model_id: str) -> str:
    """Return filesystem-safe path segment from model_id."""
    return model_id.replace("/", "_").replace("\\", "_")


class ModelStore:
    """Persist and load model artifacts (joblib)."""

    def __init__(self, models_path: Optional[Path] = None, cache_size: int = 50):
        if models_path is None:
            models_path, _, _ = get_storage_paths()
        self._root = Path(models_path)
        self._root.mkdir(parents=True, exist_ok=True)
        
        # Industry-grade LRU Cache to manage RAM usage for 1000+ models
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def save_model(self, model: Any, model_id: str, path: Optional[str] = None) -> str:
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
        logger.debug("Saving model %s to %s", model_id, full_path)
        try:
            joblib.dump(model, full_path)
            logger.info("Model saved successfully: %s", model_id, extra={"model_id": model_id, "path": str(full_path)})
            return str(full_path)
        except Exception as e:
            logger.error("Failed to save model %s: %s", model_id, e, exc_info=True)
            raise StorageError(f"Failed to save model: {e}") from e

    def load_model(self, model_id: str) -> Any:
        """
        Load model from disk with LRU caching.
        """
        with self._lock:
            if model_id in self._cache:
                # Move to end (mark as most recently used)
                self._cache.move_to_end(model_id)
                return self._cache[model_id]

        segment = _safe_model_path(model_id)
        full_path = self._root / f"{segment}.joblib"
        
        if not full_path.exists():
            raise StorageError(f"Model artifact not found: {model_id}", details={"path": str(full_path)})
            
        try:
            model = joblib.load(full_path)
            
            with self._lock:
                self._cache[model_id] = model
                self._cache.move_to_end(model_id)
                # Evict oldest if cache is full
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                    
            logger.debug("Model loaded and cached: %s", model_id)
            return model
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_id, e, exc_info=True)
            raise StorageError(f"Failed to load model: {e}") from e

    def clear_cache(self) -> None:
        """Evict all items from RAM."""
        with self._lock:
            self._cache.clear()

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
