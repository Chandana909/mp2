"""
ML Reliability Platform - Core Module

Provides model registry, configuration, validation, and schema definitions.
"""

from ml_platform.registry import ModelRegistry
from ml_platform.config import load_config, get_config, get_storage_paths
from ml_platform.exceptions import (
    PlatformError,
    ModelNotFoundError,
    ValidationError,
    DuplicateModelError,
    StorageError,
    PromotionError,
    RollbackError,
)
from ml_platform.models import ModelMetadata
from ml_platform.schema import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    DriftReport,
    ModelRegistrationRequest,
    RollbackRequest,
)
from ml_platform.validator import InputValidator
from ml_platform.audit import AuditLogger

__all__ = [
    # Registry
    "ModelRegistry",
    # Config
    "load_config",
    "get_config",
    "get_storage_paths",
    # Exceptions
    "PlatformError",
    "ModelNotFoundError",
    "ValidationError",
    "DuplicateModelError",
    "StorageError",
    "PromotionError",
    "RollbackError",
    # Models
    "ModelMetadata",
    # Schema
    "PredictionRequest",
    "PredictionResponse",
    "ModelInfo",
    "DriftReport",
    "ModelRegistrationRequest",
    "RollbackRequest",
    # Utilities
    "InputValidator",
    "AuditLogger",
]
