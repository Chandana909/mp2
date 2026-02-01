"""
Custom exceptions for the ML Reliability Platform.

All platform-specific errors inherit from PlatformError.
Never expose internal errors to API clients; map to meaningful messages.
"""

from typing import Optional


class PlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(PlatformError):
    """Raised when a model or model version is not found in the registry."""

    def __init__(self, message: str, model_id: Optional[str] = None, model_name: Optional[str] = None):
        details = {}
        if model_id:
            details["model_id"] = model_id
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details)


class ValidationError(PlatformError):
    """Raised when request validation fails (missing/invalid features, schema mismatch)."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, details)


class DuplicateModelError(PlatformError):
    """Raised when registering a model version that already exists."""

    def __init__(self, message: str, model_name: Optional[str] = None, version: Optional[str] = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if version:
            details["version"] = version
        super().__init__(message, details)


class StorageError(PlatformError):
    """Raised when storage operations fail."""

    pass


class PromotionError(PlatformError):
    """Raised when model promotion criteria are not met or promotion fails."""

    pass


class RollbackError(PlatformError):
    """Raised when rollback to a previous version fails."""

    pass
