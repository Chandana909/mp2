"""
ML Reliability Platform - Core components.

Provides model registry, schema definitions, validation, audit logging,
and configuration management.
"""

from platform.models import ModelMetadata
from platform.registry import ModelRegistry
from platform.schema import (
    PredictionRequest,
    PredictionResponse,
    ModelRegistrationRequest,
    DriftReport,
)
from platform.config import get_config

__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "PredictionRequest",
    "PredictionResponse",
    "ModelRegistrationRequest",
    "DriftReport",
    "get_config",
]
