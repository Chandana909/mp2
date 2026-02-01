"""
Pydantic models for API contracts and validation.

Defines request/response schemas for predictions, model registration,
and drift reports. Used by FastAPI for serialization and validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for POST /predict."""

    model_name: str = Field(..., description="Name of the model to use for prediction")
    features: Dict[str, Any] = Field(..., description="Feature names and values")

    model_config = {"json_schema_extra": {"example": {"model_name": "fraud_detector", "features": {"amount": 250.0, "merchant_id": "merch_123", "time_of_day": 14}}}}


class PredictionResponse(BaseModel):
    """Response from POST /predict."""

    model_id: str = Field(..., description="Unique model identifier")
    model_version: str = Field(..., description="Model version used")
    prediction: Union[int, float, str] = Field(..., description="Prediction value")
    confidence: Optional[float] = Field(None, description="Confidence score when available")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    model_config = {"json_schema_extra": {"example": {"model_id": "fraud_detector_v1_abc", "model_version": "v1", "prediction": 0, "confidence": 0.87, "timestamp": "2026-02-01T10:30:00Z"}}}


class ModelRegistrationRequest(BaseModel):
    """Request body for registering a new model version."""

    model_name: str = Field(..., description="Logical name of the model")
    version: str = Field(..., description="Version string, e.g. v1, v2")
    model_type: str = Field(..., description="classification or regression")
    features: List[str] = Field(..., description="List of feature names")
    target: str = Field(..., description="Target variable name")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")

    model_config = {"json_schema_extra": {"example": {"model_name": "fraud_detector", "version": "v1", "model_type": "classification", "features": ["amount", "merchant_id"], "target": "is_fraud", "performance_metrics": {"accuracy": 0.94, "f1": 0.91}}}}


class DriftReport(BaseModel):
    """Report from drift detection."""

    model_id: str = Field(..., description="Model identifier")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Aggregate drift score in [0, 1]")
    feature_drifts: Dict[str, float] = Field(default_factory=dict, description="Per-feature drift scores")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Report timestamp")

    model_config = {"json_schema_extra": {"example": {"model_id": "fraud_detector_v1_abc", "drift_detected": True, "drift_score": 0.22, "feature_drifts": {"amount": 0.15, "time_of_day": 0.31}, "timestamp": "2026-02-01T10:30:00Z"}}


class ModelInfo(BaseModel):
    """Summary of a registered model for listing endpoints."""

    model_id: str
    model_name: str
    version: str
    model_type: str
    status: str
    training_date: Optional[datetime] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class RollbackRequest(BaseModel):
    """Request body for POST /models/{model_name}/rollback."""

    target_version: str = Field(..., description="Version to rollback to")
