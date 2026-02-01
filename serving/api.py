"""
FastAPI application for serving predictions.

Endpoints: /predict, /models, /models/{model_name}, /models/{model_name}/versions,
POST /models/{model_name}/rollback, /health, /metrics/{model_name}.
Uses dependency injection for registry and validator; request logging middleware.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from platform.exceptions import (
    ModelNotFoundError,
    ValidationError as PlatformValidationError,
    PlatformError,
)
from platform.models import ModelMetadata
from platform.schema import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    RollbackRequest,
)
from platform.registry import ModelRegistry
from platform.validator import InputValidator
from platform.audit import AuditLogger
from platform.config import load_config, get_config
from serving.predictor import ModelPredictor
from lifecycle.rollback import RollbackHandler

logger = logging.getLogger(__name__)

# Globals set by lifespan
_registry: Optional[ModelRegistry] = None
_validator: Optional[InputValidator] = None
_audit: Optional[AuditLogger] = None
_predictor: Optional[ModelPredictor] = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        load_config()
        _reg = ModelRegistry()
        globals()["_registry"] = _reg
        return _reg
    return _registry


def get_validator() -> InputValidator:
    if _validator is None:
        globals()["_validator"] = InputValidator()
        return _validator
    return _validator


def get_audit() -> AuditLogger:
    if _audit is None:
        globals()["_audit"] = AuditLogger()
        return _audit
    return _audit


def get_predictor() -> ModelPredictor:
    if _predictor is None:
        globals()["_predictor"] = ModelPredictor()
        return _predictor
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()
    globals()["_registry"] = ModelRegistry()
    globals()["_validator"] = InputValidator()
    globals()["_audit"] = AuditLogger()
    globals()["_predictor"] = ModelPredictor()
    yield
    # Teardown if needed
    pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Reliability Platform",
        description="Model registry, serving, monitoring, and lifecycle management",
        version=get_config("platform.version") or "1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        logger.info("Request %s %s", request.method, request.url.path, extra={"request_id": request_id})
        response = await call_next(request)
        return response

    @app.exception_handler(PlatformError)
    def platform_error_handler(request: Request, exc: PlatformError):
        if isinstance(exc, ModelNotFoundError):
            return JSONResponse(status_code=404, content={"detail": exc.message, "details": exc.details})
        if isinstance(exc, PlatformValidationError):
            return JSONResponse(status_code=422, content={"detail": exc.message, "details": exc.details})
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})

    @app.exception_handler(HTTPException)
    def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: PredictionRequest,
        registry: ModelRegistry = Depends(get_registry),
        validator: InputValidator = Depends(get_validator),
        audit: AuditLogger = Depends(get_audit),
        predictor: ModelPredictor = Depends(get_predictor),
    ):
        """Run prediction using the active model for the given model_name."""
        try:
            meta = registry.get_active_metadata(request.model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        validator.validate_prediction_request(request, meta)
        model = registry.get_model(meta.model_id)
        pred, confidence = predictor.predict(model, request.features, meta)
        response = PredictionResponse(
            model_id=meta.model_id,
            model_version=meta.version,
            prediction=pred,
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )
        audit.log_prediction(request, response)
        return response

    @app.get("/models", response_model=List[ModelInfo])
    def list_models(registry: ModelRegistry = Depends(get_registry)):
        """List all registered models."""
        metas = registry.list_models()
        return [
            ModelInfo(
                model_id=m.model_id,
                model_name=m.model_name,
                version=m.version,
                model_type=m.model_type,
                status=m.status,
                training_date=m.training_date,
                performance_metrics=m.performance_metrics or {},
            )
            for m in metas
        ]

    @app.get("/models/{model_name}", response_model=ModelInfo)
    def get_model_metadata(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Get metadata for the active model of model_name."""
        try:
            meta = registry.get_active_metadata(model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        return ModelInfo(
            model_id=meta.model_id,
            model_name=meta.model_name,
            version=meta.version,
            model_type=meta.model_type,
            status=meta.status,
            training_date=meta.training_date,
            performance_metrics=meta.performance_metrics or {},
        )

    @app.get("/models/{model_name}/versions", response_model=List[ModelInfo])
    def list_versions(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """List all versions of a model."""
        metas = registry.list_models(model_name=model_name)
        if not metas:
            raise HTTPException(status_code=404, detail=f"No models found for: {model_name}")
        return [
            ModelInfo(
                model_id=m.model_id,
                model_name=m.model_name,
                version=m.version,
                model_type=m.model_type,
                status=m.status,
                training_date=m.training_date,
                performance_metrics=m.performance_metrics or {},
            )
            for m in metas
        ]

    @app.post("/models/{model_name}/rollback")
    def rollback_model(
        model_name: str,
        body: RollbackRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Rollback to a previous version of the model."""
        handler = RollbackHandler(registry=registry)
        try:
            handler.rollback_to_version(model_name=model_name, target_version=body.target_version)
            return {"status": "ok", "message": f"Rolled back {model_name} to {body.target_version}"}
        except (ModelNotFoundError, Exception) as e:
            if isinstance(e, ModelNotFoundError):
                raise HTTPException(status_code=404, detail=e.message)
            raise HTTPException(status_code=500, detail="Rollback failed")

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "ml-reliability-platform"}

    @app.get("/metrics/{model_name}")
    def get_metrics(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Return monitoring metrics for the model (placeholder; integrate with monitoring module)."""
        try:
            meta = registry.get_active_metadata(model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        return {
            "model_id": meta.model_id,
            "model_name": meta.model_name,
            "version": meta.version,
            "metrics": meta.performance_metrics or {},
            "message": "Integrate with monitoring.metrics for live metrics",
        }

    return app


app = create_app()
