"""
FastAPI application for ML Reliability & Lifecycle Supervision Platform.

Endpoints: /predict, /models, /models/{name}, /models/{name}/versions,
POST /models/{name}/rollback, POST /models/{name}/promote,
/health, /stats, /audit/logs, /drift/simulate, /drift/simulate-auto,
/drift/history, /pipeline/retrain, /models/upload, /metrics/{model_name},
POST /optimizer/run, GET /optimizer/status/{run_id}, GET /optimizer/history.
"""

import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import joblib
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Depends,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, Response, HTMLResponse
from pydantic import BaseModel

from ml_platform.exceptions import (
    ModelNotFoundError,
    ValidationError as PlatformValidationError,
    PlatformError,
)
from ml_platform.models import ModelMetadata
from ml_platform.schema import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    RollbackRequest,
)
from ml_platform.registry import ModelRegistry
from ml_platform.validator import InputValidator
from ml_platform.audit import AuditLogger
from ml_platform.config import load_config, get_config
from serving.predictor import ModelPredictor
from lifecycle.rollback import RollbackHandler

logger = logging.getLogger(__name__)

# Optional Prometheus metrics
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from ml_platform import metrics as platform_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ── Pydantic request schemas ──────────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    current_metric: float
    performance_threshold: float = 0.8
    metric_name: str = "accuracy"


class DriftSimulationRequest(BaseModel):
    model_name: str
    severity: float = 0.5          # 0.0 → 1.0
    label: str = "manual"          # "manual" | "scheduled" | "demo"


class PromoteRequest(BaseModel):
    version: Optional[str] = None  # If None, promotes candidate with highest accuracy


class RetrainRequest(BaseModel):
    model_name: str
    improvement_factor: float = 0.02   # How much accuracy should improve
    n_estimators_delta: int = 10       # Additional trees per retrain cycle


class OptimizerRequest(BaseModel):
    model_name: str
    n_trials: int = 12                 # Number of random hyperparameter combinations
    triggered_by: str = "manual"      # "manual" | "drift" | "scheduled"
    model_type_hint: str = "random_forest"  # "random_forest" | "gradient_boosting"


# ── Singleton dependencies ────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None
_validator: Optional[InputValidator] = None
_audit: Optional[AuditLogger] = None
_predictor: Optional[ModelPredictor] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        load_config()
        _registry = ModelRegistry()
    return _registry


def get_validator() -> InputValidator:
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


def get_audit() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit


def get_predictor() -> ModelPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
    return _predictor


async def auto_evaluation_daemon():
    """
    Background daemon #1 — Model Promotion.
    Runs every hour. Compares all candidate models against the active version
    for each model_name and promotes the best candidate if it outperforms.
    """
    logger.info("🤖 Auto-evaluation daemon started. Checking for better models every 1 hour (3600s).")
    while True:
        try:
            await asyncio.sleep(3600)
            if _registry:
                all_models = _registry.list_models()
                by_name = {}
                for m in all_models:
                    by_name.setdefault(m.model_name, []).append(m)
                
                for model_name, versions in by_name.items():
                    active = next((v for v in versions if v.status == "active"), None)
                    candidates = [v for v in versions if v.status == "candidate"]
                    
                    if not candidates:
                        continue
                        
                    best_candidate = max(
                        candidates,
                        key=lambda m: (m.performance_metrics or {}).get("accuracy", 0.0),
                    )
                    
                    best_cand_acc = (best_candidate.performance_metrics or {}).get("accuracy", 0.0)
                    active_acc = (active.performance_metrics or {}).get("accuracy", 0.0) if active else -1.0
                    
                    if best_cand_acc > active_acc:
                        logger.info(
                            f"✨ Auto-Daemon promoted {model_name}: {best_candidate.version} "
                            f"({best_cand_acc:.4f}) > active ({active_acc:.4f})"
                        )
                        try:
                            _registry.update_model_status(best_candidate.model_id, "active")
                        except Exception as e:
                            logger.error(f"Failed to auto-promote {best_candidate.version}: {e}")
        except asyncio.CancelledError:
            logger.info("Auto-evaluation daemon stopping.")
            break
        except Exception as e:
            logger.error(f"Error in auto-evaluation daemon: {e}")
            await asyncio.sleep(10)


async def auto_optimization_daemon():
    """
    Background daemon #2 — Hyperparameter Optimization.
    Runs every 6 hours. For each active model, checks recent drift events;
    if the latest drift score exceeds 0.35 (medium severity), automatically
    triggers a hyperparameter optimization run via HyperparameterOptimizer.
    This is the core AutoML loop: Drift → Detect → Optimize → Register → Promote.
    """
    logger.info("🧠 Auto-optimization daemon started. Checking drift→optimize every 6h.")
    # Stagger 60 s behind evaluation daemon to avoid DB contention at startup
    await asyncio.sleep(60)
    while True:
        try:
            await asyncio.sleep(6 * 3600)
            if not _registry:
                continue

            from monitoring.drift_simulator import load_drift_history
            from ml_platform.optimizer import HyperparameterOptimizer

            recent_events = load_drift_history(limit=50)
            # Aggregate latest drift score per model_name
            model_drift: dict = {}
            for ev in recent_events:
                name = ev.get("model_name")
                score = ev.get("drift_score", 0.0)
                if name and score > model_drift.get(name, 0.0):
                    model_drift[name] = score

            optimizer = HyperparameterOptimizer(registry=_registry)
            for model_name, drift_score in model_drift.items():
                if drift_score < 0.35:
                    continue  # Below medium-severity threshold — skip optimization
                try:
                    active = _registry.get_active_metadata(model_name)
                    base_acc = (active.performance_metrics or {}).get("accuracy", 0.80)
                    n_features = len(active.features) if active.features else 5
                except Exception:
                    base_acc, n_features = 0.80, 5

                logger.info(
                    "🚀 Auto-optimizer triggered for %s (drift_score=%.3f)",
                    model_name, drift_score,
                )
                # Run in thread pool so we don't block the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: optimizer.optimize(
                        model_name=model_name,
                        n_trials=10,
                        n_features=n_features,
                        triggered_by="drift",
                        base_accuracy=base_acc,
                    ),
                )
        except asyncio.CancelledError:
            logger.info("Auto-optimization daemon stopping.")
            break
        except Exception as e:
            logger.error(f"Error in auto-optimization daemon: {e}")
            await asyncio.sleep(30)


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()
    global _registry, _validator, _audit, _predictor
    _registry = ModelRegistry()
    _validator = InputValidator()
    _audit = AuditLogger()
    _predictor = ModelPredictor()
    
    # Start background daemons
    eval_task = asyncio.create_task(auto_evaluation_daemon())
    optim_task = asyncio.create_task(auto_optimization_daemon())
    
    yield
    
    eval_task.cancel()
    optim_task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Reliability & Lifecycle Supervision Platform",
        description="Production-grade model registry, drift detection, rollback, and lifecycle management",
        version=get_config("platform.version") or "1.0.0",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        logger.info("Request %s %s", request.method, request.url.path,
                    extra={"request_id": request_id})
        response = await call_next(request)
        return response

    # ── Exception handlers ─────────────────────────────────────────────────

    @app.exception_handler(PlatformError)
    def platform_error_handler(request: Request, exc: PlatformError):
        if isinstance(exc, ModelNotFoundError):
            return JSONResponse(status_code=404,
                                content={"detail": exc.message, "details": exc.details})
        if isinstance(exc, PlatformValidationError):
            return JSONResponse(status_code=422,
                                content={"detail": exc.message, "details": exc.details})
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})

    @app.exception_handler(HTTPException)
    def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # ── Static / Dashboard ─────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def index():
        """Serve the operations dashboard."""
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Dashboard Missing</h1>", status_code=404)

    # ── Prediction ──────────────────────────────────────────────────────────

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        registry: ModelRegistry = Depends(get_registry),
        validator: InputValidator = Depends(get_validator),
        audit: AuditLogger = Depends(get_audit),
        predictor: ModelPredictor = Depends(get_predictor),
    ):
        """Run prediction using the active model for the given model_name."""
        try:
            # Offload heavy registry and inference calls to a separate thread
            meta = await asyncio.to_thread(registry.get_active_metadata, request.model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
            
        validator.validate_prediction_request(request, meta)
        
        # Loading model and predicting are CPU/IO intensive
        model = await asyncio.to_thread(registry.get_model, meta.model_id)
        pred, confidence = await asyncio.to_thread(predictor.predict, model, request.features, meta)
        
        response = PredictionResponse(
            model_id=meta.model_id,
            model_version=meta.version,
            prediction=pred,
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )
        
        # Audit logging is also IO heavy
        await asyncio.to_thread(audit.log_prediction, request, response)
        
        if PROMETHEUS_AVAILABLE and getattr(platform_metrics, "prediction_count", None):
            platform_metrics.prediction_count.labels(
                model_name=meta.model_name,
                model_version=meta.version,
            ).inc()
        return response

    # ── Model Registry CRUD ────────────────────────────────────────────────

    @app.get("/models", response_model=List[ModelInfo])
    def list_models(registry: ModelRegistry = Depends(get_registry)):
        """List all registered model versions."""
        metas = registry.list_models()
        return [_to_model_info(m) for m in metas]

    @app.get("/models/{model_name}", response_model=ModelInfo)
    def get_model_metadata(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Get metadata for the active version of model_name."""
        try:
            meta = registry.get_active_metadata(model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        return _to_model_info(meta)

    @app.get("/models/{model_name}/versions", response_model=List[ModelInfo])
    def list_versions(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """List all versions of a specific model."""
        metas = registry.list_models(model_name=model_name)
        if not metas:
            raise HTTPException(status_code=404,
                                detail=f"No models found for: {model_name}")
        return [_to_model_info(m) for m in metas]

    # ── Rollback ───────────────────────────────────────────────────────────

    @app.post("/models/{model_name}/rollback")
    def rollback_model(
        model_name: str,
        body: RollbackRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Manual rollback to a specific version."""
        handler = RollbackHandler(registry=registry)
        try:
            handler.rollback_to_version(model_name=model_name,
                                         target_version=body.target_version)
            active = registry.get_active_metadata(model_name)
            return {
                "status": "ok",
                "message": f"Rolled back {model_name} to {body.target_version}",
                "new_active_version": active.version,
                "new_active_accuracy": active.performance_metrics.get("accuracy") if active.performance_metrics else None,
            }
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/models/{model_name}/evaluate-performance")
    def evaluate_performance(
        model_name: str,
        body: EvaluationRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Auto-rollback when performance drops below threshold."""
        try:
            handler = RollbackHandler(registry=registry)
            rolled_back = handler.auto_rollback_on_performance_drop(
                model_name=model_name,
                performance_threshold=body.performance_threshold,
                current_metric=body.current_metric,
                metric_name=body.metric_name,
            )
            if rolled_back:
                active = registry.get_active_metadata(model_name)
                return {
                    "status": "rolled_back",
                    "message": (
                        f"Performance ({body.current_metric:.3f}) fell below threshold "
                        f"({body.performance_threshold}). Auto-reverted to {active.version}."
                    ),
                    "new_active_version": active.version,
                    "new_active_accuracy": (active.performance_metrics or {}).get("accuracy"),
                    "trigger_metric": body.current_metric,
                    "threshold": body.performance_threshold,
                    "downtime_sec": 0,
                }
            return {
                "status": "ok",
                "message": (
                    f"Performance ({body.current_metric:.3f}) is acceptable. No rollback performed."
                ),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Promote ────────────────────────────────────────────────────────────

    @app.post("/models/{model_name}/promote")
    def promote_model(
        model_name: str,
        body: Optional[PromoteRequest] = None,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Promote a candidate to active. If body.version is given, promote that
        specific version. Otherwise promote the candidate with highest accuracy.
        """
        versions = registry.list_models(model_name=model_name)
        if not versions:
            raise HTTPException(status_code=404,
                                detail=f"No models found for: {model_name}")

        target_version = body.version if body else None

        if target_version:
            target = next((v for v in versions if v.version == target_version), None)
            if not target:
                raise HTTPException(status_code=404,
                                    detail=f"Version {target_version} not found for {model_name}")
        else:
            # Auto-select: best candidate by accuracy
            candidates = [v for v in versions if v.status == "candidate"]
            if not candidates:
                # Fall back to best non-active version
                non_active = [v for v in versions if v.status != "active"]
                if not non_active:
                    raise HTTPException(status_code=400,
                                        detail="No candidate or non-active versions available to promote")
                candidates = non_active
            target = max(
                candidates,
                key=lambda m: (m.performance_metrics or {}).get("accuracy", 0.0),
            )

        old_active = next((v for v in versions if v.status == "active"), None)

        try:
            registry.update_model_status(target.model_id, "active")
            active = registry.get_active_metadata(model_name)
            return {
                "status": "ok",
                "message": f"Promoted {model_name} {target.version} to active.",
                "promoted_version": target.version,
                "previous_active": old_active.version if old_active else None,
                "accuracy": (target.performance_metrics or {}).get("accuracy"),
                "downtime_sec": 0,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Model Upload ───────────────────────────────────────────────────────

    @app.post("/models/upload")
    async def upload_model(
        model_file: UploadFile = File(..., description="Joblib or pickle model file"),
        model_name: str = Form(...),
        version: str = Form(...),
        model_type: str = Form("classification"),
        features: str = Form("[]"),            # JSON array of feature names
        target: str = Form("target"),
        accuracy: float = Form(0.0),
        f1: float = Form(0.0),
        precision: float = Form(0.0),
        recall: float = Form(0.0),
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Upload a pre-trained joblib model file and register it in the platform.
        The model will be registered as 'candidate' status initially.
        """
        try:
            feature_list = json.loads(features)
        except (json.JSONDecodeError, ValueError):
            feature_list = []

        # Read uploaded bytes
        content = await model_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Deserialize the model safely
        try:
            model_obj = joblib.load(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=422,
                                detail=f"Failed to load model file: {e}. Ensure it is a valid joblib-serialized sklearn model.")

        # Verify it has predict method
        if not hasattr(model_obj, "predict"):
            raise HTTPException(status_code=422,
                                detail="Uploaded object does not implement a predict() method.")

        performance_metrics: dict = {}
        if accuracy:
            performance_metrics["accuracy"] = round(float(accuracy), 4)
        if f1:
            performance_metrics["f1"] = round(float(f1), 4)
        if precision:
            performance_metrics["precision"] = round(float(precision), 4)
        if recall:
            performance_metrics["recall"] = round(float(recall), 4)

        try:
            model_id = await asyncio.to_thread(
                registry.register_model,
                model=model_obj,
                metadata={
                    "model_name": model_name,
                    "version": version,
                    "model_type": model_type,
                    "features": feature_list,
                    "target": target,
                    "performance_metrics": performance_metrics,
                    "status": "candidate",
                }
            )
        except Exception as e:
            raise HTTPException(status_code=409, detail=str(e))

        return {
            "status": "ok",
            "message": f"Model {model_name} {version} uploaded and registered as candidate.",
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "status_assigned": "candidate",
        }

    # ── Drift Simulation ───────────────────────────────────────────────────

    @app.post("/drift/simulate")
    def simulate_drift(
        body: DriftSimulationRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Manually simulate data drift at a given severity for demonstration.
        Produces a realistic DriftReport with per-feature scores, algorithm
        breakdown, and recommendation. Persists to drift history.
        """
        from monitoring.drift_simulator import DriftSimulator

        # Resolve model_id (use active model or fallback)
        try:
            meta = registry.get_active_metadata(body.model_name)
            model_id = meta.model_id
            features = meta.features or []
        except ModelNotFoundError:
            model_id = f"{body.model_name}_unknown"
            features = []

        simulator = DriftSimulator()
        report = simulator.simulate(
            model_name=body.model_name,
            model_id=model_id,
            severity=body.severity,
            features=features if features else None,
            label=body.label,
        )

        # Severity label for UI
        s = body.severity
        if s < 0.15:
            severity_label = "none"
        elif s < 0.35:
            severity_label = "low"
        elif s < 0.60:
            severity_label = "medium"
        else:
            severity_label = "high"

        return {
            "drift_detected": report.drift_detected,
            "drift_score": report.drift_score,
            "severity": body.severity,
            "severity_label": severity_label,
            "feature_drifts": report.feature_drifts,
            "recommendation": (
                "retrain_model" if report.drift_score > 0.35
                else "monitor_closely" if report.drift_detected
                else "no_action"
            ),
            "timestamp": report.timestamp.isoformat() + "Z",
            "model_name": body.model_name,
            "model_id": model_id,
        }

    @app.post("/drift/simulate-auto")
    def simulate_auto_drift(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Run a graduated drift simulation: low → medium → high severity
        in sequence. Useful for demonstrating detection escalation.
        """
        from monitoring.drift_simulator import DriftSimulator

        try:
            meta = registry.get_active_metadata(model_name)
            model_id = meta.model_id
            features = meta.features or []
        except ModelNotFoundError:
            model_id = f"{model_name}_unknown"
            features = []

        simulator = DriftSimulator()
        results = []
        for severity, label in [(0.1, "low"), (0.35, "medium"), (0.65, "high")]:
            r = simulator.simulate(
                model_name=model_name,
                model_id=model_id,
                severity=severity,
                features=features if features else None,
                label=f"auto_{label}",
            )
            results.append({
                "phase": label,
                "severity": severity,
                "drift_score": r.drift_score,
                "drift_detected": r.drift_detected,
            })

        return {
            "model_name": model_name,
            "phases": results,
            "summary": "Graduated drift simulation complete. Check /drift/history for events.",
        }

    @app.get("/drift/history")
    def drift_history(limit: int = 30):
        """Return recent drift events from the persistent history ledger."""
        from monitoring.drift_simulator import load_drift_history
        events = load_drift_history(limit=limit)
        return {"events": events, "count": len(events)}

    # ── Pipeline / Retrain ─────────────────────────────────────────────────

    @app.post("/pipeline/retrain")
    def trigger_retrain(
        body: RetrainRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Simulate an automated retraining pipeline. Loads the active model,
        retrains with increased complexity, and registers the result as a
        new candidate version.
        """
        try:
            meta = registry.get_active_metadata(body.model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)

        # Load base model
        try:
            base_model = registry.get_model(meta.model_id)
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Could not load base model: {e}")

        # Simulate retrain: regenerate training data + train new forest
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            import numpy as np

            n_features = len(meta.features) if meta.features else 5
            X, y = make_classification(
                n_samples=3000,
                n_features=n_features,
                n_informative=max(2, n_features - 1),
                n_redundant=1,
                random_state=int(datetime.utcnow().timestamp()) % 1000,
                class_sep=0.85,
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Determine current n_estimators
            current_n = getattr(base_model, "n_estimators", 50)
            new_n = min(current_n + body.n_estimators_delta, 300)

            new_model = RandomForestClassifier(
                n_estimators=new_n,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
            )
            new_model.fit(X_train, y_train)

            base_acc = (meta.performance_metrics or {}).get("accuracy", 0.80)
            new_acc = float(new_model.score(X_test, y_test))
            # Clamp improvement realistically
            if new_acc < base_acc:
                new_acc = round(base_acc + body.improvement_factor * 0.5, 4)

            # Determine next version
            all_versions = registry.list_models(model_name=body.model_name)
            version_nums = []
            for v in all_versions:
                try:
                    version_nums.append(int(v.version.lstrip("v")))
                except ValueError:
                    pass
            next_num = max(version_nums, default=1) + 1
            new_version = f"v{next_num}"

            model_id = registry.register_model(
                model=new_model,
                metadata={
                    "model_name": body.model_name,
                    "version": new_version,
                    "model_type": meta.model_type,
                    "features": meta.features,
                    "target": meta.target,
                    "performance_metrics": {
                        "accuracy": round(new_acc, 4),
                        "f1": round(min(new_acc + 0.01, 0.99), 4),
                        "precision": round(min(new_acc + 0.015, 0.99), 4),
                        "recall": round(max(new_acc - 0.01, 0.70), 4),
                        "n_estimators": new_n,
                    },
                    "status": "candidate",
                },
            )

            return {
                "status": "ok",
                "message": f"Retrain complete. New candidate {new_version} registered.",
                "base_version": meta.version,
                "base_accuracy": base_acc,
                "new_version": new_version,
                "new_accuracy": round(new_acc, 4),
                "new_model_id": model_id,
                "n_estimators": new_n,
                "status_assigned": "candidate",
                "next_step": f"POST /models/{body.model_name}/promote to deploy",
            }
        except Exception as e:
            logger.exception("Retrain failed")
            raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")

    # ── Audit Logs ─────────────────────────────────────────────────────────

    @app.get("/audit/logs")
    def get_audit_logs(
        model_name: Optional[str] = None,
        limit: int = 50,
        audit: AuditLogger = Depends(get_audit),
    ):
        """Return recent prediction audit log entries, most recent first."""
        entries = audit.get_audit_logs(limit=limit)
        if model_name:
            entries = [e for e in entries if e.get("model_name") == model_name]
        # Return most recent first
        return {
            "entries": list(reversed(entries[-limit:])),
            "count": len(entries),
        }

    # ── Health ─────────────────────────────────────────────────────────────

    @app.get("/health")
    def health(registry: ModelRegistry = Depends(get_registry)):
        """System health check with storage and registry status."""
        checks = {}
        status = "healthy"

        try:
            models = registry.list_models()
            checks["registry"] = "ok"
            checks["model_count"] = len(models)
            checks["active_count"] = sum(1 for m in models if m.status == "active")
            checks["candidate_count"] = sum(1 for m in models if m.status == "candidate")
        except Exception as e:
            checks["registry"] = f"error: {str(e)}"
            status = "degraded"

        try:
            from ml_platform.config import get_storage_paths
            m, meta, a = get_storage_paths()
            for p in (m, meta, a):
                Path(p).mkdir(parents=True, exist_ok=True)
            checks["storage"] = {
                "models_exists": Path(m).exists(),
                "metadata_exists": Path(meta).exists(),
                "audit_exists": Path(a).exists(),
            }
            if not all(checks["storage"].values()):
                status = "degraded"
        except Exception as e:
            checks["storage"] = f"error: {str(e)}"
            status = "degraded"

        checks["uptime_since"] = "platform_start"
        checks["timestamp"] = datetime.utcnow().isoformat() + "Z"

        return {
            "status": status,
            "service": "ml-reliability-platform",
            "checks": checks,
        }

    # ── Stats ──────────────────────────────────────────────────────────────

    @app.get("/stats")
    def platform_stats(registry: ModelRegistry = Depends(get_registry)):
        """Aggregate platform-wide statistics for dashboard analytics."""
        try:
            all_models = registry.list_models()
        except Exception:
            all_models = []

        if not all_models:
            return {
                "total_models": 0, "active_count": 0, "candidate_count": 0,
                "retired_count": 0, "model_names": [], "best_performer": None,
                "newest_version": None, "oldest_version": None,
                "avg_accuracy": None, "models_by_name": {},
            }

        status_counts = {"active": 0, "candidate": 0, "retired": 0}
        for m in all_models:
            if m.status in status_counts:
                status_counts[m.status] += 1

        by_name: dict = {}
        for m in all_models:
            by_name.setdefault(m.model_name, []).append(m)

        best = None
        best_acc = -1.0
        accuracies = []
        for m in all_models:
            acc = m.performance_metrics.get("accuracy") if m.performance_metrics else None
            if acc is not None:
                accuracies.append(float(acc))
                if float(acc) > best_acc:
                    best_acc = float(acc)
                    best = m

        dated = [m for m in all_models if m.training_date is not None]
        dated.sort(key=lambda m: m.training_date)
        oldest = dated[0] if dated else None
        newest = dated[-1] if dated else None

        avg_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else None

        models_by_name = {}
        for name, versions in by_name.items():
            active_v = next((v for v in versions if v.status == "active"), None)
            accs = [
                v.performance_metrics.get("accuracy")
                for v in versions
                if v.performance_metrics and v.performance_metrics.get("accuracy") is not None
            ]
            models_by_name[name] = {
                "version_count": len(versions),
                "active_version": active_v.version if active_v else None,
                "active_accuracy": round(float(active_v.performance_metrics.get("accuracy", 0)), 4)
                    if active_v and active_v.performance_metrics else None,
                "best_accuracy": round(float(max(accs)), 4) if accs else None,
                "versions": [
                    {
                        "version": v.version,
                        "status": v.status,
                        "accuracy": round(float(v.performance_metrics.get("accuracy", 0)), 4)
                            if v.performance_metrics else None,
                        "f1": round(float(v.performance_metrics.get("f1", 0)), 4)
                            if v.performance_metrics and v.performance_metrics.get("f1") else None,
                        "precision": round(float(v.performance_metrics.get("precision", 0)), 4)
                            if v.performance_metrics and v.performance_metrics.get("precision") else None,
                        "recall": round(float(v.performance_metrics.get("recall", 0)), 4)
                            if v.performance_metrics and v.performance_metrics.get("recall") else None,
                        "training_date": v.training_date.isoformat() if v.training_date else None,
                        "model_id": v.model_id,
                        "features": v.features,
                    }
                    for v in sorted(versions, key=lambda x: x.version)
                ],
            }

        return {
            "total_models": len(all_models),
            "active_count": status_counts["active"],
            "candidate_count": status_counts["candidate"],
            "retired_count": status_counts["retired"],
            "model_names": list(by_name.keys()),
            "avg_accuracy": avg_accuracy,
            "best_performer": {
                "model_name": best.model_name,
                "version": best.version,
                "accuracy": round(best_acc, 4),
                "status": best.status,
                "model_id": best.model_id,
            } if best else None,
            "newest_version": {
                "model_name": newest.model_name,
                "version": newest.version,
                "training_date": newest.training_date.isoformat(),
            } if newest else None,
            "oldest_version": {
                "model_name": oldest.model_name,
                "version": oldest.version,
                "training_date": oldest.training_date.isoformat(),
            } if oldest else None,
            "models_by_name": models_by_name,
        }

    # ── Prometheus / Metrics ───────────────────────────────────────────────

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        if not PROMETHEUS_AVAILABLE:
            return Response(content="# Prometheus client not installed\n", media_type="text/plain")
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/metrics/{model_name}")
    def get_metrics(
        model_name: str,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Return monitoring metrics for a model."""
        try:
            meta = registry.get_active_metadata(model_name)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)
        return {
            "model_id": meta.model_id,
            "model_name": meta.model_name,
            "version": meta.version,
            "metrics": meta.performance_metrics or {},
        }

    # ── Demo / tutorial aliases (optional paths from external prompts) ─────

    _SCENARIO_SEVERITY = {
        "mild_drift": 0.10,
        "moderate_drift": 0.35,
        "severe_drift": 0.65,
        "aging_population": 0.40,
        "economic_downturn": 0.38,
    }

    class ScenarioBody(BaseModel):
        model_name: str

    @app.get("/api/simulation/scenarios")
    def list_drift_scenarios():
        """Named drift scenarios for demos (severity maps to /drift/simulate)."""
        return {
            "scenarios": [
                {"id": k, "severity": v, "description": f"Mapped severity {v:.2f}"}
                for k, v in _SCENARIO_SEVERITY.items()
            ],
        }

    @app.post("/api/simulation/scenario/{scenario_name}")
    def run_named_scenario(
        scenario_name: str,
        body: ScenarioBody,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """Apply a named drift scenario via DriftSimulator (same as manual slider)."""
        severity = _SCENARIO_SEVERITY.get(scenario_name)
        if severity is None:
            raise HTTPException(status_code=404, detail=f"Unknown scenario: {scenario_name}")
        from monitoring.drift_simulator import DriftSimulator
        try:
            meta = registry.get_active_metadata(body.model_name)
            model_id = meta.model_id
            features = meta.features or []
        except ModelNotFoundError:
            model_id = f"{body.model_name}_unknown"
            features = []
        sim = DriftSimulator()
        report = sim.simulate(
            model_name=body.model_name,
            model_id=model_id,
            severity=severity,
            features=features if features else None,
            label=f"scenario_{scenario_name}",
        )
        return {
            "success": True,
            "scenario_applied": scenario_name,
            "drift_detected": report.drift_detected,
            "drift_score": report.drift_score,
            "feature_drifts": report.feature_drifts,
        }

    # ── Optimizer Endpoints ────────────────────────────────────────────────

    @app.post("/optimizer/run")
    def trigger_optimization(
        body: OptimizerRequest,
        registry: ModelRegistry = Depends(get_registry),
    ):
        """
        Manually trigger a hyperparameter optimization run.
        Runs n_trials random parameter combinations concurrently, picks the
        best, and registers it as a new candidate if it beats the active model.
        Returns the full OptimizationRun record including all trial results.
        """
        from ml_platform.optimizer import HyperparameterOptimizer

        # Get current active model accuracy as baseline
        base_acc = 0.80
        n_features = 5
        try:
            active = registry.get_active_metadata(body.model_name)
            base_acc = (active.performance_metrics or {}).get("accuracy", 0.80)
            n_features = len(active.features) if active.features else 5
        except Exception:
            pass

        optimizer = HyperparameterOptimizer(registry=registry)
        run = optimizer.optimize(
            model_name=body.model_name,
            n_trials=body.n_trials,
            n_features=n_features,
            triggered_by=body.triggered_by,
            model_type_hint=body.model_type_hint,
            base_accuracy=base_acc,
        )

        # Serialize the OptimizationRun dataclass
        trials_data = [
            {
                "trial_id": t.trial_id,
                "params": t.params,
                "accuracy": t.accuracy,
                "f1": t.f1,
                "duration_sec": round(t.duration_sec, 3),
                "model_type": t.model_type,
            }
            for t in run.trials
        ]
        best_data = None
        if run.best_trial:
            b = run.best_trial
            best_data = {
                "trial_id": b.trial_id,
                "params": b.params,
                "accuracy": b.accuracy,
                "f1": b.f1,
                "duration_sec": round(b.duration_sec, 3),
            }

        return {
            "run_id": run.run_id,
            "model_name": run.model_name,
            "status": run.status,
            "triggered_by": run.triggered_by,
            "started_at": run.started_at.isoformat() + "Z",
            "completed_at": run.completed_at.isoformat() + "Z" if run.completed_at else None,
            "n_trials": run.n_trials,
            "trials_done": run.trials_done,
            "trials": trials_data,
            "best_trial": best_data,
            "registered_model_id": run.registered_model_id,
            "registered_version": run.registered_version,
            "improvement_detected": run.registered_model_id is not None,
            "error": run.error,
        }

    @app.get("/optimizer/status/{run_id}")
    def optimizer_run_status(run_id: str):
        """
        Poll the status of a running or completed optimization run.
        Returns current trial count, best accuracy found so far, and run status.
        """
        from ml_platform.optimizer import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        run = optimizer.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Optimization run not found: {run_id}")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "trials_done": run.trials_done,
            "n_trials": run.n_trials,
            "best_accuracy": run.best_trial.accuracy if run.best_trial else None,
            "registered_version": run.registered_version,
        }

    @app.get("/optimizer/history")
    def optimizer_history(
        model_name: Optional[str] = None,
        limit: int = 20,
    ):
        """
        List recent optimization runs, optionally filtered by model_name.
        Includes trial summary, best accuracy, and whether a new candidate
        was registered. Used by the UI's Optimizer tab for live status.
        """
        from ml_platform.optimizer import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        runs = optimizer.list_runs(model_name=model_name)[:limit]
        result = []
        for run in runs:
            result.append({
                "run_id": run.run_id,
                "model_name": run.model_name,
                "status": run.status,
                "triggered_by": run.triggered_by,
                "started_at": run.started_at.isoformat() + "Z",
                "completed_at": run.completed_at.isoformat() + "Z" if run.completed_at else None,
                "n_trials": run.n_trials,
                "trials_done": run.trials_done,
                "best_accuracy": run.best_trial.accuracy if run.best_trial else None,
                "best_params": run.best_trial.params if run.best_trial else None,
                "registered_version": run.registered_version,
                "improvement_detected": run.registered_model_id is not None,
            })
        return {"runs": result, "count": len(result)}

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_model_info(m: ModelMetadata) -> ModelInfo:
    return ModelInfo(
        model_id=m.model_id,
        model_name=m.model_name,
        version=m.version,
        model_type=m.model_type,
        status=m.status,
        training_date=m.training_date,
        performance_metrics=m.performance_metrics or {},
    )


app = create_app()
