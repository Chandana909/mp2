# ML Reliability Platform

Industry-grade, model-agnostic ML reliability and lifecycle supervision platform. Provides model registry, serving, monitoring, versioning, and lifecycle management—similar to MLflow, Seldon, or internal ML infrastructure at tech companies.

## Features

- **Model Registry & Versioning**: Register models with metadata (name, version, training date, features, metrics). Store multiple versions; track active / candidate / retired; rollback to previous versions.
- **Prediction Serving**: FastAPI REST API with `/predict`, schema validation, routing to active model, confidence scores for classification and regression.
- **Audit Logging**: Log every prediction request/response; queryable by model and date.
- **Monitoring & Drift Detection**: Track feature and prediction distributions; detect data drift (KS, chi-square, Jensen-Shannon, PSI); configurable thresholds.
- **Lifecycle Management**: Retrain on drift; evaluate candidate vs active; promote candidate if better; rollback on performance drop.
- **Production Ready**: Enhanced health checks, Prometheus metrics, structured logging, verification and demo scripts.

## Recent Features

### Interactive Dashboard
- Glassmorphism-styled UI at `GET /`
- Live health polling with automatic updates
- "Simulate Performance Drop" for testing auto-rollback
- Real-time model status display

### Auto-Rollback System
- Endpoint: `POST /models/{name}/evaluate-performance`
- Autonomous failover on performance degradation
- Zero-downtime hot-swap capability
- Configurable performance thresholds

### Enhanced Monitoring
- Component-level health telemetry
- Real-time drift detection (KS, Chi-Square, Jensen-Shannon, PSI)
- Performance tracking and alerts

### Demo Scripts
- `demo_workflow.py` - Generic ML lifecycle
- `demo_churn.py` - Customer churn 5-version progression
- `verify_setup.py` - Pre-flight diagnostics

## Quick Start

### 1. Verify setup

```bash
cd mp2   # or your project root
pip install -r requirements.txt
python scripts/verify_setup.py
```

### 2. Start the API

```bash
python run.py
```

Options: `--host 0.0.0.0`, `--port 8080`, `--reload` (development), `--workers 4`.

Or with Uvicorn directly:

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

### 3. Register a model

From Python:

```python
from ml_platform.registry import ModelRegistry
import joblib

model = joblib.load("path/to/model.joblib")
registry = ModelRegistry()
model_id = registry.register_model(model, metadata={
    "model_name": "fraud_detector",
    "version": "v1",
    "model_type": "classification",
    "features": ["amount", "merchant_id", "time_of_day"],
    "target": "is_fraud",
    "performance_metrics": {"accuracy": 0.94, "f1": 0.91},
})
registry.update_model_status(model_id, "active")
```

Or run the full demo workflow:

```bash
python scripts/demo_workflow.py
```

### 4. Get predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "fraud_detector", "features": {"amount": 250, "merchant_id": "merch_123", "time_of_day": 14}}'
```

## Architecture Overview

- **ml_platform**: Core registry, configuration, Pydantic schemas, validation, audit logger. No storage implementation; delegates to `storage/`.
- **storage**: Model artifacts (joblib), metadata (JSON), audit logs (JSONL). Paths from `config/platform_config.yaml`.
- **serving**: FastAPI app, dependency injection for registry/validator/audit/predictor. Middleware for request logging; custom exception handlers.
- **monitoring**: Drift detection (KS, chi-square, JS, PSI), metrics helpers, alert hooks.
- **lifecycle**: Trainer (config-driven), evaluator, promoter, rollback handler. Integrates with registry and config.

All components are model-agnostic; behavior is driven by YAML config and model metadata.

## API Documentation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Run prediction (body: `model_name`, `features`) |
| GET | `/models` | List all registered models |
| GET | `/models/{model_name}` | Metadata for active model |
| GET | `/models/{model_name}/versions` | All versions of a model |
| POST | `/models/{model_name}/rollback` | Rollback to `target_version` (body: `{"target_version": "v1"}`) |
| GET | `/health` | Health check with registry and storage status |
| GET | `/metrics` | Prometheus metrics (for scraping) |
| GET | `/metrics/{model_name}` | Model-level metrics (placeholder) |

**Health response** includes `status` ("healthy" or "degraded"), `checks.registry`, `checks.model_count`, and `checks.storage` (paths existence).

Interactive API docs: **http://localhost:8000/docs**

## Development Setup

1. Clone or open the project; create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or: source .venv/bin/activate  # Linux/macOS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run verification:
   ```bash
   python scripts/verify_setup.py
   ```
4. Run tests:
   ```bash
   pytest tests/ -v
   ```
5. Start API in development (auto-reload):
   ```bash
   python run.py --reload
   ```

## Testing

```bash
pytest tests/ -v
```

Requires: `pytest`, `pytest-asyncio`, `httpx` (in `requirements.txt`). Tests cover registry, serving API (predict, list, rollback, errors), and monitoring (drift, metrics).

## Deployment

- **Docker**: `docker-compose up --build`. Image uses `Dockerfile` (Python 3.11, `ml_platform/` and other packages copied). Volumes for `models/`, `storage/`, `data/`.
- **Production**: Use `python run.py --host 0.0.0.0 --port 8000 --workers 4` or run Uvicorn with multiple workers. Set `serving.host`, `serving.port`, `serving.workers` in `config/platform_config.yaml` as needed.
- **Logging**: `run.py` writes to stdout and `platform.log` in the project root. Adjust `setup_logging()` for your environment.

## Configuration

- **Platform**: `config/platform_config.yaml` — storage paths, drift thresholds, promotion criteria, serving host/port, log level.
- **Per-model**: `config/model_configs/<model_name>.yaml` — features, target, algorithm, hyperparameters, validation, monitoring metrics.

Storage paths are relative to project root; leading `./` is optional.

## Monitoring & Drift Detection

- **CLI**: `python scripts/check_drift.py --reference training_sample.csv --current recent_requests.csv --threshold 0.15`
- **Code**:
  ```python
  from monitoring.drift_detector import DriftDetector
  import pandas as pd

  detector = DriftDetector(drift_threshold=0.15)
  report = detector.compute_data_drift(reference_data, current_data, model_id="my_model")
  if report.drift_detected:
      # Trigger retrain or alert
      pass
  ```
- **Prometheus**: GET `/metrics` exposes counters/histograms (e.g. `platform_predictions_total`, `platform_request_duration_seconds`). Scrape this endpoint for Grafana or other monitoring.

## Troubleshooting

- **Import errors**: Run from project root and ensure `PYTHONPATH` includes the project root (e.g. `export PYTHONPATH=.` or use `python run.py` / `python scripts/verify_setup.py` which set path).
- **Config not found**: Ensure `config/platform_config.yaml` exists; check path in `ml_platform.config`. On failure, defaults are used and a warning is logged.
- **Model not found / 404**: Register the model and set status to `active` via `registry.update_model_status(model_id, "active")`.
- **Validation errors on /predict**: Ensure request body has `model_name` and `features` with all keys matching the model’s registered `features` list.
- **Storage errors**: Ensure `models/`, `storage/metadata/`, `storage/audit_logs/` exist or are writable; `run.py` creates them on startup.

## Contributing

- Follow PEP 8 and use type hints.
- Add tests for new behavior; run `pytest tests/ -v`.
- Use custom exceptions (`ModelNotFoundError`, `ValidationError`, etc.); log with context (`model_id`, `request_id`).
- Keep the platform model-agnostic; no hardcoded business logic—drive behavior via config and metadata.

## Lifecycle Workflow

1. **Retrain** (e.g. on drift or schedule): `ModelTrainer.retrain_active_model(model_name, data)` → new candidate.
2. **Evaluate**: `ModelEvaluator.compare_models(candidate_id, active_id, test_data, target_column)` → comparison report.
3. **Promote**: `ModelPromoter.promote_to_active(candidate_id, {"candidate_is_better": True})` if criteria met.
4. **Rollback**: `RollbackHandler.rollback_to_version(model_name, target_version)` or `POST /models/{name}/rollback`.

## License

Use as needed for portfolio or internal use.
