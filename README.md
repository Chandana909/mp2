# ML Reliability Platform

Industry-grade, model-agnostic ML reliability and lifecycle supervision platform. Provides model registry, serving, monitoring, versioning, and lifecycle management—similar to MLflow, Seldon, or internal ML infrastructure at tech companies.

## Features

- **Model Registry & Versioning**: Register models with metadata (name, version, training date, features, metrics). Store multiple versions; track active / candidate / retired; rollback to previous versions.
- **Prediction Serving**: FastAPI REST API with `/predict`, schema validation, routing to active model, confidence scores for classification.
- **Audit Logging**: Log every prediction request/response; queryable by model and date.
- **Monitoring & Drift Detection**: Track feature and prediction distributions; detect data drift (KS, chi-square, Jensen-Shannon, PSI); configurable thresholds.
- **Lifecycle Management**: Retrain on drift; evaluate candidate vs active; promote candidate if better; rollback on performance drop.

## Project Structure

```
ml-platform/
├── platform/          # Registry, schema, validation, audit, config
├── monitoring/        # Drift detection, metrics, alerts
├── lifecycle/         # Trainer, evaluator, promoter, rollback
├── serving/           # FastAPI app, predictor
├── storage/           # Model store, metadata store, audit store
├── config/            # platform_config.yaml, model_configs/
├── scripts/           # register_model, check_drift, promote_model
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Quick Start

### 1. Install

```bash
cd ml-platform   # or your project root (mp2)
pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

### 3. Register a Model

From Python:

```python
from platform.registry import ModelRegistry
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

Or via CLI:

```bash
python scripts/register_model.py --model-path model.joblib --name fraud_detector --version v1 --type classification --features amount,merchant_id,time_of_day --target is_fraud --activate
```

### 4. Get Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "fraud_detector", "features": {"amount": 250, "merchant_id": "merch_123", "time_of_day": 14}}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Run prediction (body: `model_name`, `features`) |
| GET | `/models` | List all registered models |
| GET | `/models/{model_name}` | Metadata for active model |
| GET | `/models/{model_name}/versions` | All versions of a model |
| POST | `/models/{model_name}/rollback` | Rollback to `target_version` (body: `{"target_version": "v1"}`) |
| GET | `/health` | Health check |
| GET | `/metrics/{model_name}` | Monitoring metrics (placeholder) |

## Configuration

- **Platform**: `config/platform_config.yaml` — storage paths, drift thresholds, promotion criteria, serving host/port.
- **Per-model**: `config/model_configs/<model_name>.yaml` — features, target, algorithm, hyperparameters, validation.

## Drift Detection

```bash
python scripts/check_drift.py --reference training_sample.csv --current recent_requests.csv --threshold 0.15
```

Or in code:

```python
from monitoring.drift_detector import DriftDetector
import pandas as pd

detector = DriftDetector(drift_threshold=0.15)
report = detector.compute_data_drift(reference_data, current_data, model_id="my_model")
if report.drift_detected:
    # Trigger retrain or alert
    pass
```

## Lifecycle Workflow

1. **Retrain** (e.g. on drift or schedule): `ModelTrainer.retrain_active_model(model_name, data)` → new candidate.
2. **Evaluate**: `ModelEvaluator.compare_models(candidate_id, active_id, test_data, target_column)` → comparison report.
3. **Promote**: `ModelPromoter.promote_to_active(candidate_id, {"candidate_is_better": True})` if criteria met.
4. **Rollback**: `RollbackHandler.rollback_to_version(model_name, target_version)` or use `POST /models/{name}/rollback`.

## Testing

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## Development Guidelines

- **Type hints** and **PEP 8** throughout.
- **Custom exceptions**: `ModelNotFoundError`, `ValidationError`, `StorageError`, `PromotionError`, `RollbackError`.
- **Logging**: Python `logging`; include `model_id`, `request_id` where applicable.
- **No business logic**: Platform is model-agnostic; no hardcoded fraud/churn logic—everything driven by config and metadata.

## License

Use as needed for portfolio or internal use.
