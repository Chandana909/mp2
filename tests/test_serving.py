"""Tests for serving layer: predictor and API."""

import pytest
from fastapi.testclient import TestClient

from serving.api import create_app, get_registry, get_validator, get_audit, get_predictor
from ml_platform.registry import ModelRegistry
from ml_platform.models import ModelMetadata


@pytest.fixture
def client(registry):
    """Test client with app that uses provided registry."""
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


@pytest.fixture
def registered_active_model(registry, sample_model):
    meta = {
        "model_name": "test_api_model",
        "version": "v1",
        "model_type": "classification",
        "features": ["f1", "f2", "f3"],
        "target": "y",
        "status": "candidate",
    }
    model_id = registry.register_model(sample_model, meta)
    registry.update_model_status(model_id, "active")
    return model_id


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"


def test_predict_success(client, registered_active_model):
    r = client.post(
        "/predict",
        json={
            "model_name": "test_api_model",
            "features": {"f1": 0.5, "f2": 0.5, "f3": 0.5},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "model_version" in data
    assert data["model_version"] == "v1"


def test_predict_missing_model(client):
    r = client.post(
        "/predict",
        json={
            "model_name": "nonexistent",
            "features": {"f1": 0.5},
        },
    )
    assert r.status_code == 404


def test_predict_validation_error(client, registered_active_model):
    r = client.post(
        "/predict",
        json={
            "model_name": "test_api_model",
            "features": {"f1": 0.5},  # missing f2, f3
        },
    )
    assert r.status_code == 422


def test_list_models(client, registered_active_model):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    names = [m["model_name"] for m in data]
    assert "test_api_model" in names


def test_get_model_metadata(client, registered_active_model):
    r = client.get("/models/test_api_model")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "test_api_model"
    assert data["version"] == "v1"
    assert data["status"] == "active"


def test_list_versions(client, registered_active_model):
    r = client.get("/models/test_api_model/versions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert any(v["version"] == "v1" for v in data)


def test_rollback(client, registry, sample_model, registered_active_model):
    # Register v2 and set as active (so v1 becomes retired)
    meta = {
        "model_name": "test_api_model",
        "version": "v2",
        "model_type": "classification",
        "features": ["f1", "f2", "f3"],
        "target": "y",
        "status": "candidate",
    }
    id2 = registry.register_model(sample_model, meta)
    registry.update_model_status(id2, "active")
    # Rollback to v1
    r = client.post("/models/test_api_model/rollback", json={"target_version": "v1"})
    assert r.status_code == 200
    meta_active = registry.get_active_metadata("test_api_model")
    assert meta_active.version == "v1"
