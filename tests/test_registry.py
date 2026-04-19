"""Unit tests for model registry."""

import pytest
from datetime import datetime

from ml_platform.registry import ModelRegistry
from ml_platform.exceptions import ModelNotFoundError, DuplicateModelError
from ml_platform.models import ModelMetadata


def test_register_and_get_model(registry, sample_model, sample_metadata):
    metadata = {
        "model_name": sample_metadata.model_name,
        "version": sample_metadata.version,
        "model_type": sample_metadata.model_type,
        "features": sample_metadata.features,
        "target": sample_metadata.target,
        "performance_metrics": sample_metadata.performance_metrics,
        "status": "candidate",
    }
    model_id = registry.register_model(sample_model, metadata)
    assert model_id is not None
    assert sample_metadata.model_name in model_id

    meta = registry.get_model_metadata(model_id)
    assert meta.model_name == sample_metadata.model_name
    assert meta.version == sample_metadata.version
    assert meta.features == sample_metadata.features

    loaded = registry.get_model(model_id)
    assert loaded is not None
    pred = loaded.predict([[0.5, 0.5, 0.5]])
    assert pred.shape == (1,)


def test_duplicate_version_rejected(registry, sample_model):
    metadata = {
        "model_name": "dup_model",
        "version": "v1",
        "model_type": "classification",
        "features": ["f1", "f2", "f3"],
        "target": "y",
        "status": "candidate",
    }
    registry.register_model(sample_model, metadata)
    with pytest.raises(DuplicateModelError):
        registry.register_model(sample_model, metadata)


def test_get_active_model(registry, sample_model):
    metadata = {
        "model_name": "active_test",
        "version": "v1",
        "model_type": "classification",
        "features": ["f1", "f2", "f3"],
        "target": "y",
        "status": "candidate",
    }
    model_id = registry.register_model(sample_model, metadata)
    registry.update_model_status(model_id, "active")

    active = registry.get_active_model("active_test")
    assert active is not None
    meta = registry.get_active_metadata("active_test")
    assert meta.model_id == model_id


def test_get_active_raises_when_none(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get_active_model("nonexistent")


def test_list_models(registry, sample_model):
    metadata = {
        "model_name": "list_test",
        "version": "v1",
        "model_type": "classification",
        "features": ["f1"],
        "target": "y",
        "status": "candidate",
    }
    registry.register_model(sample_model, metadata)
    metadata["version"] = "v2"
    registry.register_model(sample_model, metadata)

    all_models = registry.list_models(model_name="list_test")
    assert len(all_models) == 2
    versions = {m.version for m in all_models}
    assert "v1" in versions and "v2" in versions


def test_update_status_retires_previous_active(registry, sample_model):
    metadata = {
        "model_name": "retire_test",
        "version": "v1",
        "model_type": "classification",
        "features": ["f1"],
        "target": "y",
        "status": "candidate",
    }
    id1 = registry.register_model(sample_model, metadata)
    metadata["version"] = "v2"
    id2 = registry.register_model(sample_model, metadata)

    registry.update_model_status(id1, "active")
    registry.update_model_status(id2, "active")

    m1 = registry.get_model_metadata(id1)
    m2 = registry.get_model_metadata(id2)
    assert m1.status == "retired"
    assert m2.status == "active"
