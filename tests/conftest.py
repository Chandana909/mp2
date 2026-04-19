"""
Pytest fixtures for ML Reliability Platform tests.

Provides temporary storage paths, sample models, and registry/API clients.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_platform.config import load_config
from ml_platform.registry import ModelRegistry
from ml_platform.models import ModelMetadata
from storage.model_store import ModelStore
from storage.metadata_store import MetadataStore
from storage.audit_store import AuditStore


@pytest.fixture
def tmp_paths(tmp_path):
    """Temporary directories for models, metadata, audit."""
    models = tmp_path / "models"
    metadata = tmp_path / "metadata"
    audit = tmp_path / "audit"
    models.mkdir()
    metadata.mkdir()
    audit.mkdir()
    return models, metadata, audit


@pytest.fixture
def model_store(tmp_paths):
    models_path, _, _ = tmp_paths
    return ModelStore(models_path=models_path)


@pytest.fixture
def metadata_store(tmp_paths):
    _, metadata_path, _ = tmp_paths
    return MetadataStore(metadata_path=metadata_path)


@pytest.fixture
def audit_store(tmp_paths):
    _, _, audit_path = tmp_paths
    return AuditStore(audit_logs_path=audit_path)


@pytest.fixture
def registry(tmp_paths):
    """ModelRegistry with temporary storage."""
    models_path, metadata_path, _ = tmp_paths
    return ModelRegistry(
        model_store=ModelStore(models_path=models_path),
        metadata_store=MetadataStore(metadata_path=metadata_path),
    )


@pytest.fixture
def sample_model():
    """Trained sklearn classifier for tests."""
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    X = np.random.RandomState(42).rand(100, 3)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    clf.fit(X, y)
    return clf


@pytest.fixture
def sample_metadata():
    """ModelMetadata for tests."""
    from datetime import datetime
    return ModelMetadata(
        model_id="test_model_v1_abc",
        model_name="test_model",
        version="v1",
        model_type="classification",
        features=["f1", "f2", "f3"],
        target="y",
        training_date=datetime.utcnow(),
        performance_metrics={"accuracy": 0.9, "f1_score": 0.88},
        status="active",
        artifact_path="",
    )


@pytest.fixture
def sample_data():
    """DataFrame with features and target."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
        "f3": np.random.rand(n),
        "y": (np.random.rand(n) > 0.5).astype(int),
    })
