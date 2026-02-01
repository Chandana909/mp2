"""
Generic model training pipeline.

Loads config, preprocesses data, trains model via sklearn estimator,
and registers new version as candidate. Model-agnostic; parameters from config.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from platform.registry import ModelRegistry

logger = logging.getLogger(__name__)


def _load_model_config(model_name: str) -> dict:
    """Load model config from config/model_configs/{model_name}.yaml or example_model.yaml."""
    root = Path(__file__).resolve().parent.parent
    for name in (f"{model_name}.yaml", "example_model.yaml"):
        path = root / "config" / "model_configs" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_estimator(algorithm: str, hyperparameters: dict) -> Any:
    """Instantiate sklearn estimator from algorithm name and hyperparameters."""
    if algorithm == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**hyperparameters)
    if algorithm == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**hyperparameters)
    if algorithm == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**hyperparameters)
    if algorithm == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**hyperparameters)
    # Try sklearn submodules
    for module in ("sklearn.ensemble", "sklearn.linear_model", "sklearn.tree", "sklearn.svm"):
        try:
            mod = importlib.import_module(module)
            cls = getattr(mod, algorithm, None)
            if cls is not None:
                return cls(**hyperparameters)
        except Exception:
            continue
    raise ValueError(f"Unknown algorithm: {algorithm}")


class ModelTrainer:
    """
    Generic training pipeline: load data, preprocess from config, train, register as candidate.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or ModelRegistry()

    def train_model(self, data: pd.DataFrame, config: dict) -> Any:
        """
        Train a model from config (algorithm, hyperparameters, target, features).

        config should contain: model.type, model.name, features (list of names),
        target.name, training.algorithm, training.hyperparameters.

        Returns:
            Trained model object.
        """
        model_config = config.get("model", {})
        model_type = model_config.get("type", "classification")
        target_name = config.get("target", {}).get("name")
        if not target_name:
            raise ValueError("config.target.name is required")
        features = []
        for f in config.get("features", []):
            if isinstance(f, dict):
                features.append(f.get("name"))
            else:
                features.append(str(f))
        features = [x for x in features if x]
        if not features:
            features = [c for c in data.columns if c != target_name]
        train_config = config.get("training", {})
        algorithm = train_config.get("algorithm", "RandomForestClassifier")
        hyperparameters = train_config.get("hyperparameters", {})
        estimator = _get_estimator(algorithm, hyperparameters)
        X = data[features]
        y = data[target_name]
        estimator.fit(X, y)
        return estimator

    def retrain_active_model(
        self,
        model_name: str,
        data: pd.DataFrame,
        version: Optional[str] = None,
    ) -> str:
        """
        Load active model config, train new version on data, register as candidate.

        Returns:
            New model_id.
        """
        config = _load_model_config(model_name)
        if not config:
            raise ValueError(f"No config found for model: {model_name}")
        model = self.train_model(data, config)
        # Next version
        existing = self._registry.list_models(model_name=model_name)
        versions = [m.version for m in existing]
        if version is None:
            if not versions:
                version = "v1"
            else:
                try:
                    nums = [int(v.replace("v", "")) for v in versions if v.startswith("v")]
                    version = f"v{max(nums) + 1}" if nums else "v1"
                except ValueError:
                    version = f"v{len(versions) + 1}"
        model_config = config.get("model", {})
        features = []
        for f in config.get("features", []):
            if isinstance(f, dict):
                features.append(f.get("name"))
            else:
                features.append(str(f))
        features = [x for x in features if x]
        if not features:
            features = [c for c in data.columns if c != config.get("target", {}).get("name", "")]

        metadata = {
            "model_name": model_name,
            "version": version,
            "model_type": model_config.get("type", "classification"),
            "features": features,
            "target": config.get("target", {}).get("name", ""),
            "performance_metrics": {},
            "status": "candidate",
        }
        model_id = self._registry.register_model(model, metadata)
        logger.info("Retrained model registered as candidate", extra={"model_id": model_id, "model_name": model_name})
        return model_id
