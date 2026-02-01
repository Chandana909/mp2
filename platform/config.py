"""
Configuration management for the ML Reliability Platform.

Loads platform_config.yaml and exposes settings via get_config().
Uses a single config load; paths are relative to project root or explicit.
"""

from pathlib import Path
from typing import Any, Optional

import yaml

# Default path relative to project root (mp2 or ml-platform)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "platform_config.yaml"

_config: Optional[dict] = None


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load platform configuration from YAML file.

    Args:
        config_path: Path to platform_config.yaml. If None, uses default.

    Returns:
        Configuration dictionary. Never returns None; returns empty dict on error.
    """
    global _config
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        _config = _default_config()
        return _config
    try:
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
        return _config
    except Exception:
        _config = _default_config()
        return _config


def _default_config() -> dict:
    """Return minimal default config when file is missing."""
    return {
        "platform": {"name": "ML Reliability Platform", "version": "1.0.0"},
        "storage": {
            "models_path": "./models",
            "metadata_path": "./storage/metadata",
            "audit_logs_path": "./storage/audit_logs",
        },
        "monitoring": {
            "drift_detection": {
                "enabled": True,
                "drift_threshold": 0.15,
                "window_size": 1000,
            },
            "metrics": {"track_performance": True},
        },
        "lifecycle": {
            "retraining": {"auto_trigger_on_drift": True, "drift_trigger_threshold": 0.20},
            "promotion": {"criteria": [{"metric": "accuracy", "improvement_threshold": 0.02}]},
            "rollback": {"auto_rollback_on_performance_drop": True, "performance_drop_threshold": 0.05},
        },
        "serving": {"host": "0.0.0.0", "port": 8000, "workers": 4, "log_level": "info"},
    }


def get_config(key_path: Optional[str] = None) -> Any:
    """
    Get configuration value(s).

    Args:
        key_path: Dot-separated path, e.g. "storage.models_path" or "monitoring.drift_detection.enabled".
                  If None, returns full config dict.

    Returns:
        Value at key_path, or full config if key_path is None.
    """
    global _config
    if _config is None:
        load_config()
    assert _config is not None
    if key_path is None:
        return _config
    keys = key_path.split(".")
    value: Any = _config
    for k in keys:
        value = value.get(k)
        if value is None:
            return None
    return value


def get_storage_paths() -> tuple[Path, Path, Path]:
    """Return (models_path, metadata_path, audit_logs_path) as Paths."""
    root = Path(__file__).resolve().parent.parent
    cfg = get_config("storage") or {}
    models = root / (cfg.get("models_path") or "models")
    metadata = root / (cfg.get("metadata_path") or "storage/metadata")
    audit_logs = root / (cfg.get("audit_logs_path") or "storage/audit_logs")
    return models, metadata, audit_logs
