"""Tests for monitoring: drift detection and metrics."""

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import DriftDetector
from monitoring.metrics import (
    compute_feature_statistics,
    compute_prediction_statistics,
    track_model_performance,
)
from ml_platform.schema import DriftReport


def test_compute_feature_statistics():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": ["x", "y", "x"],
    })
    stats = compute_feature_statistics(df)
    assert "a" in stats
    assert stats["a"]["mean"] == 2.0
    assert "b" in stats
    assert "value_counts" in stats["b"]


def test_compute_prediction_statistics_classification():
    preds = np.array([0, 1, 1, 0, 1])
    stats = compute_prediction_statistics(preds, "classification")
    assert "distribution" in stats
    assert stats["num_classes"] == 2


def test_compute_prediction_statistics_regression():
    preds = np.array([1.0, 2.0, 3.0])
    stats = compute_prediction_statistics(preds, "regression")
    assert "mean" in stats
    assert stats["mean"] == 2.0


def test_track_model_performance_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = track_model_performance(y_pred, y_true, "classification")
    assert "accuracy" in m
    assert "f1_score" in m
    assert 0 <= m["accuracy"] <= 1


def test_track_model_performance_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.0, 2.9])
    m = track_model_performance(y_pred, y_true, "regression")
    assert "rmse" in m
    assert "mae" in m
    assert "r2" in m


def test_drift_detector_no_drift():
    ref = pd.DataFrame({"x": np.random.rand(100), "y": np.random.rand(100)})
    curr = pd.DataFrame({"x": np.random.rand(100) + 0.01, "y": np.random.rand(100) + 0.01})
    detector = DriftDetector(drift_threshold=0.5)
    report = detector.compute_data_drift(ref, curr, model_id="m1")
    assert isinstance(report, DriftReport)
    assert report.model_id == "m1"
    assert "x" in report.feature_drifts
    # With high threshold, may or may not detect drift
    assert report.drift_score >= 0 and report.drift_score <= 1


def test_drift_detector_identical_no_drift():
    ref = pd.DataFrame({"x": np.random.rand(50), "y": np.random.rand(50)})
    curr = ref.copy()
    detector = DriftDetector(drift_threshold=0.15)
    report = detector.compute_data_drift(ref, curr, model_id="m1")
    assert report.drift_score == 0.0
    assert report.drift_detected is False


def test_compute_prediction_drift():
    detector = DriftDetector(drift_threshold=0.2)
    ref_preds = np.array([0, 1, 0, 1] * 25)
    curr_preds = np.array([0, 1, 0, 1] * 25)
    out = detector.compute_prediction_drift(ref_preds, curr_preds, "classification")
    assert "jensen_shannon" in out
    assert out["drift_detected"] is False
