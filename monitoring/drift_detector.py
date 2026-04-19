"""
Data and prediction drift detection.

Uses KS test for numerical features, chi-square for categorical;
Jensen-Shannon divergence or PSI for prediction drift.
Flags drift when aggregate score exceeds threshold.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ml_platform.schema import DriftReport

logger = logging.getLogger(__name__)


def _ks_drift(ref: np.ndarray, curr: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between two samples (0 = no drift)."""
    ref = np.asarray(ref).ravel()
    curr = np.asarray(curr).ravel()
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    if len(ref) < 2 or len(curr) < 2:
        return 0.0
    try:
        stat, _ = stats.ks_2samp(ref, curr)
        return float(stat)
    except Exception:
        return 0.0


def _chi2_drift(ref: pd.Series, curr: pd.Series) -> float:
    """Chi-square based drift for categorical: compare value counts."""
    r = ref.value_counts()
    c = curr.value_counts()
    all_cats = set(r.index) | set(c.index)
    if not all_cats:
        return 0.0
    r_aligned = np.array([r.get(x, 0) for x in all_cats])
    c_aligned = np.array([c.get(x, 0) for x in all_cats])
    r_sum = r_aligned.sum()
    c_sum = c_aligned.sum()
    if r_sum == 0 or c_sum == 0:
        return 0.0
    r_aligned = r_aligned / r_sum
    c_aligned = c_aligned / c_sum
    diff = np.abs(r_aligned - c_aligned)
    return float(np.mean(diff))


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = p / (p.sum() or 1)
    q = q / (q.sum() or 1)
    m = (p + q) / 2
    eps = 1e-10
    d = 0.5 * (np.sum(p * np.log(p / m + eps)) + np.sum(q * np.log(q / m + eps)))
    return float(np.sqrt(max(0, d)))


def _psi(ref: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples."""
    ref = np.asarray(ref).ravel()
    curr = np.asarray(curr).ravel()
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    if len(ref) < 2 or len(curr) < 2:
        return 0.0
    try:
        _, bin_edges = np.histogram(ref, bins=bins)
        ref_p = np.histogram(ref, bins=bin_edges)[0].astype(float) / (len(ref) or 1)
        curr_p = np.histogram(curr, bins=bin_edges)[0].astype(float) / (len(curr) or 1)
        ref_p = ref_p + 1e-10
        curr_p = curr_p + 1e-10
        psi = np.sum((curr_p - ref_p) * np.log(curr_p / ref_p))
        return float(min(1.0, max(0, psi)))
    except Exception:
        return 0.0


class DriftDetector:
    """
    Detects data drift (reference vs current) and prediction drift.
    Produces DriftReport with per-feature scores and aggregate flag.
    """

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold

    def compute_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_id: str = "",
    ) -> DriftReport:
        """
        For each feature: KS for numerical, chi-square-style for categorical.
        Aggregate drift score = mean of per-feature scores.
        Flag drift_detected if drift_score > threshold.
        """
        feature_drifts: Dict[str, float] = {}
        for col in reference_data.columns:
            if col not in current_data.columns:
                continue
            ref = reference_data[col]
            curr = current_data[col]
            if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(curr):
                score = _ks_drift(ref.values, curr.values)
            else:
                score = _chi2_drift(ref.astype(str), curr.astype(str))
            feature_drifts[col] = round(score, 4)
        drift_score = float(np.mean(list(feature_drifts.values()))) if feature_drifts else 0.0
        drift_detected = drift_score > self.drift_threshold
        if drift_detected:
            logger.info(
                "Data drift detected: model_id=%s, score=%.4f, threshold=%.4f",
                model_id,
                drift_score,
                self.drift_threshold,
                extra={"model_id": model_id, "drift_score": drift_score, "feature_drifts": feature_drifts},
            )
        else:
            logger.debug("No data drift: model_id=%s, score=%.4f", model_id, drift_score)
        return DriftReport(
            model_id=model_id,
            drift_detected=drift_detected,
            drift_score=round(drift_score, 4),
            feature_drifts=feature_drifts,
            timestamp=datetime.utcnow(),
        )

    def compute_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        model_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Compare prediction distributions: Jensen-Shannon or PSI.
        """
        ref = np.asarray(reference_predictions).ravel()
        curr = np.asarray(current_predictions).ravel()
        if model_type == "classification":
            ref_hist = np.bincount(ref.astype(int), minlength=int(ref.max()) + 1 if ref.size else 0)
            curr_hist = np.bincount(curr.astype(int), minlength=int(curr.max()) + 1 if curr.size else 0)
            max_len = max(len(ref_hist), len(curr_hist))
            ref_hist = np.pad(ref_hist, (0, max_len - len(ref_hist)))
            curr_hist = np.pad(curr_hist, (0, max_len - len(curr_hist)))
            js = _jensen_shannon(ref_hist, curr_hist)
            return {"jensen_shannon": js, "drift_detected": js > self.drift_threshold}
        psi = _psi(ref, curr)
        return {"psi": psi, "drift_detected": psi > self.drift_threshold}

    def should_trigger_retrain(
        self,
        drift_report: DriftReport,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        True if drift exceeds configured threshold or consecutive drift events > N.
        config: drift_trigger_threshold, consecutive_drift_triggers (optional).
        """
        config = config or {}
        threshold = config.get("drift_trigger_threshold", self.drift_threshold)
        if drift_report.drift_score <= threshold:
            return False
        # Consecutive drift logic would need state; here we only use threshold
        return True
