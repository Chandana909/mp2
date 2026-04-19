"""
Drift Simulator for manual/demo drift injection.

Allows simulating data drift at configurable severity levels
for demonstration and testing purposes. Produces DriftReport
output identical to real drift detection so the UI/API response
is consistent with actual monitoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml_platform.schema import DriftReport
from ml_platform.config import get_storage_paths

logger = logging.getLogger(__name__)

_DRIFT_EVENTS_FILE = None


def _drift_events_path() -> Path:
    global _DRIFT_EVENTS_FILE
    if _DRIFT_EVENTS_FILE is None:
        _, _, audit_path = get_storage_paths()
        p = Path(audit_path).parent / "drift_events.jsonl"
        _DRIFT_EVENTS_FILE = p
    return _DRIFT_EVENTS_FILE


def _append_drift_event(event: dict) -> None:
    path = _drift_events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def load_drift_history(limit: int = 50) -> List[dict]:
    """Load recent drift events from the append-only JSONL store."""
    path = _drift_events_path()
    if not path.exists():
        return []
    events: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    # Return most-recent first
    return list(reversed(events[-limit:]))


class DriftSimulator:
    """
    Simulates data drift at configurable severity.

    Severity levels (0.0 → 1.0):
      0.0–0.15  → None    (below detection threshold)
      0.15–0.35 → Low     (borderline)
      0.35–0.60 → Medium  (clearly detected)
      0.60–1.0  → High    (severe, triggers retrain recommendation)

    The per-feature scores are generated with realistic statistical noise
    so the output mirrors what the real DriftDetector would produce.
    """

    # Realistic feature sets for known model archetypes
    FEATURE_TEMPLATES = {
        "customer_churn": ["age", "tenure_months", "monthly_charge", "total_charge", "support_calls"],
        "fraud_detector": ["transaction_amount", "merchant_category", "time_of_day", "card_present", "user_age"],
        "default": ["feature_1", "feature_2", "feature_3", "feature_4"],
    }

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold

    def simulate(
        self,
        model_name: str,
        model_id: str,
        severity: float = 0.5,
        features: Optional[List[str]] = None,
        label: str = "manual",
    ) -> DriftReport:
        """
        Generate a simulated DriftReport.

        Args:
            model_name: Logical model name (used to look up feature templates).
            model_id:   Model artifact ID.
            severity:   0.0–1.0 drift intensity.
            features:   Override feature list; falls back to template or default.
            label:      Tag stored in drift history ("manual", "scheduled", "auto").

        Returns:
            DriftReport compatible with real detection output.
        """
        severity = float(np.clip(severity, 0.0, 1.0))

        # Resolve features
        feature_list = (
            features
            or self.FEATURE_TEMPLATES.get(model_name)
            or self.FEATURE_TEMPLATES["default"]
        )

        # Generate per-feature drift scores with noise around severity
        rng = np.random.default_rng(seed=int(severity * 1000) % 9999)
        raw_scores = rng.normal(loc=severity, scale=0.07, size=len(feature_list))
        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        feature_drifts: Dict[str, float] = {
            feat: round(float(s), 4)
            for feat, s in zip(feature_list, raw_scores)
        }

        # Aggregate
        drift_score = round(float(np.mean(raw_scores)), 4)
        drift_detected = drift_score > self.drift_threshold

        # Algorithm breakdown (for display)
        ks_score = round(float(np.clip(rng.normal(severity, 0.05), 0, 1)), 4)
        js_score = round(float(np.clip(rng.normal(severity * 0.9, 0.06), 0, 1)), 4)
        chi_score = round(float(np.clip(rng.normal(severity * 0.85, 0.07), 0, 1)), 4)
        psi_score = round(float(np.clip(rng.normal(severity * 1.05, 0.06), 0, 1)), 4)

        # Determine severity label
        if severity < 0.15:
            severity_label = "none"
        elif severity < 0.35:
            severity_label = "low"
        elif severity < 0.60:
            severity_label = "medium"
        else:
            severity_label = "high"

        report = DriftReport(
            model_id=model_id,
            drift_detected=drift_detected,
            drift_score=drift_score,
            feature_drifts=feature_drifts,
            timestamp=datetime.utcnow(),
        )

        # Persist event
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "model_id": model_id,
            "severity": severity,
            "severity_label": severity_label,
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "feature_drifts": feature_drifts,
            "algorithms": {
                "kolmogorov_smirnov": ks_score,
                "jensen_shannon": js_score,
                "chi_square": chi_score,
                "population_stability_index": psi_score,
            },
            "label": label,
            "recommendation": (
                "retrain_model" if drift_score > 0.35
                else "monitor_closely" if drift_detected
                else "no_action"
            ),
        }
        _append_drift_event(event)
        logger.info(
            "Drift simulated: model=%s severity=%.2f score=%.4f detected=%s",
            model_name, severity, drift_score, drift_detected,
        )
        return report
