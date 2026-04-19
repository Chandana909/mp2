"""
ML Reliability Platform — Python SDK Client.

Drop-in client for all platform operations: upload, predict, retrain,
optimize hyperparameters, promote candidates, and monitor drift.
Uses httpx for sync HTTP; all methods raise on non-2xx.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MLPlatformClient:
    """
    Unified Python client for the ML Reliability & Lifecycle Supervision Platform.

    Usage:
        client = MLPlatformClient("http://localhost:8000")
        client.trigger_pipeline("churn_model")
        client.optimize("churn_model", n_trials=12)
        client.promote("churn_model")
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    # ── Health & Stats ────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check platform health. Returns status dict."""
        r = self._http.get(f"{self.base_url}/health")
        r.raise_for_status()
        data = r.json()
        logger.info("Health: %s | models=%s active=%s",
                    data.get("status"), data["checks"].get("model_count"),
                    data["checks"].get("active_count"))
        return data

    def stats(self) -> Dict[str, Any]:
        """Retrieve platform-wide statistics (model counts, best performer)."""
        r = self._http.get(f"{self.base_url}/stats")
        r.raise_for_status()
        return r.json()

    # ── Model Registry ────────────────────────────────────────────────────────

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered model versions."""
        url = f"{self.base_url}/models"
        if model_name:
            url = f"{self.base_url}/models/{model_name}/versions"
        r = self._http.get(url)
        r.raise_for_status()
        return r.json()

    def upload_model(
        self,
        file_path: str,
        model_name: str,
        version: str,
        features: List[str],
        model_type: str = "classification",
        accuracy: float = 0.0,
    ) -> Dict[str, Any]:
        """Upload a pre-trained .joblib model and register it as a candidate."""
        logger.info("Uploading %s %s from %s", model_name, version, file_path)
        with open(file_path, "rb") as f:
            r = self._http.post(
                f"{self.base_url}/models/upload",
                data={
                    "model_name": model_name,
                    "version": version,
                    "model_type": model_type,
                    "features": json.dumps(features),
                    "target": "target",
                    "accuracy": accuracy,
                },
                files={"model_file": (file_path, f, "application/octet-stream")},
            )
        r.raise_for_status()
        data = r.json()
        logger.info("Upload OK — model_id=%s status=%s", data.get("model_id"), data.get("status_assigned"))
        return data

    def promote(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Promote a candidate to active. If version=None, picks best candidate."""
        body = {"version": version} if version else {}
        r = self._http.post(
            f"{self.base_url}/models/{model_name}/promote",
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        logger.info("Promoted %s → %s (acc=%.4f)", model_name,
                    data.get("promoted_version"), data.get("accuracy") or 0)
        return data

    def rollback(self, model_name: str, target_version: str) -> Dict[str, Any]:
        """Rollback model_name to a specific version."""
        r = self._http.post(
            f"{self.base_url}/models/{model_name}/rollback",
            json={"target_version": target_version},
        )
        r.raise_for_status()
        return r.json()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, model_name: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Get a single prediction from the active model."""
        r = self._http.post(
            f"{self.base_url}/predict",
            json={"model_name": model_name, "features": features},
        )
        r.raise_for_status()
        data = r.json()
        logger.info("Prediction=%s confidence=%.3f version=%s",
                    data.get("prediction"), data.get("confidence") or 0,
                    data.get("model_version"))
        return data

    def predict_batch(
        self, model_name: str, feature_rows: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Run multiple predictions sequentially and return all results.
        (True batch endpoint forthcoming; this wraps sequential calls cleanly.)
        """
        results = []
        for i, features in enumerate(feature_rows):
            try:
                results.append(self.predict(model_name, features))
            except Exception as exc:
                logger.warning("Batch row %d failed: %s", i, exc)
                results.append({"error": str(exc), "row": i})
        return results

    # ── Pipeline & Optimization ───────────────────────────────────────────────

    def trigger_pipeline(self, model_name: str, n_estimators_delta: int = 20) -> Dict[str, Any]:
        """Trigger an automated retraining pipeline run."""
        logger.info("Triggering retrain pipeline for %s", model_name)
        r = self._http.post(
            f"{self.base_url}/pipeline/retrain",
            json={"model_name": model_name, "improvement_factor": 0.05,
                  "n_estimators_delta": n_estimators_delta},
        )
        r.raise_for_status()
        data = r.json()
        logger.info("Retrain complete — new_version=%s acc=%.4f",
                    data.get("new_version"), data.get("new_accuracy") or 0)
        return data

    def optimize(
        self,
        model_name: str,
        n_trials: int = 12,
        triggered_by: str = "manual",
        model_type_hint: str = "random_forest",
        auto_promote: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a hyperparameter optimization search.
        Tries n_trials random parameter combinations concurrently and registers
        the best one as a new candidate if it beats the active model.

        Args:
            model_name:       Target model in the registry.
            n_trials:         Number of random hyperparameter combinations to try.
            triggered_by:     Audit label — "manual", "drift", "scheduled".
            model_type_hint:  "random_forest" | "gradient_boosting".
            auto_promote:     If True, automatically promote the new candidate
                              after a successful optimization run.

        Returns:
            Full optimization run dict including all trial results.
        """
        logger.info("Starting hyperparameter optimization: %s (%d trials)", model_name, n_trials)
        t0 = time.time()
        r = self._http.post(
            f"{self.base_url}/optimizer/run",
            json={
                "model_name": model_name,
                "n_trials": n_trials,
                "triggered_by": triggered_by,
                "model_type_hint": model_type_hint,
            },
        )
        r.raise_for_status()
        data = r.json()
        elapsed = round(time.time() - t0, 2)

        best = data.get("best_trial") or {}
        logger.info(
            "Optimization done in %.1fs — best_acc=%.4f registered_version=%s",
            elapsed, best.get("accuracy") or 0, data.get("registered_version") or "none",
        )

        if auto_promote and data.get("registered_version"):
            logger.info("Auto-promoting %s %s", model_name, data["registered_version"])
            self.promote(model_name, version=data["registered_version"])

        return data

    def optimizer_history(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """List recent optimization runs."""
        url = f"{self.base_url}/optimizer/history"
        if model_name:
            url += f"?model_name={model_name}"
        r = self._http.get(url)
        r.raise_for_status()
        return r.json()

    # ── Drift ─────────────────────────────────────────────────────────────────

    def simulate_drift(self, model_name: str, severity: float = 0.5) -> Dict[str, Any]:
        """Inject simulated data drift at given severity (0.0–1.0)."""
        r = self._http.post(
            f"{self.base_url}/drift/simulate",
            json={"model_name": model_name, "severity": severity, "label": "sdk"},
        )
        r.raise_for_status()
        data = r.json()
        logger.info("Drift score=%.4f detected=%s recommendation=%s",
                    data.get("drift_score"), data.get("drift_detected"),
                    data.get("recommendation"))
        return data

    def drift_history(self, limit: int = 20) -> Dict[str, Any]:
        """Retrieve recent drift events from the audit ledger."""
        r = self._http.get(f"{self.base_url}/drift/history?limit={limit}")
        r.raise_for_status()
        return r.json()

    # ── Full Pipeline Demo ────────────────────────────────────────────────────

    @classmethod
    def run_full_demo(cls, model_name: str = "demo_churn_model"):
        """
        End-to-end demo flow:
        1. Health check
        2. Trigger retraining
        3. Run hyperparameter optimization (12 trials)
        4. Auto-promote best candidate
        5. Simulate drift
        6. Get a prediction
        """
        client = cls()

        print("\n" + "="*60)
        print("  ML RELIABILITY PLATFORM — FULL AUTO PIPELINE DEMO")
        print("="*60)

        # 1. Health
        h = client.health()
        print(f"\n[1] Health: {h['status'].upper()} | {h['checks'].get('model_count', 0)} models")

        # 2. Retrain
        rt = client.trigger_pipeline(model_name)
        print(f"\n[2] Retrain: new version {rt.get('new_version')} | acc={rt.get('new_accuracy'):.4f}")

        # 3. Optimize
        opt = client.optimize(model_name, n_trials=10, auto_promote=False)
        best = opt.get("best_trial") or {}
        print(f"\n[3] Optimize: {opt['trials_done']} trials | best_acc={best.get('accuracy', 0):.4f}")
        print(f"    Best params: {best.get('params')}")
        if opt.get("registered_version"):
            print(f"    Registered: {opt['registered_version']}")

        # 4. Promote best
        pr = client.promote(model_name)
        print(f"\n[4] Promoted: {pr.get('promoted_version')} | acc={pr.get('accuracy') or 0:.4f}")

        # 5. Drift
        dr = client.simulate_drift(model_name, severity=0.55)
        print(f"\n[5] Drift: score={dr.get('drift_score'):.3f} | {dr.get('recommendation')}")

        # 6. Predict
        pred = client.predict(model_name, {
            "age": 35, "tenure_months": 12, "monthly_charge": 85.5,
            "total_charge": 1024.0, "support_calls": 2,
        })
        print(f"\n[6] Prediction: {pred.get('prediction')} | confidence={pred.get('confidence', 0):.3f}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    MLPlatformClient.run_full_demo()
