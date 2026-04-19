"""
Abstraction layer for making predictions.

Converts features to the correct format, invokes the model,
and returns standardized (prediction, confidence) for classification/regression.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml_platform.models import ModelMetadata

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Makes predictions from a loaded model using model metadata.

    Handles feature ordering, classification vs regression,
    and extraction of confidence when available.
    """

    def predict(
        self,
        model: Any,
        features: Dict[str, Any],
        metadata: ModelMetadata,
    ) -> Tuple[Union[int, float, str], Optional[float]]:
        """
        Run prediction and return (prediction, confidence).

        Args:
            model: Loaded sklearn-compatible model.
            features: Dict of feature name -> value.
            metadata: Model metadata (for feature order and model_type).

        Returns:
            (prediction, confidence). confidence is None for regression
            or when not available.
        """
        # Build input in feature order
        ordered = [features.get(name) for name in metadata.features]
        # Handle single sample: 2D array expected by sklearn
        X = np.array(ordered, dtype=float, ndmin=2)
        # Some models expect float; handle mixed types
        try:
            pred = model.predict(X)
        except (ValueError, TypeError):
            # Try with object dtype for categorical
            X = np.array([ordered], dtype=object)
            pred = model.predict(X)
        out = pred.flat[0]
        # Confidence: predict_proba for classification if available
        confidence: Optional[float] = None
        if metadata.model_type == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba is not None and proba.size > 0:
                    confidence = float(np.max(proba))
            except Exception as e:
                logger.debug("predict_proba failed: %s", e)
        # Convert to Python scalar
        if isinstance(out, (np.integer, np.floating)):
            out = out.item()
        return out, confidence

    def batch_predict(
        self,
        model: Any,
        features_list: List[Dict[str, Any]],
        metadata: ModelMetadata,
    ) -> List[Tuple[Union[int, float, str], Optional[float]]]:
        """
        Run predictions for a list of feature dicts.

        Returns list of (prediction, confidence) in same order.
        """
        results: List[Tuple[Union[int, float, str], Optional[float]]] = []
        for feat in features_list:
            results.append(self.predict(model, feat, metadata))
        return results
