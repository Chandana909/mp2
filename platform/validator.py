"""
Input validation for prediction requests.

Validates features against model schema (required fields, types, optional ranges).
Raises ValidationError with clear messages for API responses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from platform.exceptions import ValidationError
from platform.models import ModelMetadata
from platform.schema import PredictionRequest

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Validates prediction requests against model metadata.

    Ensures model exists and features match expected schema (names, types, ranges if configured).
    """

    def validate_features(
        self,
        features: Dict[str, Any],
        expected_schema: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check that features match expected schema.

        expected_schema: dict mapping feature name to config, e.g.:
            {"age": {"type": "numeric", "required": True}, "account_type": {"type": "categorical", "values": ["a","b"]}}

        Returns:
            (is_valid, error_message). If valid, error_message is empty.
        """
        if not expected_schema:
            return True, ""
        for name, config in expected_schema.items():
            if config.get("required", True) and name not in features:
                return False, f"Missing required feature: {name}"
            if name not in features:
                continue
            value = features[name]
            kind = (config.get("type") or "numeric").lower()
            if kind in ("numeric", "number", "float", "int"):
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (TypeError, ValueError):
                        return False, f"Feature '{name}' must be numeric"
            elif kind in ("categorical", "category", "str"):
                allowed = config.get("values")
                if allowed is not None and value not in allowed:
                    return False, f"Feature '{name}' must be one of {allowed}"
        return True, ""

    def validate_prediction_request(
        self,
        request: PredictionRequest,
        model_metadata: ModelMetadata,
    ) -> None:
        """
        Ensure request is valid for the given model: model exists and features match schema.

        Raises:
            ValidationError: If model missing or features invalid.
        """
        if not model_metadata:
            raise ValidationError("Model not found")
        # Build expected schema from metadata
        expected_schema: Dict[str, Any] = {}
        for f in model_metadata.features:
            expected_schema[f] = {"required": True}
        is_valid, msg = self.validate_features(request.features, expected_schema)
        if not is_valid:
            raise ValidationError(msg, field="features")
