"""
Prometheus metrics for monitoring platform health.

Exposes metrics for request count, latency, predictions, drift detections, and errors.
Expose via GET /metrics using prometheus_client.generate_latest().
If prometheus_client is not installed, metrics are no-ops and GET /metrics returns a message.
"""

try:
    from prometheus_client import Counter, Histogram, Gauge
    _prometheus_available = True
except ImportError:
    _prometheus_available = False

if _prometheus_available:
    request_count = Counter(
        "platform_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    request_latency = Histogram(
        "platform_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
    )
    prediction_count = Counter(
        "platform_predictions_total",
        "Total predictions made",
        ["model_name", "model_version"],
    )
    drift_detected_count = Counter(
        "platform_drift_detections_total",
        "Total drift detections",
        ["model_id"],
    )
    active_models_gauge = Gauge(
        "platform_active_models",
        "Number of active models",
    )
    error_count = Counter(
        "platform_errors_total",
        "Total errors",
        ["error_type"],
    )
else:
    request_count = None
    request_latency = None
    prediction_count = None
    drift_detected_count = None
    active_models_gauge = None
    error_count = None
