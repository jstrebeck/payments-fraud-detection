"""Prometheus metrics. Names and meaning are listed in docs/architecture.md."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PAYMENTS = Counter("fraud_payments", "Payments scored, by decision", ["decision"])
PAYMENTS_REPLAYED = Counter(
    "fraud_payments_replayed", "Duplicate submissions answered from the stored decision"
)
SCORE = Histogram("fraud_score", "Fraud score distribution", buckets=[i / 20 for i in range(1, 21)])
SCORER_LATENCY = Histogram(
    "fraud_scorer_latency_seconds",
    "Time spent in the fraud scorer",
    ["scorer"],
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
SCORER_FALLBACK = Counter(
    "fraud_scorer_fallback", "Requests scored by the fallback scorer", ["reason"]
)
MODEL_VERSION = Gauge(
    "fraud_model_version_info", "Model version currently used for scoring", ["scorer", "version"]
)
HTTP_REQUESTS = Counter(
    "fraud_http_requests", "HTTP requests by route and status", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "fraud_http_request_duration_seconds", "HTTP request latency", ["method", "route"]
)


def set_model_version(scorer: str, version: str) -> None:
    """Expose exactly one live (scorer, version) pair."""
    MODEL_VERSION.clear()
    MODEL_VERSION.labels(scorer, version).set(1)
