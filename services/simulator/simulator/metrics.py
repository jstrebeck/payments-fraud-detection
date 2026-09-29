"""Simulator-side Prometheus metrics (prefix `fraud_sim_`)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter("fraud_sim_requests", "Payments sent to the API", ["outcome"])
LATENCY = Histogram("fraud_sim_request_duration_seconds", "Round-trip time of POST /payments")
DECISIONS = Counter(
    "fraud_sim_decisions",
    "API decisions against generator ground truth",
    ["decision", "truth"],
)
