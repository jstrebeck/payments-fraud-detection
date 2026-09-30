from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from alembic import command
from fastapi.testclient import TestClient

from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions
from ml.features import FEATURE_NAMES
from ml.features.batch import featurize
from payments_api.migrate import alembic_config

TxnFactory = Callable[..., dict[str, Any]]


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"database": "ok", "scorer": "ok"}


def test_create_and_fetch_payment(client: TestClient, make_txn: TxnFactory) -> None:
    resp = client.post("/payments", json=make_txn())
    assert resp.status_code == 201
    body = resp.json()
    assert body["decision"] == "approved"
    assert 0.0 <= body["score"] <= 1.0
    assert body["scorer"] == "rule"
    assert body["model_version"] == "rules-v1"
    assert resp.headers["x-request-id"]

    assert body["request_id"] == resp.headers["x-request-id"]

    record = client.get(f"/payments/{body['payment_id']}").json()
    assert record["request_id"] == body["request_id"]
    assert record["transaction"] == make_txn() | {"timestamp": "2026-01-05T12:00:00Z"}
    assert tuple(record["features"]) == FEATURE_NAMES
    assert record["decision"] == body["decision"]


def test_duplicate_transaction_is_idempotent(client: TestClient, make_txn: TxnFactory) -> None:
    first = client.post("/payments", json=make_txn())
    second = client.post("/payments", json=make_txn(amount=9999.0))
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json() == first.json()


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_fraud": True},  # labels must never reach the API
        {"card_token": "4111111111111111"},  # only opaque tokens
        {"amount": -1},
        {"merchant_category": "casino"},
        {"ip_country": "XX"},
        {"timestamp": "2026-01-05T12:00:00"},  # naive timestamps are ambiguous
    ],
)
def test_invalid_payments_are_rejected(
    client: TestClient, overrides: dict[str, object], make_txn: TxnFactory
) -> None:
    assert client.post("/payments", json=make_txn(**overrides)).status_code == 422


def test_unknown_payment_is_404(client: TestClient) -> None:
    resp = client.get("/payments/00000000-0000-4000-8000-000000000000")
    assert resp.status_code == 404


def test_card_history_drives_the_decision(client: TestClient, make_txn: TxnFactory) -> None:
    """A card-testing burst: later charges see the earlier ones through the stored history."""
    start = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    decisions = []
    for i in range(8):
        txn = make_txn(
            transaction_id=f"burst-{i}",
            timestamp=(start + timedelta(seconds=30 * i)).isoformat(),
            merchant_id="mer_000099",
            merchant_category="digital_goods",
            channel="ecommerce",
            amount=1.0 + i,
        )
        decisions.append(client.post("/payments", json=txn).json()["decision"])
    assert decisions[0] == "approved"
    assert decisions[-1] != "approved"


def test_metrics_exposed(client: TestClient, make_txn: TxnFactory) -> None:
    client.post("/payments", json=make_txn())
    text = client.get("/metrics").text
    for name in (
        "fraud_payments_total",
        "fraud_score_bucket",
        "fraud_scorer_latency_seconds_bucket",
        "fraud_model_version_info",
        "fraud_http_requests_total",
    ):
        assert name in text


def test_online_features_match_batch_features(client: TestClient) -> None:
    """Training/serving parity: features stored by the API equal the batch path's output."""
    table = generate(GeneratorConfig(seed=7, customers=15, days=4, fraud_rate=0.05))
    txns = [t.unlabelled() for t in iter_transactions(table)]
    assert len(txns) > 50

    payment_ids = []
    for txn in txns:  # in time order, as the simulator sends them
        resp = client.post("/payments", json=txn.model_dump(mode="json"))
        assert resp.status_code == 201
        payment_ids.append(resp.json()["payment_id"])

    expected = featurize(txns)
    for payment_id, want in zip(payment_ids, expected, strict=True):
        got = client.get(f"/payments/{payment_id}").json()["features"]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_caller_request_id_is_kept_and_replays_keep_the_original(
    client: TestClient, make_txn: TxnFactory
) -> None:
    first = client.post("/payments", json=make_txn(), headers={"x-request-id": "sim-1"})
    assert first.status_code == 201
    assert first.json()["request_id"] == "sim-1"

    replay = client.post("/payments", json=make_txn(), headers={"x-request-id": "sim-2"})
    assert replay.status_code == 200
    assert replay.headers["x-request-id"] == "sim-2"  # this call's own id
    assert replay.json()["request_id"] == "sim-1"  # the stored decision's id


@pytest.mark.parametrize("bad", ["x" * 65, "has space", 'forged" level="error'])
def test_malformed_request_id_is_replaced(
    client: TestClient, make_txn: TxnFactory, bad: str
) -> None:
    resp = client.post("/payments", json=make_txn(), headers={"x-request-id": bad})
    rid = resp.headers["x-request-id"]
    assert rid != bad
    assert len(rid) == 32
    assert resp.json()["request_id"] == rid


def test_migrations_match_the_models(database_url: str) -> None:
    """`alembic check`: the migrated schema has no drift from models.py."""
    command.check(alembic_config(database_url))
