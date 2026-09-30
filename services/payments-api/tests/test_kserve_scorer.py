"""KServeScorer against a fake V2 predictor (httpx MockTransport) and a fake registry lookup."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx2 as httpx
import pytest
import structlog
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from structlog.testing import capture_logs

from ml.features import FEATURE_NAMES, FEATURE_VERSION
from payments_api.config import Settings
from payments_api.main import create_app
from payments_api.scoring.errors import IncompatibleModelError
from payments_api.scoring.kserve import (
    KServeHTTPError,
    KServeResponseError,
    KServeScorer,
    KServeTimeoutError,
    ModelVersionCheckError,
)

URL = "http://predictor/v2/models/fraud-detector/infer"
FEATURES = dict.fromkeys(FEATURE_NAMES, 0.0)
TxnFactory = Callable[..., dict[str, Any]]


class FakePredictor:
    """A V2 endpoint returning predict_proba for a batch of one."""

    def __init__(self, fraud: float = 0.25, version: str | None = "2") -> None:
        self.fraud = fraud
        self.version = version
        self.status = 200
        self.raise_exc: Exception | None = None
        self.requests: list[dict[str, Any]] = []
        self.request_ids: list[str | None] = []  # x-request-id header per inference call

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ready"):
            return httpx.Response(200)
        if self.raise_exc is not None:
            raise self.raise_exc
        self.requests.append(json.loads(request.content))
        self.request_ids.append(request.headers.get("x-request-id"))
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        body: dict[str, Any] = {
            "model_name": "fraud-detector",
            "outputs": [
                {"name": "output-1", "shape": [1, 2], "datatype": "FP64",
                 "data": [1 - self.fraud, self.fraud]}
            ],
        }  # fmt: skip
        if self.version is not None:
            body["model_version"] = self.version
        if "id" in self.requests[-1]:  # MLServer echoes the request id
            body["id"] = self.requests[-1]["id"]
        return httpx.Response(200, json=body)


class FakeRegistry:
    """feature_version tag per registry version; counts lookups; can fail."""

    def __init__(self, tags: dict[str, str | None]) -> None:
        self.tags = tags
        self.calls: list[str] = []
        self.fail = False

    def __call__(self, version: str) -> str | None:
        self.calls.append(version)
        if self.fail:
            raise ConnectionError("mlflow down")
        return self.tags[version]


def _scorer(predictor: FakePredictor, registry: FakeRegistry) -> KServeScorer:
    return KServeScorer(
        url=URL,
        model_name="fraud-detector",
        timeout_s=0.3,
        lookup=registry,
        transport=httpx.MockTransport(predictor),
    )


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry({"2": FEATURE_VERSION, "3": FEATURE_VERSION, "9": "some-other-version"})


def _score(scorer: KServeScorer, n: int = 1) -> list[Any]:
    """Start, score n times (collecting results or exceptions), close."""

    async def go() -> list[Any]:
        await scorer.start()
        out: list[Any] = []
        try:
            for _ in range(n):
                try:
                    out.append(await scorer.score(FEATURES))
                except Exception as exc:
                    out.append(exc)
        finally:
            await scorer.close()
        return out

    return asyncio.run(go())


def test_scores_with_v2_request_and_reports_the_served_version(registry: FakeRegistry) -> None:
    predictor = FakePredictor(fraud=0.8)
    scorer = _scorer(predictor, registry)
    (result,) = _score(scorer)

    assert (result.score, result.scorer, result.model_version) == (
        0.8,
        "kserve",
        "fraud-detector/2",
    )
    inputs = predictor.requests[0]["inputs"]
    assert [i["name"] for i in inputs] == list(FEATURE_NAMES)
    assert all(i["datatype"] == "FP64" and i["shape"] == [1, 1] for i in inputs)
    assert scorer.model_version == "fraud-detector/2"
    live = {"scorer": "kserve", "version": "fraud-detector/2"}
    assert REGISTRY.get_sample_value("fraud_model_version_info", live) == 1.0


def test_request_id_is_propagated_to_the_predictor(registry: FakeRegistry) -> None:
    predictor = FakePredictor()
    scorer = _scorer(predictor, registry)

    async def go() -> None:
        await scorer.start()
        try:
            with structlog.contextvars.bound_contextvars(request_id="req-42"):
                await scorer.score(FEATURES)
            await scorer.score(FEATURES)  # outside a request: no id sent
        finally:
            await scorer.close()

    asyncio.run(go())
    assert predictor.requests[0]["id"] == "req-42"
    assert predictor.request_ids == ["req-42", None]
    assert "id" not in predictor.requests[1]


def test_feature_version_is_checked_once_per_version(registry: FakeRegistry) -> None:
    predictor = FakePredictor()
    scorer = _scorer(predictor, registry)
    _score(scorer, n=3)
    assert registry.calls == ["2"]


def test_version_change_updates_the_metric(registry: FakeRegistry) -> None:
    predictor = FakePredictor(version="2")
    scorer = _scorer(predictor, registry)

    async def go() -> None:
        await scorer.start()
        await scorer.score(FEATURES)
        predictor.version = "3"  # predictor restarted after a promotion
        result = await scorer.score(FEATURES)
        assert result.model_version == "fraud-detector/3"
        await scorer.close()

    asyncio.run(go())
    assert (
        REGISTRY.get_sample_value(
            "fraud_model_version_info", {"scorer": "kserve", "version": "fraud-detector/3"}
        )
        == 1.0
    )
    assert (
        REGISTRY.get_sample_value(
            "fraud_model_version_info", {"scorer": "kserve", "version": "fraud-detector/2"}
        )
        is None
    )


def test_mismatched_feature_version_is_refused_and_remembered(registry: FakeRegistry) -> None:
    scorer = _scorer(FakePredictor(version="9"), registry)
    first, second = _score(scorer, n=2)
    assert isinstance(first, IncompatibleModelError)
    assert isinstance(second, IncompatibleModelError)
    assert registry.calls == ["9"]
    assert scorer.model_version == "unknown"


def test_missing_model_version_is_refused(registry: FakeRegistry) -> None:
    (result,) = _score(_scorer(FakePredictor(version=None), registry))
    assert isinstance(result, IncompatibleModelError)
    assert registry.calls == []


def test_registry_lookup_failures_are_not_cached(registry: FakeRegistry) -> None:
    registry.fail = True
    scorer = _scorer(FakePredictor(), registry)

    async def go() -> list[Any]:
        await scorer.start()
        out: list[Any] = []
        for fail in (True, False):
            registry.fail = fail
            try:
                out.append(await scorer.score(FEATURES))
            except Exception as exc:
                out.append(exc)
        await scorer.close()
        return out

    failed, recovered = asyncio.run(go())
    assert isinstance(failed, ModelVersionCheckError)
    assert recovered.model_version == "fraud-detector/2"
    assert registry.calls == ["2", "2"]


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (lambda p: setattr(p, "raise_exc", httpx.ReadTimeout("slow")), KServeTimeoutError),
        (lambda p: setattr(p, "raise_exc", httpx.ConnectError("refused")), KServeHTTPError),
        (lambda p: setattr(p, "status", 503), KServeHTTPError),
        (lambda p: setattr(p, "fraud", 7.0), KServeResponseError),
    ],
    ids=["timeout", "connect", "5xx", "bad-score"],
)
def test_predictor_failures_raise_typed_errors(
    registry: FakeRegistry, setup: Callable[[FakePredictor], None], error: type[Exception]
) -> None:
    predictor = FakePredictor()
    setup(predictor)
    (result,) = _score(_scorer(predictor, registry))
    assert isinstance(result, error)


def test_malformed_body_is_a_response_error(registry: FakeRegistry) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"outputs": []})

    scorer = KServeScorer(
        url=URL, model_name="fraud-detector", timeout_s=0.3, lookup=registry,
        transport=httpx.MockTransport(handler),
    )  # fmt: skip
    (result,) = _score(scorer)
    assert isinstance(result, KServeResponseError)


def test_ready_reflects_the_predictor(registry: FakeRegistry) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def go() -> tuple[bool, bool]:
        up = _scorer(FakePredictor(), registry)
        dead = KServeScorer(
            url=URL, model_name="fraud-detector", timeout_s=0.3, lookup=registry,
            transport=httpx.MockTransport(down),
        )  # fmt: skip
        await up.start()
        await dead.start()
        result = (await up.ready(), await dead.ready())
        await up.close()
        await dead.close()
        return result

    assert asyncio.run(go()) == (True, False)


@pytest.fixture
def kserve_app(
    settings: Settings, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FakePredictor]]:
    predictor = FakePredictor(fraud=0.9)
    monkeypatch.setattr("payments_api.main.build_scorer", lambda _s: _scorer(predictor, registry))
    with TestClient(create_app(settings)) as client:
        yield client, predictor


def test_api_records_the_kserve_decision(
    kserve_app: tuple[TestClient, FakePredictor], make_txn: TxnFactory
) -> None:
    client, _ = kserve_app
    resp = client.post("/payments", json=make_txn())
    assert resp.status_code == 201
    body = resp.json()
    assert (body["scorer"], body["model_version"], body["decision"]) == (
        "kserve", "fraud-detector/2", "declined",
    )  # fmt: skip
    assert client.get("/readyz").json()["scorer"] == "ok"


def test_api_request_id_reaches_the_predictor_and_the_record(
    kserve_app: tuple[TestClient, FakePredictor], make_txn: TxnFactory
) -> None:
    client, predictor = kserve_app
    resp = client.post("/payments", json=make_txn(), headers={"x-request-id": "sim-abc123"})
    assert resp.headers["x-request-id"] == "sim-abc123"
    assert resp.json()["request_id"] == "sim-abc123"
    assert predictor.requests[-1]["id"] == "sim-abc123"
    assert predictor.request_ids[-1] == "sim-abc123"
    record = client.get(f"/payments/{resp.json()['payment_id']}").json()
    assert record["request_id"] == "sim-abc123"


def test_api_falls_back_to_rules_on_predictor_timeout(
    kserve_app: tuple[TestClient, FakePredictor], make_txn: TxnFactory
) -> None:
    client, predictor = kserve_app
    predictor.raise_exc = httpx.ReadTimeout("slow")
    labels = {"reason": "KServeTimeoutError"}
    before = REGISTRY.get_sample_value("fraud_scorer_fallback_total", labels) or 0.0

    resp = client.post("/payments", json=make_txn(transaction_id="t-timeout"))
    assert resp.status_code == 201
    assert resp.json()["scorer"] == "rule"
    assert REGISTRY.get_sample_value("fraud_scorer_fallback_total", labels) == before + 1


def test_api_logs_carry_the_request_id_on_fallback(
    kserve_app: tuple[TestClient, FakePredictor], make_txn: TxnFactory
) -> None:
    client, predictor = kserve_app
    predictor.status = 503
    with capture_logs(processors=[structlog.contextvars.merge_contextvars]) as logs:
        client.post(
            "/payments", json=make_txn(transaction_id="t-logs"), headers={"x-request-id": "rid-7"}
        )
    events = {e["event"]: e for e in logs}
    for name in ("scorer_failed_using_fallback", "payment_scored", "request"):
        assert events[name]["request_id"] == "rid-7", name
