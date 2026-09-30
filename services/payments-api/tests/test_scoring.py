from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from pydantic import ValidationError

from ml.features import FEATURE_NAMES
from payments_api import scoring
from payments_api.config import Settings
from payments_api.main import create_app
from payments_api.policy import DecisionPolicy
from payments_api.scoring import RuleScorer, ScoreResult, build_scorer

TxnFactory = Callable[..., dict[str, Any]]


def _features(**overrides: float) -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0) | overrides


@pytest.mark.parametrize(
    ("score", "decision"),
    [(0.0, "approved"), (0.49, "approved"), (0.5, "review"), (0.79, "review"), (0.8, "declined")],
)
def test_policy_thresholds(score: float, decision: str) -> None:
    assert DecisionPolicy(0.5, 0.8).decide(score) == decision


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Settings(review_threshold=0.9, decline_threshold=0.5, _env_file=None)


def test_rule_scorer_clean_transaction_scores_zero() -> None:
    result = asyncio.run(RuleScorer().score(_features()))
    assert result == ScoreResult(0.0, "rule", "rules-v1")


@pytest.mark.parametrize(
    ("features", "minimum"),
    [
        ({"txn_count_1h": 6}, 0.5),
        ({"speed_from_last_kmh": 5000, "channel_code": 0}, 0.5),
        ({"amount_zscore": 10, "is_new_device": 1, "merchant_risk_tier": 2}, 0.6),
    ],
)
def test_rule_scorer_flags_patterns(features: dict[str, float], minimum: float) -> None:
    assert asyncio.run(RuleScorer().score(_features(**features))).score >= minimum


def test_rule_scorer_caps_at_one() -> None:
    f = _features(
        txn_count_1h=10, speed_from_last_kmh=9000, amount_zscore=9, is_new_device=1,
        is_new_merchant=1, merchant_risk_tier=2, ip_billing_mismatch=1,
    )  # fmt: skip
    assert asyncio.run(RuleScorer().score(f)).score == 1.0


def test_kserve_scorer_is_built_from_settings() -> None:
    from payments_api.scoring.kserve import KServeScorer

    settings = Settings(
        fraud_scorer="kserve", kserve_url="http://p/v2/models/m/infer", _env_file=None
    )
    scorer = build_scorer(settings)
    assert isinstance(scorer, KServeScorer)
    assert (scorer.url, scorer.ready_url) == (
        "http://p/v2/models/m/infer",
        "http://p/v2/models/m/ready",
    )
    assert scorer.timeout_s == settings.kserve_timeout_seconds


class BrokenScorer:
    name = "broken"
    model_version = "v0"

    async def score(self, features: Mapping[str, float]) -> ScoreResult:
        raise TimeoutError("model unavailable")

    async def ready(self) -> bool:
        return False

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_scorer_failure_falls_back_to_rules(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, make_txn: TxnFactory
) -> None:
    monkeypatch.setattr("payments_api.main.build_scorer", lambda _s: BrokenScorer())
    labels = {"reason": "TimeoutError"}
    before = REGISTRY.get_sample_value("fraud_scorer_fallback_total", labels) or 0.0

    with TestClient(create_app(settings)) as client:
        resp = client.post("/payments", json=make_txn())
        assert resp.status_code == 201
        assert resp.json()["scorer"] == "rule"
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["scorer"] == "fallback"

    assert REGISTRY.get_sample_value("fraud_scorer_fallback_total", labels) == before + 1
    assert scoring.build_scorer is build_scorer  # monkeypatch was scoped to main


def test_require_scorer_fails_readiness(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("payments_api.main.build_scorer", lambda _s: BrokenScorer())
    strict = settings.model_copy(update={"require_scorer": True})
    with TestClient(create_app(strict)) as client:
        ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["scorer"] == "unavailable"
