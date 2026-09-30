"""PaymentService decides with the scoring model's thresholds when it has them."""

from __future__ import annotations

from typing import Any

from payments_api.policy import DecisionPolicy
from payments_api.scoring import RuleScorer, ScoreResult
from payments_api.service import PaymentService

CONFIGURED = DecisionPolicy(0.5, 0.8)


def _service() -> PaymentService:
    sessions: Any = None  # not used by _policy_for
    return PaymentService(sessions, RuleScorer(), CONFIGURED)


def test_model_thresholds_win_over_configured_policy() -> None:
    result = ScoreResult(0.05, "kserve", "fraud-detector/3", 0.02, 0.14)
    policy = _service()._policy_for(result)
    assert policy == DecisionPolicy(0.02, 0.14)
    assert policy.decide(0.05) == "review"


def test_configured_policy_covers_rules_and_untagged_models() -> None:
    for result in (
        ScoreResult(0.3, "rule", "rules-v1"),
        ScoreResult(0.3, "kserve", "fraud-detector/2"),
    ):
        assert _service()._policy_for(result) is CONFIGURED
