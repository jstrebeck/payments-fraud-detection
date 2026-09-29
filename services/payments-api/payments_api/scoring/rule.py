"""Hand-written rules. The Phase 1 scorer and the permanent fallback."""

from __future__ import annotations

from collections.abc import Mapping

from payments_api.scoring.base import ScoreResult

RULES_VERSION = "rules-v1"


class RuleScorer:
    """Additive points per red flag, capped at 1.0.

    Each rule targets one of the generator's fraud patterns. Deliberately
    simple: the point is to have a baseline the trained model must beat.
    """

    name = "rule"
    model_version = RULES_VERSION

    async def score(self, features: Mapping[str, float]) -> ScoreResult:
        f = features
        points = 0.0
        if f["txn_count_1h"] >= 4:  # card testing: many charges in a short window
            points += 0.5
        if f["speed_from_last_kmh"] > 1000 and f["channel_code"] == 0:  # impossible travel
            points += 0.5
        if f["amount_zscore"] > 4:  # unusually large for this card
            points += 0.3
        if f["is_new_device"] and f["merchant_risk_tier"] >= 2:
            points += 0.3
        if f["is_new_merchant"] and f["merchant_risk_tier"] >= 2:
            points += 0.1
        if f["ip_billing_mismatch"]:
            points += 0.1
        return ScoreResult(score=min(points, 1.0), scorer=self.name, model_version=RULES_VERSION)

    async def ready(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None
