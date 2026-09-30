"""The canonical transaction schema.

This module is the only definition of a transaction (see `ml/README.md`).
`Transaction` is what the API accepts; `LabelledTransaction` adds the ground
truth that only the generator and training know about. The matching Arrow
schema for Parquet lives in `ml.data.io` so that importing this module does
not require pyarrow (the payments API image does not ship it).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ml.data.reference import COUNTRIES, Channel, MerchantCategory

FraudPattern = Literal[
    "card_testing",
    "impossible_travel",
    "high_value_new_merchant",
    "account_takeover",
    "session_hijack",  # only in drifted traffic (ml.data.drift, profile fraud-shift)
]
FRAUD_PATTERNS: tuple[FraudPattern, ...] = (
    "card_testing",
    "impossible_travel",
    "high_value_new_merchant",
    "account_takeover",
)
# Patterns that appear only under a drift profile (ml.data.drift), never in
# baseline data. FRAUD_PATTERNS is what every baseline dataset contains.
DRIFT_FRAUD_PATTERNS: tuple[FraudPattern, ...] = ("session_hijack",)


def _known_country(code: str) -> str:
    if code not in COUNTRIES:
        raise ValueError(f"unknown country code {code!r}")
    return code


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC)


CountryCode = Annotated[str, AfterValidator(_known_country)]
UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]


class Transaction(BaseModel):
    """A card payment as the payments API receives it. No label fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str = Field(min_length=1, max_length=64)
    timestamp: UtcDatetime
    # Opaque token, never card-number shaped (CLAUDE.md).
    card_token: str = Field(pattern=r"^tok_[0-9a-f]{16}$")
    customer_id: str = Field(min_length=1, max_length=32)
    merchant_id: str = Field(min_length=1, max_length=32)
    merchant_category: MerchantCategory
    amount: float = Field(gt=0, le=1_000_000)
    currency: Literal["USD"] = "USD"  # amounts are normalised to USD
    channel: Channel
    device_id: str = Field(min_length=1, max_length=32)
    ip_country: CountryCode
    billing_country: CountryCode


LABEL_FIELDS: frozenset[str] = frozenset({"is_fraud", "fraud_pattern"})


class LabelledTransaction(Transaction):
    """A transaction plus ground truth. Training and the simulator only."""

    is_fraud: bool
    fraud_pattern: FraudPattern | None = None

    def unlabelled(self) -> Transaction:
        """The same transaction with label fields stripped, ready to send to the API."""
        return Transaction.model_validate(self.model_dump(exclude=set(LABEL_FIELDS)))
