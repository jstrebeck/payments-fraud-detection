"""The rolling per-card context that features are computed against.

Both the batch path (training) and the online path (the API) build a
`CardContext` through `CardContext.from_history`, so the window rules below
are applied identically in both places (ADR-0010).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ml.data.schema import Transaction

LOOKBACK = timedelta(days=7)
MAX_HISTORY = 100


@dataclass(frozen=True, slots=True)
class PastTransaction:
    """The subset of a prior transaction that features look at."""

    transaction_id: str
    timestamp: datetime
    amount: float
    merchant_id: str
    merchant_category: str
    channel: str
    device_id: str
    ip_country: str

    @classmethod
    def from_transaction(cls, txn: Transaction) -> PastTransaction:
        return cls(
            transaction_id=txn.transaction_id,
            timestamp=txn.timestamp,
            amount=txn.amount,
            merchant_id=txn.merchant_id,
            merchant_category=txn.merchant_category,
            channel=txn.channel,
            device_id=txn.device_id,
            ip_country=txn.ip_country,
        )


@dataclass(frozen=True, slots=True)
class CardContext:
    """Prior transactions on the same card, oldest first.

    Only transactions strictly before `as_of` and no older than `LOOKBACK`
    are kept, capped at the most recent `MAX_HISTORY`.
    """

    history: tuple[PastTransaction, ...] = ()

    @classmethod
    def from_history(cls, items: Iterable[PastTransaction], as_of: datetime) -> CardContext:
        window_start = as_of - LOOKBACK
        kept = sorted(
            (p for p in items if window_start <= p.timestamp < as_of),
            key=lambda p: (p.timestamp, p.transaction_id),
        )
        return cls(tuple(kept[-MAX_HISTORY:]))
