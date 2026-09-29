"""Batch featurisation for training. Same code path as the online API.

Walks each card's transactions in time order, maintaining a sliding window,
and calls the same `CardContext.from_history` and `build_features` that the
API calls with history loaded from Postgres.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from ml.data.schema import Transaction
from ml.features.build import build_features
from ml.features.context import LOOKBACK, CardContext, PastTransaction


def featurize(transactions: Iterable[Transaction]) -> list[dict[str, float]]:
    """Feature vectors in input order, each using only earlier history for its card."""
    txns = list(transactions)
    order = sorted(
        range(len(txns)),
        key=lambda i: (txns[i].card_token, txns[i].timestamp, txns[i].transaction_id),
    )
    out: list[dict[str, float]] = [{} for _ in txns]
    window: deque[PastTransaction] = deque()
    card: str | None = None
    for i in order:
        txn = txns[i]
        if txn.card_token != card:
            card = txn.card_token
            window.clear()
        cutoff = txn.timestamp - LOOKBACK
        while window and window[0].timestamp < cutoff:
            window.popleft()
        out[i] = build_features(txn, CardContext.from_history(window, as_of=txn.timestamp))
        window.append(PastTransaction.from_transaction(txn))
    return out
