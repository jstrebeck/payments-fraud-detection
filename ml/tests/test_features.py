from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from ml.data.io import iter_transactions
from ml.data.schema import LabelledTransaction, Transaction
from ml.features import (
    FEATURE_NAMES,
    LOOKBACK,
    MAX_HISTORY,
    CardContext,
    PastTransaction,
    build_features,
)
from ml.features.batch import featurize

T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def txn(i: int = 0, **overrides: Any) -> Transaction:
    fields: dict[str, Any] = {
        "transaction_id": f"t{i:04d}",
        "timestamp": T0 + timedelta(minutes=i),
        "card_token": "tok_0123456789abcdef",
        "customer_id": "cus_1",
        "merchant_id": "mer_1",
        "merchant_category": "grocery",
        "amount": 40.0,
        "channel": "card_present",
        "device_id": "dev_1",
        "ip_country": "US",
        "billing_country": "US",
    }
    return Transaction.model_validate(fields | overrides)


def past(t: Transaction) -> PastTransaction:
    return PastTransaction.from_transaction(t)


def test_feature_vector_shape_and_types() -> None:
    features = build_features(txn(), CardContext())
    assert tuple(features) == FEATURE_NAMES
    assert all(isinstance(v, float) and math.isfinite(v) for v in features.values())


def test_first_transaction_on_a_card() -> None:
    f = build_features(txn(), CardContext())
    assert f["txn_count_7d"] == 0
    assert f["is_new_merchant"] == f["is_new_device"] == 1.0
    assert f["seconds_since_last"] == LOOKBACK.total_seconds()
    assert f["amount_zscore"] == 0.0
    assert f["speed_from_last_kmh"] == 0.0


def test_velocity_windows() -> None:
    history = [
        past(txn(1, timestamp=T0 - timedelta(days=3), amount=10)),
        past(txn(2, timestamp=T0 - timedelta(hours=5), amount=20)),
        past(txn(3, timestamp=T0 - timedelta(minutes=30), amount=30)),
    ]
    f = build_features(txn(), CardContext.from_history(history, as_of=T0))
    assert (f["txn_count_1h"], f["txn_count_24h"], f["txn_count_7d"]) == (1, 2, 3)
    assert (f["amount_sum_1h"], f["amount_sum_24h"], f["amount_sum_7d"]) == (30, 50, 60)
    assert f["seconds_since_last"] == 1800


def test_novelty_flags() -> None:
    history = [past(txn(1, timestamp=T0 - timedelta(hours=1)))]
    ctx = CardContext.from_history(history, as_of=T0)
    same = build_features(txn(), ctx)
    assert same["is_new_merchant"] == same["is_new_device"] == same["is_new_category"] == 0.0
    new = build_features(
        txn(merchant_id="mer_2", device_id="dev_2", merchant_category="jewelry", ip_country="GB"),
        ctx,
    )
    assert new["is_new_merchant"] == new["is_new_device"] == new["is_new_category"] == 1.0
    assert new["is_new_ip_country"] == new["ip_billing_mismatch"] == 1.0


def test_impossible_travel_speed() -> None:
    history = [past(txn(1, timestamp=T0 - timedelta(minutes=30), ip_country="US"))]
    f = build_features(txn(ip_country="JP"), CardContext.from_history(history, as_of=T0))
    assert f["km_from_last"] > 8000
    assert f["speed_from_last_kmh"] > 10_000


def test_amount_zscore() -> None:
    amounts = [20.0, 30.0, 40.0]
    history = [past(txn(i, timestamp=T0 - timedelta(hours=3 - i), amount=a))
               for i, a in enumerate(amounts)]  # fmt: skip
    ctx = CardContext.from_history(history, as_of=T0)
    f = build_features(txn(amount=30 + statistics.pstdev(amounts)), ctx)
    assert f["amount_zscore"] == pytest.approx(1.0)


def test_context_window_rules() -> None:
    items = [
        past(txn(1, timestamp=T0 - LOOKBACK - timedelta(seconds=1))),  # too old
        past(txn(2, timestamp=T0 - LOOKBACK)),  # boundary: kept
        past(txn(3, timestamp=T0)),  # same instant: not "prior"
        past(txn(4, timestamp=T0 + timedelta(seconds=1))),  # future
    ]
    ctx = CardContext.from_history(reversed(items), as_of=T0)
    assert [p.transaction_id for p in ctx.history] == ["t0002"]


def test_context_is_capped_and_ordered() -> None:
    items = [past(txn(i, timestamp=T0 - timedelta(minutes=i + 1))) for i in range(MAX_HISTORY + 20)]
    ctx = CardContext.from_history(items, as_of=T0)
    assert len(ctx.history) == MAX_HISTORY
    assert list(ctx.history) == sorted(ctx.history, key=lambda p: p.timestamp)
    assert ctx.history[-1].transaction_id == "t0000"  # most recent kept


def test_batch_matches_online_per_transaction(small_table: pa.Table) -> None:
    """The batch featuriser equals calling build_features with each card's full prior history."""
    rows = [r.unlabelled() for r in iter_transactions(small_table.slice(0, 3000))]
    batch = featurize(rows)
    by_card: dict[str, list[PastTransaction]] = {}
    for row, expected in zip(rows, batch, strict=True):
        seen = by_card.setdefault(row.card_token, [])
        online = build_features(row, CardContext.from_history(seen, as_of=row.timestamp))
        assert online == expected
        seen.append(past(row))


def _pattern_means(table: pa.Table, feature: str) -> dict[str, float]:
    rows: list[LabelledTransaction] = list(iter_transactions(table))
    features = featurize(rows)
    groups: dict[str, list[float]] = {}
    for r, f in zip(rows, features, strict=True):
        groups.setdefault(r.fraud_pattern or "legit", []).append(f[feature])
    return {k: statistics.fmean(v) for k, v in groups.items()}


@pytest.mark.parametrize(
    ("pattern", "feature", "factor"),
    [
        ("card_testing", "txn_count_1h", 10.0),
        ("impossible_travel", "speed_from_last_kmh", 2.0),
        ("high_value_new_merchant", "amount_zscore", 5.0),
        ("account_takeover", "is_new_category", 1.3),
    ],
)
def test_each_pattern_is_detectable_from_features(
    small_table: pa.Table, pattern: str, feature: str, factor: float
) -> None:
    """Each pattern shifts at least one feature well away from legit traffic, so it is
    learnable, without any feature seeing the label."""
    means = _pattern_means(small_table, feature)
    assert means[pattern] > factor * means["legit"]
