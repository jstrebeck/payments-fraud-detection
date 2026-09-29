"""Transaction + card context -> feature vector. Pure functions, no I/O."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import timedelta

from ml.data.reference import CATEGORIES, CHANNELS, distance_km
from ml.data.schema import Transaction
from ml.features.context import LOOKBACK, CardContext

# Bump when a feature is added, removed or changes meaning; recorded with the model.
FEATURE_VERSION = "1"

FEATURE_NAMES: tuple[str, ...] = (
    # amount
    "amount",
    "log_amount",
    "amount_zscore",
    # velocity (prior transactions on this card, current one excluded)
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "amount_sum_7d",
    "seconds_since_last",
    # novelty against the card's recent history
    "is_new_merchant",
    "is_new_category",
    "is_new_device",
    "is_new_ip_country",
    # geography
    "ip_billing_mismatch",
    "km_from_last",
    "speed_from_last_kmh",
    # time (UTC)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # merchant and channel
    "merchant_category_code",
    "merchant_risk_tier",
    "channel_code",
)

_CATEGORY_CODES = {name: i for i, name in enumerate(CATEGORIES)}
_CHANNEL_CODES = {name: i for i, name in enumerate(CHANNELS)}
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_MIN_STD = 1.0  # USD; avoids huge z-scores for cards with near-constant spend
_MIN_HOURS = 1.0 / 60.0  # speed is computed over at least one minute


def build_features(txn: Transaction, ctx: CardContext) -> dict[str, float]:
    """Compute the feature vector for `txn` given its card's prior history."""
    history = ctx.history
    ts = txn.timestamp
    amounts = [p.amount for p in history]

    if len(amounts) >= 2:
        mean = math.fsum(amounts) / len(amounts)
        std = math.sqrt(math.fsum((a - mean) ** 2 for a in amounts) / len(amounts))
        zscore = (txn.amount - mean) / max(std, _MIN_STD)
    else:
        zscore = 0.0

    last_1h = [p.amount for p in history if ts - p.timestamp <= _HOUR]
    last_24h = [p.amount for p in history if ts - p.timestamp <= _DAY]

    if history:
        last = history[-1]
        seconds_since_last = (ts - last.timestamp).total_seconds()
        km = distance_km(last.ip_country, txn.ip_country)
        speed = km / max(seconds_since_last / 3600.0, _MIN_HOURS)
    else:
        seconds_since_last = LOOKBACK.total_seconds()
        km = speed = 0.0

    hour = ts.hour + ts.minute / 60.0
    dow = ts.weekday() + hour / 24.0

    features = {
        "amount": txn.amount,
        "log_amount": math.log1p(txn.amount),
        "amount_zscore": zscore,
        "txn_count_1h": float(len(last_1h)),
        "txn_count_24h": float(len(last_24h)),
        "txn_count_7d": float(len(history)),
        "amount_sum_1h": math.fsum(last_1h),
        "amount_sum_24h": math.fsum(last_24h),
        "amount_sum_7d": math.fsum(amounts),
        "seconds_since_last": seconds_since_last,
        "is_new_merchant": _is_new(txn.merchant_id, (p.merchant_id for p in history)),
        "is_new_category": _is_new(txn.merchant_category, (p.merchant_category for p in history)),
        "is_new_device": _is_new(txn.device_id, (p.device_id for p in history)),
        "is_new_ip_country": _is_new(txn.ip_country, (p.ip_country for p in history)),
        "ip_billing_mismatch": float(txn.ip_country != txn.billing_country),
        "km_from_last": km,
        "speed_from_last_kmh": speed,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2 * math.pi * dow / 7.0),
        "merchant_category_code": float(_CATEGORY_CODES[txn.merchant_category]),
        "merchant_risk_tier": float(CATEGORIES[txn.merchant_category].risk_tier),
        "channel_code": float(_CHANNEL_CODES[txn.channel]),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def _is_new(value: str, seen: Iterable[str]) -> float:
    return 0.0 if value in seen else 1.0
