# ml.features

Pure functions from a transaction plus its card's recent history to a fixed
feature vector. The API and training call the same code (ADR-0010).

## Contract

```python
CardContext.from_history(items: Iterable[PastTransaction], as_of: datetime) -> CardContext
build_features(txn: Transaction, ctx: CardContext) -> dict[str, float]   # keys == FEATURE_NAMES, in order
featurize(txns: Iterable[Transaction]) -> list[dict[str, float]]         # ml.features.batch, needs the data extra
```

- `CardContext` holds the card's prior transactions: strictly before
  `as_of`, within `LOOKBACK` (7 days), at most `MAX_HISTORY` (100), oldest
  first. `from_history` is the only place these rules live.
- Online, the API loads candidate rows from Postgres and calls
  `from_history`. In batch, `featurize` keeps a sliding window per card and
  calls `from_history`. A test posts a generated stream through the API and
  asserts the stored features equal `featurize` output.
- The model itself takes this vector as input (ADR-0013); feature code is
  not serialised into it. Each registered version carries a
  `feature_version` tag, and scorers refuse versions that do not match
  `FEATURE_VERSION`.

## Features (`FEATURE_VERSION = "1"`)

| Family | Features |
|---|---|
| amount | `amount`, `log_amount`, `amount_zscore` (vs card history, std floored at $1) |
| velocity | `txn_count_{1h,24h,7d}`, `amount_sum_{1h,24h,7d}`, `seconds_since_last` |
| novelty | `is_new_merchant`, `is_new_category`, `is_new_device`, `is_new_ip_country` (1.0 when history is empty) |
| geo | `ip_billing_mismatch`, `km_from_last`, `speed_from_last_kmh` (country centroids) |
| time | `hour_{sin,cos}`, `dow_{sin,cos}` (UTC, cyclic) |
| merchant | `merchant_category_code`, `merchant_risk_tier`, `channel_code` |

## Rules

- No I/O, no database, no network. Inputs in, floats out.
- Adding or changing a feature bumps `FEATURE_VERSION`; note it in the model card.
- Never include the label or the fraud pattern.
