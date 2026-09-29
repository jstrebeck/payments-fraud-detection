# ml.data

Seeded synthetic generator and the canonical schema (ADR-0003).

| Module | Needs | Contents |
|---|---|---|
| `schema.py` | core | `Transaction` (what the API accepts), `LabelledTransaction` (+ `is_fraud`, `fraud_pattern`), `FRAUD_PATTERNS` |
| `reference.py` | core | Fixed countries (with centroids), merchant categories (risk tier, amount profile, channel mix), `distance_km` |
| `generator.py` | `data` | `GeneratorConfig`, `generate(config) -> pyarrow.Table`, `GENERATOR_VERSION` |
| `io.py` | `data` | `ARROW_SCHEMA`, `write_parquet`, `read_parquet`, `iter_transactions` |
| `cli.py` | `data` | `python -m ml.data generate|stream` |

## Schema

Fields as listed in `docs/architecture.md`. `Transaction` forbids extra
fields, so a request carrying `is_fraud` is rejected by the API. Cards are
opaque `tok_<16 hex>` tokens. Amounts are in USD (`currency` is always
`USD`). Timestamps must be timezone-aware and are normalised to UTC.

## Generator design

Three random streams are spawned from the seed (entities, legit behaviour,
fraud), so changing `fraud_rate` leaves legitimate traffic identical.

1. **Entities.** Customers with a home country, activity rate, spend scale,
   a set of categories (everyday ones plus 2 to 5 others), 1 to 2 devices,
   1 to 3 cards, and a few familiar merchants per category. Merchants have a
   category and a country; categories carry a risk tier.
2. **Legit behaviour.** Poisson transaction count per customer, weekday and
   hour-of-day seasonality, log-normal amounts per category, mostly familiar
   merchants, primary card and device. Noise that makes fraud non-trivial:
   occasional trips abroad (with no card-present spend during 3 to 12 hour
   flights), new phones (a quarter of customers), e-commerce from foreign
   IPs, and look-alikes of the fraud patterns: bursts of small in-app or
   transit charges, and big purchases at unfamiliar high-risk merchants,
   sometimes from a new device.
3. **Fraud patterns**, injected as whole incidents until the requested row
   rate is reached (weights in `PATTERN_WEIGHTS` balance rows per pattern):
   - `card_testing`: 3 to 12 small e-commerce charges a few minutes apart at
     one digital-goods or gift-card merchant, new device, foreign IP half the
     time
   - `impossible_travel`: card-present spend in a country >800 km away,
     30 minutes to 4 hours after a genuine card-present transaction
   - `high_value_new_merchant`: 2 to 6x a normal amount at an unfamiliar
     high-risk merchant, new device, usually a home IP
   - `account_takeover`: 3 to 8 escalating e-commerce charges, mostly in
     categories the customer does not use, from a new device or (half the
     time) the victim's own
4. **Drift profiles** (Phase 7): `amount_inflation`, `new_channel_mix`,
   `merchant_shift`; applied after generation, never labelled as fraud.

Changing output for an existing seed means bumping `GENERATOR_VERSION` and
updating the golden numbers in `ml/tests/test_generator.py`.

## CLI

```
uv run python -m ml.data generate --seed 42 --customers 5000 --days 90 --fraud-rate 0.015 --out data/transactions.parquet
uv run python -m ml.data stream   --seed 42 --customers 50 --days 2 --limit 10   # JSON lines, labels stripped
```

The defaults (5000 customers, 90 days) give about 740k rows in under 20 s.

Generator version 2 (Phase 2) added the look-alikes and softened the
patterns: version 1 was trivially separable (PR-AUC 0.99), which contradicts
ADR-0003 and leaves the promotion gate nothing to decide.
`make generate` wraps the first command.

## Tests (`ml/tests/test_generator.py`)

- Determinism: same config, byte-identical Parquet.
- Golden row and per-pattern counts for a fixed seed.
- Fraud rate within tolerance (one incident's worth of rows) of the request.
- Legit traffic unchanged by the fraud rate.
- Each pattern is detectable from features alone (`ml/tests/test_features.py`).
