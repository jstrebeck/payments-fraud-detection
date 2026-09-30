# ml.data

Seeded synthetic generator and the canonical schema (ADR-0003).

| Module | Needs | Contents |
|---|---|---|
| `schema.py` | core | `Transaction` (what the API accepts), `LabelledTransaction` (+ `is_fraud`, `fraud_pattern`), `FRAUD_PATTERNS` (in every baseline dataset), `DRIFT_FRAUD_PATTERNS` (drift profiles only) |
| `drift.py` | core | `DriftProfile`, `DRIFT_PROFILES`, `get_profile`: traffic drift for exercising drift detection and retraining |
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

## Drift profiles (`drift.py`, Phase 7)

`GeneratorConfig(drift=...)` / `--drift` on the `ml.data` and simulator
CLIs. The default `none` is baseline traffic, byte-for-byte what the
generator produced before profiles existed (the Parquet footer only records
`drift` when it is not `none`). A profile applies its changes on a separate
random stream, so it never disturbs the baseline streams.

| Profile | Population shift | Fraud |
|---|---|---|
| `none` | none | baseline patterns |
| `fraud-shift` | every amount x2.5; 60% of legit card-present payments become e-commerce from the home IP | half of incidents are `session_hijack`: 3-5 delivery-style orders (pharmacy, restaurants, grocery online) over one afternoon, hours apart, at merchants new to the customer, on the victim's own device and home IP |

What `fraud-shift` does to a model trained on baseline data (champion
`fraud-detector` v2; 2,000 customers x 30 days per sample, measured
2026-09-30):

- PSI against baseline traffic (decile bins): `amount` and `log_amount`
  0.53, `channel_code` 0.52, `amount_sum_7d` 0.40, `amount_sum_24h` 0.23.
  Two baseline seeds differ by at most 0.002.
- Champion recall on fraud, baseline vs `fraud-shift`: 0.975 vs 0.488 at the
  review threshold (0.019), 0.940 vs 0.424 at the decline threshold (0.138).
  On `session_hijack` alone: 0.077 and 0.027.
- Learnable: a LightGBM trained on baseline history plus a small labelled
  `fraud-shift` slice reaches PR-AUC 0.54 on drifted traffic (champion 0.32),
  and 0.31 recall on `session_hijack` at 1% FPR (champion 0.01).

`ml/tests/test_drift.py` checks baseline equality, determinism, the price
and channel shift, and that at least three features pass PSI 0.25.

## CLI

```
uv run python -m ml.data generate --seed 42 --customers 5000 --days 90 --fraud-rate 0.015 --out data/transactions.parquet
uv run python -m ml.data stream   --seed 42 --customers 50 --days 2 --limit 10   # JSON lines, labels stripped
uv run python -m ml.data generate --seed 7 --drift fraud-shift --out data/drifted.parquet
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
