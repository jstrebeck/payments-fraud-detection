"""Seeded synthetic transaction generator (ADR-0003).

Three independent random streams are derived from the seed: entities,
legitimate behaviour and fraud. Changing the fraud rate therefore leaves the
legitimate traffic untouched, which keeps experiments comparable.

The generator works in integer microseconds since the epoch internally and
returns a PyArrow table sorted by (timestamp, transaction_id).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow as pa

from ml.data.io import ARROW_SCHEMA
from ml.data.reference import (
    CATEGORIES,
    CHANNELS,
    COUNTRIES,
    EVERYDAY_CATEGORIES,
    HIGH_RISK_CATEGORIES,
    distance_km,
)
from ml.data.schema import FraudPattern

# Bump when a change alters the output for an existing seed.
GENERATOR_VERSION = "2"

US_PER_SECOND = 1_000_000
US_PER_MINUTE = 60 * US_PER_SECOND
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR

# Monday..Sunday and 00..23 UTC. Relative weights, normalised at use.
WEEKDAY_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.05, 1.2, 1.3, 1.1])
HOUR_WEIGHTS = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 1.6, 1.6, 1.8,
     2.2, 2.0, 1.6, 1.5, 1.6, 1.9, 2.2, 2.1, 1.7, 1.2, 0.8, 0.4]
)  # fmt: skip

# Relative frequency of each fraud pattern among incidents. Incidents differ in
# size (a card-testing burst is ~12 rows, a high-value purchase 1-2), so these
# weights are chosen to give roughly balanced *row* counts per pattern.
PATTERN_WEIGHTS: dict[FraudPattern, float] = {
    "card_testing": 0.08,
    "impossible_travel": 0.37,
    "high_value_new_merchant": 0.40,
    "account_takeover": 0.15,
}

# Behavioural knobs for legitimate customers.
P_FAMILIAR_MERCHANT = 0.85
P_PRIMARY_DEVICE = 0.85
P_PRIMARY_CARD = 0.75
P_ECOMMERCE_FOREIGN_IP = 0.05  # VPNs, travel sites
P_TRAVELS = 0.06
P_NEW_DEVICE = 0.25
# Legitimate look-alikes of the fraud patterns, per customer per day. Without
# them the patterns are trivially separable (ADR-0003: learnable, not trivial).
BURSTS_PER_DAY = 0.02  # in-app purchases, transit taps: small charges in quick succession
BIG_PURCHASES_PER_DAY = 0.01  # a new laptop or a flight at an unfamiliar merchant
IMPOSSIBLE_TRAVEL_MIN_KM = 800.0


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    customers: int = 5000
    days: int = 90
    fraud_rate: float = 0.015
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    merchants: int | None = None  # default: max(100, customers // 5)

    def __post_init__(self) -> None:
        if self.customers < 1:
            raise ValueError("customers must be >= 1")
        if self.days < 1:
            raise ValueError("days must be >= 1")
        if not 0.0 <= self.fraud_rate < 0.5:
            raise ValueError("fraud_rate must be in [0, 0.5)")
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone-aware")

    @property
    def n_merchants(self) -> int:
        return self.merchants if self.merchants is not None else max(100, self.customers // 5)

    def metadata(self) -> dict[str, str]:
        """Deterministic key/values stored in the Parquet footer."""
        values: dict[str, Any] = asdict(self)
        values["start"] = self.start.astimezone(UTC).isoformat()
        values["generator_version"] = GENERATOR_VERSION
        return {k: str(v) for k, v in sorted(values.items())}


@dataclass(frozen=True, slots=True)
class Merchant:
    merchant_id: str
    category: str
    country: str


@dataclass(slots=True)
class Customer:
    customer_id: str
    home_country: str
    rate_per_day: float
    spend_scale: float
    categories: list[str]
    category_weights: np.ndarray[Any, np.dtype[np.float64]]
    devices: list[str]
    cards: list[str]
    familiar: dict[str, list[int]]  # category -> merchant indices
    row_indices: list[int] = field(default_factory=list)  # legit rows, filled later


class _Rows:
    """Column-oriented row buffer; cheaper than a list of dicts at 1M rows."""

    def __init__(self) -> None:
        self.cols: dict[str, list[Any]] = {name: [] for name in ARROW_SCHEMA.names}

    def __len__(self) -> int:
        return len(self.cols["transaction_id"])

    def add(
        self,
        *,
        ts_us: int,
        card_token: str,
        customer: Customer,
        merchant: Merchant,
        amount: float,
        channel: str,
        device_id: str,
        ip_country: str,
        fraud_pattern: str | None,
        txn_id: str,
    ) -> int:
        c = self.cols
        c["transaction_id"].append(txn_id)
        c["timestamp"].append(ts_us)
        c["card_token"].append(card_token)
        c["customer_id"].append(customer.customer_id)
        c["merchant_id"].append(merchant.merchant_id)
        c["merchant_category"].append(merchant.category)
        c["amount"].append(max(0.5, round(amount, 2)))
        c["currency"].append("USD")
        c["channel"].append(channel)
        c["device_id"].append(device_id)
        c["ip_country"].append(ip_country)
        c["billing_country"].append(customer.home_country)
        c["is_fraud"].append(fraud_pattern is not None)
        c["fraud_pattern"].append(fraud_pattern)
        return len(self) - 1


def _token(rng: np.random.Generator, prefix: str, n_bytes: int) -> str:
    return prefix + rng.bytes(n_bytes).hex()


def _txn_id(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16), version=4))


class _World:
    """Entities plus lookup tables built from the entity stream."""

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.country_codes = list(COUNTRIES)
        weights = np.array([COUNTRIES[c].weight for c in self.country_codes])
        self.country_p = weights / weights.sum()
        self.category_names = list(CATEGORIES)

        self.merchants: list[Merchant] = []
        self.by_category: dict[str, list[int]] = {c: [] for c in self.category_names}
        self.by_category_country: dict[tuple[str, str], list[int]] = {}
        m_countries = rng.choice(len(self.country_codes), size=config.n_merchants, p=self.country_p)
        m_categories = rng.integers(0, len(self.category_names), size=config.n_merchants)
        for i in range(config.n_merchants):
            m = Merchant(
                merchant_id=f"mer_{i:06d}",
                category=self.category_names[int(m_categories[i])],
                country=self.country_codes[int(m_countries[i])],
            )
            self.merchants.append(m)
            self.by_category[m.category].append(i)
            self.by_category_country.setdefault((m.category, m.country), []).append(i)
        # Guarantee every category has at least one merchant.
        for cat in self.category_names:
            if not self.by_category[cat]:
                i = len(self.merchants)
                m = Merchant(f"mer_{i:06d}", cat, self.country_codes[0])
                self.merchants.append(m)
                self.by_category[cat].append(i)
                self.by_category_country.setdefault((cat, m.country), []).append(i)

        optional = [c for c in self.category_names if c not in EVERYDAY_CATEGORIES]
        self.customers: list[Customer] = []
        for i in range(config.customers):
            home = self.country_codes[int(rng.choice(len(self.country_codes), p=self.country_p))]
            extra = rng.choice(len(optional), size=int(rng.integers(2, 6)), replace=False)
            cats = list(EVERYDAY_CATEGORIES) + [optional[int(j)] for j in sorted(extra)]
            familiar: dict[str, list[int]] = {}
            for cat in cats:
                pool = self.local_pool(cat, home)
                k = min(len(pool), int(rng.integers(1, 4)))
                familiar[cat] = [pool[int(j)] for j in rng.choice(len(pool), size=k, replace=False)]
            self.customers.append(
                Customer(
                    customer_id=f"cus_{i:07d}",
                    home_country=home,
                    rate_per_day=float(rng.gamma(2.0, 0.75)),
                    spend_scale=float(rng.lognormal(0.0, 0.35)),
                    categories=cats,
                    category_weights=rng.dirichlet(np.full(len(cats), 2.0)),
                    devices=[_token(rng, "dev_", 6) for _ in range(1 + int(rng.random() < 0.4))],
                    cards=[_token(rng, "tok_", 8) for _ in range(1 + int(rng.binomial(2, 0.2)))],
                    familiar=familiar,
                )
            )

    def local_pool(self, category: str, country: str) -> list[int]:
        return self.by_category_country.get((category, country)) or self.by_category[category]


def _generate_legit(
    world: _World, customer: Customer, rng: np.random.Generator, rows: _Rows
) -> None:
    cfg = world.config
    start_us = int(cfg.start.timestamp() * US_PER_SECOND)
    n = int(rng.poisson(customer.rate_per_day * cfg.days))
    if n == 0:
        return

    # Timestamps: weekday-weighted day, hour-of-day profile, uniform within the hour.
    first_weekday = cfg.start.weekday()
    day_w = WEEKDAY_WEIGHTS[(np.arange(cfg.days) + first_weekday) % 7]
    days = rng.choice(cfg.days, size=n, p=day_w / day_w.sum())
    hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    within = rng.integers(0, US_PER_HOUR, size=n)
    ts = np.sort(start_us + days * US_PER_DAY + hours * US_PER_HOUR + within)

    cat_idx = rng.choice(len(customer.categories), size=n, p=customer.category_weights)
    u_channel, u_familiar, u_pick, u_device, u_card, u_ip, u_ip_pick = rng.random((7, n))
    amount_noise = rng.standard_normal(n)

    # Optional life events: one trip abroad, one new phone.
    trip: tuple[int, int, str] | None = None
    if rng.random() < P_TRAVELS and cfg.days >= 5:
        t_start = start_us + int(rng.integers(1, cfg.days - 3)) * US_PER_DAY
        t_end = t_start + int(rng.integers(2, 9)) * US_PER_DAY
        others = [c for c in world.country_codes if c != customer.home_country]
        trip = (t_start, t_end, others[int(rng.integers(0, len(others)))])
        # No card-present spend while in transit; flights vary in length.
        transit_us = int(rng.integers(3, 13)) * US_PER_HOUR
    new_device_at: int | None = None
    if rng.random() < P_NEW_DEVICE:
        new_device_at = start_us + int(rng.integers(0, cfg.days)) * US_PER_DAY
        customer.devices.insert(0, _token(rng, "dev_", 6))

    for i in range(n):
        t = int(ts[i])
        cat = customer.categories[int(cat_idx[i])]
        spec = CATEGORIES[cat]
        channel = CHANNELS[int(np.searchsorted(np.cumsum(spec.channel_mix), u_channel[i]))]

        location = customer.home_country
        if trip is not None:
            t_start, t_end, dest = trip
            flying_out = t_start - transit_us <= t < t_start
            flying_back = t_end <= t < t_end + transit_us
            if (flying_out or flying_back) and channel == "card_present":
                continue  # in transit: no card-present spend
            if t_start <= t < t_end and channel != "recurring":
                location = dest

        if location != customer.home_country and channel == "card_present":
            pool = world.local_pool(cat, location)
            merchant_idx = pool[int(u_pick[i] * len(pool))]
        elif u_familiar[i] < P_FAMILIAR_MERCHANT:
            fam = customer.familiar[cat]
            merchant_idx = fam[int(u_pick[i] * len(fam))]
        else:
            pool = world.by_category[cat]
            merchant_idx = pool[int(u_pick[i] * len(pool))]
        merchant = world.merchants[merchant_idx]

        if channel == "card_present":
            ip_country = merchant.country
        elif channel == "ecommerce" and u_ip[i] < P_ECOMMERCE_FOREIGN_IP:
            ip_country = world.country_codes[int(u_ip_pick[i] * len(world.country_codes))]
        else:
            ip_country = location

        # Before the new-device event the newest device does not exist yet.
        devices = customer.devices
        if new_device_at is not None and t < new_device_at:
            devices = devices[1:]
        if len(devices) > 1 and u_device[i] >= P_PRIMARY_DEVICE:
            device = devices[1 + int((u_device[i] - P_PRIMARY_DEVICE) / (1 - P_PRIMARY_DEVICE)
                                     * (len(devices) - 1))]  # fmt: skip
        else:
            device = devices[0]

        cards = customer.cards
        if len(cards) > 1 and u_card[i] >= P_PRIMARY_CARD:
            card = cards[1 + int((u_card[i] - P_PRIMARY_CARD) / (1 - P_PRIMARY_CARD)
                                 * (len(cards) - 1))]  # fmt: skip
        else:
            card = cards[0]

        median = spec.median_amount * customer.spend_scale
        amount = float(np.exp(math.log(median) + spec.amount_sigma * amount_noise[i]))

        row = rows.add(
            ts_us=t,
            card_token=card,
            customer=customer,
            merchant=merchant,
            amount=amount,
            channel=channel,
            device_id=device,
            ip_country=ip_country,
            fraud_pattern=None,
            txn_id=_txn_id(rng),
        )
        customer.row_indices.append(row)

    _legit_lookalikes(world, customer, rng, rows, start_us)


def _legit_lookalikes(
    world: _World, customer: Customer, rng: np.random.Generator, rows: _Rows, start_us: int
) -> None:
    """Genuine behaviour that resembles fraud: bursts and big unfamiliar purchases."""
    cfg = world.config
    end_us = start_us + cfg.days * US_PER_DAY
    home = customer.home_country
    familiar = {m for ms in customer.familiar.values() for m in ms}

    def add(t: int, merchant: Merchant, amount: float, channel: str, device: str, ip: str) -> None:
        if t < end_us:
            customer.row_indices.append(rows.add(
                ts_us=t, card_token=customer.cards[0], customer=customer, merchant=merchant,
                amount=amount, channel=channel, device_id=device, ip_country=ip,
                fraud_pattern=None, txn_id=_txn_id(rng),
            ))  # fmt: skip

    for _ in range(int(rng.poisson(BURSTS_PER_DAY * cfg.days))):
        in_app = rng.random() < 0.5
        cat, channel = ("digital_goods", "ecommerce") if in_app else ("transport", "card_present")
        pool = world.local_pool(cat, home)
        merchant = world.merchants[pool[int(rng.integers(0, len(pool)))]]
        t = start_us + int(rng.integers(0, cfg.days * US_PER_DAY))
        for _ in range(int(rng.integers(3, 9))):
            t += int(rng.exponential(120.0) * US_PER_SECOND) + US_PER_SECOND
            ip = merchant.country if channel == "card_present" else home
            add(t, merchant, float(rng.uniform(0.99, 12.0)), channel, customer.devices[0], ip)

    for _ in range(int(rng.poisson(BIG_PURCHASES_PER_DAY * cfg.days))):
        cat = ("electronics", "jewelry", "travel", "gift_cards")[int(rng.integers(0, 4))]
        pool = [m for m in world.by_category[cat] if m not in familiar] or world.by_category[cat]
        merchant = world.merchants[pool[int(rng.integers(0, len(pool)))]]
        device = _token(rng, "dev_", 6) if rng.random() < 0.35 else customer.devices[0]
        ip = home if rng.random() < 0.9 else world.country_codes[
            int(rng.integers(0, len(world.country_codes)))]  # fmt: skip
        amount = CATEGORIES[cat].median_amount * customer.spend_scale * float(rng.uniform(2, 8))
        add(start_us + int(rng.integers(0, cfg.days * US_PER_DAY)), merchant, amount,
            "ecommerce", device, ip)  # fmt: skip


class _Fraud:
    """Fraud pattern injectors. Each adds labelled rows and returns how many."""

    def __init__(self, world: _World, rows: _Rows, rng: np.random.Generator) -> None:
        self.world = world
        self.rows = rows
        self.rng = rng
        cfg = world.config
        self.start_us = int(cfg.start.timestamp() * US_PER_SECOND)
        self.end_us = self.start_us + cfg.days * US_PER_DAY
        # Leave a day of history before and room for the burst after.
        margin = US_PER_DAY if cfg.days >= 3 else 0
        self.lo, self.hi = self.start_us + margin, self.end_us - margin

    def _time(self) -> int:
        return int(self.rng.integers(self.lo, max(self.lo + 1, self.hi)))

    def _card(self, customer: Customer) -> str:
        return customer.cards[int(self.rng.integers(0, len(customer.cards)))]

    def _merchant_idx(self, categories: list[str] | tuple[str, ...], country: str | None) -> int:
        cat = categories[int(self.rng.integers(0, len(categories)))]
        pool = self.world.local_pool(cat, country) if country else self.world.by_category[cat]
        return pool[int(self.rng.integers(0, len(pool)))]

    def _merchant_in(
        self, categories: list[str] | tuple[str, ...], country: str | None
    ) -> Merchant:
        return self.world.merchants[self._merchant_idx(categories, country)]

    def _foreign(self, home: str) -> str:
        others = [c for c in self.world.country_codes if c != home]
        return others[int(self.rng.integers(0, len(others)))]

    def _add(self, pattern: FraudPattern, customer: Customer, **kw: Any) -> None:
        if kw["ts_us"] >= self.end_us:
            return
        self.rows.add(customer=customer, fraud_pattern=pattern, txn_id=_txn_id(self.rng), **kw)

    def card_testing(self, customer: Customer) -> None:
        """Burst of small e-commerce charges at one digital merchant."""
        rng = self.rng
        merchant = self._merchant_in(("digital_goods", "gift_cards"), None)
        card, device = self._card(customer), _token(rng, "dev_", 6)
        ip = self._foreign(customer.home_country) if rng.random() < 0.5 else customer.home_country
        t = self._time()
        for _ in range(int(rng.integers(3, 13))):
            t += int(rng.exponential(180.0) * US_PER_SECOND) + US_PER_SECOND
            self._add(
                "card_testing", customer, ts_us=t, card_token=card, merchant=merchant,
                amount=float(rng.uniform(0.5, 15.0)), channel="ecommerce",
                device_id=device, ip_country=ip,
            )  # fmt: skip

    def impossible_travel(self, customer: Customer) -> None:
        """Card-present spend far away shortly after a genuine card-present one."""
        rng, cols = self.rng, self.rows.cols
        anchors = [
            r for r in customer.row_indices
            if cols["channel"][r] == "card_present" and self.lo <= cols["timestamp"][r] < self.hi
        ]  # fmt: skip
        if anchors:
            r = anchors[int(rng.integers(0, len(anchors)))]
            t, origin, card = cols["timestamp"][r], cols["ip_country"][r], cols["card_token"][r]
        else:
            t, origin, card = self._time(), customer.home_country, self._card(customer)
        far = [
            c for c in self.world.country_codes if distance_km(origin, c) > IMPOSSIBLE_TRAVEL_MIN_KM
        ]
        dest = far[int(rng.integers(0, len(far)))]
        device = _token(rng, "dev_", 6)
        t += int(rng.uniform(30, 240) * US_PER_MINUTE)
        for _ in range(int(rng.integers(1, 4))):
            merchant = self._merchant_in(
                ("electronics", "jewelry", "clothing", "restaurants"), dest
            )
            spec = CATEGORIES[merchant.category]
            self._add(
                "impossible_travel", customer, ts_us=t, card_token=card, merchant=merchant,
                amount=float(spec.median_amount * rng.lognormal(0.4, 0.5)),
                channel="card_present", device_id=device, ip_country=dest,
            )  # fmt: skip
            t += int(rng.uniform(5, 30) * US_PER_MINUTE)

    def high_value_new_merchant(self, customer: Customer) -> None:
        """One or two large e-commerce purchases at an unfamiliar high-risk merchant."""
        rng = self.rng
        familiar = {m for ms in customer.familiar.values() for m in ms}
        idx = self._merchant_idx(HIGH_RISK_CATEGORIES, None)
        for _ in range(10):  # a handful of retries; familiar high-risk merchants are rare
            if idx not in familiar:
                break
            idx = self._merchant_idx(HIGH_RISK_CATEGORIES, None)
        merchant = self.world.merchants[idx]
        card, device = self._card(customer), _token(rng, "dev_", 6)
        ip = customer.home_country if rng.random() < 0.7 else self._foreign(customer.home_country)
        t = self._time()
        base = CATEGORIES[merchant.category].median_amount * customer.spend_scale
        for _ in range(int(rng.integers(1, 3))):
            self._add(
                "high_value_new_merchant", customer, ts_us=t, card_token=card, merchant=merchant,
                amount=float(base * rng.uniform(2.0, 6.0)), channel="ecommerce",
                device_id=device, ip_country=ip,
            )  # fmt: skip
            t += int(rng.uniform(2, 40) * US_PER_MINUTE)

    def account_takeover(self, customer: Customer) -> None:
        """Escalating spend, often in unfamiliar categories, often from a new device."""
        rng = self.rng
        unusual = [c for c in CATEGORIES if c not in customer.categories] or list(CATEGORIES)
        card = self._card(customer)
        device = (
            _token(rng, "dev_", 6) if rng.random() < 0.5 else customer.devices[0]
        )  # malware on the victim's own device in the remaining cases
        ip = self._foreign(customer.home_country) if rng.random() < 0.4 else customer.home_country
        t = self._time()
        amount = float(rng.uniform(15, 40))
        for _ in range(int(rng.integers(3, 9))):
            cats = unusual if rng.random() < 0.6 else customer.categories
            merchant = self._merchant_in(cats, None)
            self._add(
                "account_takeover", customer, ts_us=t, card_token=card, merchant=merchant,
                amount=amount, channel="ecommerce", device_id=device, ip_country=ip,
            )  # fmt: skip
            amount *= float(rng.uniform(1.2, 1.6))
            t += int(rng.uniform(5, 60) * US_PER_MINUTE)


def generate(config: GeneratorConfig | None = None) -> pa.Table:
    """Generate a labelled transaction table. Deterministic for a given config."""
    config = config or GeneratorConfig()
    entity_ss, legit_ss, fraud_ss = np.random.SeedSequence(config.seed).spawn(3)
    world = _World(config, np.random.default_rng(entity_ss))

    rows = _Rows()
    legit_rng = np.random.default_rng(legit_ss)
    for customer in world.customers:
        _generate_legit(world, customer, legit_rng, rows)
    n_legit = len(rows)

    fraud_rng = np.random.default_rng(fraud_ss)
    fraud = _Fraud(world, rows, fraud_rng)
    target = round(config.fraud_rate / (1.0 - config.fraud_rate) * n_legit)
    active = [c for c in world.customers if c.row_indices] or world.customers
    patterns = list(PATTERN_WEIGHTS)
    p = np.array([PATTERN_WEIGHTS[k] for k in patterns])
    while len(rows) - n_legit < target:
        pattern = patterns[int(fraud_rng.choice(len(patterns), p=p / p.sum()))]
        customer = active[int(fraud_rng.integers(0, len(active)))]
        getattr(fraud, pattern)(customer)

    cols = rows.cols
    arrays = [
        pa.array(cols[f.name], type=pa.int64()).cast(f.type)
        if f.name == "timestamp"
        else pa.array(cols[f.name], type=f.type)
        for f in ARROW_SCHEMA
    ]
    table = pa.Table.from_arrays(arrays, schema=ARROW_SCHEMA)
    order = np.lexsort((np.array(cols["transaction_id"]), np.array(cols["timestamp"])))
    table = table.take(pa.array(order))
    return table.replace_schema_metadata(config.metadata())
