"""Static reference tables shared by the generator and the feature library.

Kept deliberately small and fixed: changing anything here changes generated
data and feature values, so bump `GENERATOR_VERSION` in `generator.py` and
note it in the model card when you do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Channel = Literal["card_present", "ecommerce", "recurring"]
CHANNELS: tuple[Channel, ...] = ("card_present", "ecommerce", "recurring")


@dataclass(frozen=True, slots=True)
class Country:
    code: str
    lat: float
    lon: float
    weight: float  # share of customers and merchants based here


# Approximate centroids; only used for distance between consecutive
# transactions, so country-level precision is enough.
COUNTRIES: dict[str, Country] = {
    c.code: c
    for c in (
        Country("US", 39.8, -98.6, 0.44),
        Country("CA", 56.1, -106.3, 0.07),
        Country("MX", 23.6, -102.5, 0.04),
        Country("BR", -14.2, -51.9, 0.04),
        Country("GB", 54.0, -2.0, 0.08),
        Country("DE", 51.2, 10.4, 0.07),
        Country("FR", 46.6, 2.2, 0.06),
        Country("ES", 40.4, -3.7, 0.04),
        Country("IT", 42.8, 12.5, 0.04),
        Country("NL", 52.1, 5.3, 0.03),
        Country("JP", 36.2, 138.3, 0.03),
        Country("AU", -25.3, 133.8, 0.03),
        Country("IN", 20.6, 78.9, 0.02),
        Country("SG", 1.35, 103.8, 0.01),
    )
}


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    risk_tier: int  # 0 low, 1 medium, 2 high
    median_amount: float  # USD
    amount_sigma: float  # log-normal sigma
    # Probability of each channel, in CHANNELS order. Sums to 1.
    channel_mix: tuple[float, float, float]


CATEGORIES: dict[str, Category] = {
    c.name: c
    for c in (
        Category("grocery", 0, 45.0, 0.6, (0.90, 0.10, 0.00)),
        Category("restaurants", 0, 30.0, 0.6, (0.85, 0.15, 0.00)),
        Category("fuel", 0, 50.0, 0.4, (1.00, 0.00, 0.00)),
        Category("pharmacy", 0, 25.0, 0.7, (0.90, 0.10, 0.00)),
        Category("transport", 0, 18.0, 0.8, (0.50, 0.50, 0.00)),
        Category("utilities", 0, 90.0, 0.4, (0.00, 0.30, 0.70)),
        Category("subscriptions", 0, 15.0, 0.5, (0.00, 0.20, 0.80)),
        Category("clothing", 1, 70.0, 0.8, (0.50, 0.50, 0.00)),
        Category("travel", 1, 350.0, 0.9, (0.20, 0.80, 0.00)),
        Category("entertainment", 1, 40.0, 0.7, (0.50, 0.50, 0.00)),
        Category("electronics", 2, 250.0, 1.0, (0.40, 0.60, 0.00)),
        Category("jewelry", 2, 400.0, 1.0, (0.70, 0.30, 0.00)),
        Category("gift_cards", 2, 100.0, 0.7, (0.20, 0.80, 0.00)),
        Category("digital_goods", 2, 20.0, 0.9, (0.00, 1.00, 0.00)),
    )
}

MerchantCategory = Literal[
    "grocery",
    "restaurants",
    "fuel",
    "pharmacy",
    "transport",
    "utilities",
    "subscriptions",
    "clothing",
    "travel",
    "entertainment",
    "electronics",
    "jewelry",
    "gift_cards",
    "digital_goods",
]

# Categories every customer uses; the rest are sampled per customer.
EVERYDAY_CATEGORIES: tuple[str, ...] = ("grocery", "restaurants", "fuel", "transport")
HIGH_RISK_CATEGORIES: tuple[str, ...] = tuple(
    name for name, c in CATEGORIES.items() if c.risk_tier == 2
)


def distance_km(a: str, b: str) -> float:
    """Great-circle distance between two country centroids; 0.0 if either is unknown."""
    ca, cb = COUNTRIES.get(a), COUNTRIES.get(b)
    if ca is None or cb is None:
        return 0.0
    lat1, lon1, lat2, lon2 = map(math.radians, (ca.lat, ca.lon, cb.lat, cb.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))
