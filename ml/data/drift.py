"""Drift profiles for the generator (Phase 7).

A profile changes what "live traffic" looks like so that drift detection and
retraining can be exercised on purpose. `GeneratorConfig(drift="none")` (the
default) is the baseline and leaves the generator's output byte-for-byte
unchanged for a given seed.

Profiles:

`none`
    Baseline traffic.

`fraud-shift`
    Two things change at once, as they tend to in practice:

    1. **Population shift.** Prices rise (every amount x1.8) and card-present
       spend moves online (35% of legitimate card-present payments become
       e-commerce from the customer's home IP). Amount- and channel-based
       features move well past a PSI of 0.25.
    2. **A new fraud pattern, `session_hijack`,** is half of all fraud
       incidents: a hijacked session on the victim's own device and home IP
       places a few mid-sized e-commerce orders overnight (01:00-04:59 UTC)
       at merchants the customer has never used, in categories they do use.
       None of the baseline patterns look like this, so a model trained on
       baseline traffic scores it low; labelled examples of it make it
       learnable (night hours, new merchant, several orders in a short span).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ml.data.schema import FraudPattern

DriftName = Literal["none", "fraud-shift"]


@dataclass(frozen=True, slots=True)
class DriftProfile:
    name: str
    description: str
    # Multiplier on every amount (legit and fraud).
    amount_scale: float = 1.0
    # Share of legitimate card-present payments turned into e-commerce.
    card_present_to_ecommerce: float = 0.0
    # Extra fraud patterns and their share of fraud *incidents*; the baseline
    # patterns keep their relative weights in the remaining share.
    extra_patterns: dict[FraudPattern, float] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return self.name == "none"


DRIFT_PROFILES: dict[str, DriftProfile] = {
    p.name: p
    for p in (
        DriftProfile("none", "baseline traffic"),
        DriftProfile(
            "fraud-shift",
            "prices x2.5, card-present moves online, and a new session_hijack fraud pattern",
            amount_scale=2.5,
            card_present_to_ecommerce=0.6,
            extra_patterns={"session_hijack": 0.5},
        ),
    )
}


def get_profile(name: str) -> DriftProfile:
    try:
        return DRIFT_PROFILES[name]
    except KeyError:
        known = ", ".join(DRIFT_PROFILES)
        raise ValueError(f"unknown drift profile {name!r}; known: {known}") from None
