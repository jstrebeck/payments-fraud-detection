"""Drift profiles: baseline untouched, fraud-shift measurably different and deterministic."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ml.data.drift import DRIFT_PROFILES, get_profile
from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions
from ml.data.schema import DRIFT_FRAUD_PATTERNS, FRAUD_PATTERNS
from ml.features import FEATURE_NAMES
from ml.features.batch import featurize
from ml.tests.conftest import SMALL


@pytest.fixture(scope="module")
def drifted() -> pa.Table:
    return generate(replace(SMALL, drift="fraud-shift"))


def _features(table: pa.Table) -> pd.DataFrame:
    rows = [r.unlabelled() for r in iter_transactions(table)]
    return pd.DataFrame(featurize(rows))[list(FEATURE_NAMES)].astype("float64")


def _psi(ref: pd.Series, live: pd.Series, bins: int = 10) -> float:
    """Decile PSI; value shares for near-constant features."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        values = np.union1d(ref.unique(), live.unique())
        r = np.array([(ref == v).mean() for v in values])
        lv = np.array([(live == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        lv = np.histogram(live, edges)[0] / len(live)
    r, lv = np.clip(r, 1e-4, None), np.clip(lv, 1e-4, None)
    return float(np.sum((lv - r) * np.log(lv / r)))


def test_baseline_is_the_default_and_unchanged(small_table: pa.Table) -> None:
    assert GeneratorConfig().drift == "none"
    assert generate(replace(SMALL, drift="none")).equals(small_table)
    assert b"drift" not in small_table.schema.metadata  # footer predates drift profiles


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown drift profile"):
        GeneratorConfig(drift="nope")
    assert set(DRIFT_PROFILES) >= {"none", "fraud-shift"}


def test_fraud_shift_is_deterministic_and_labelled(drifted: pa.Table) -> None:
    assert generate(replace(SMALL, drift="fraud-shift")).equals(drifted)
    assert drifted.schema.metadata[b"drift"] == b"fraud-shift"
    patterns = Counter(drifted.column("fraud_pattern").drop_null().to_pylist())
    assert set(patterns) == set(FRAUD_PATTERNS) | set(DRIFT_FRAUD_PATTERNS)
    # About half of fraud incidents; incidents are 3-5 orders, so more than half of rows.
    assert patterns["session_hijack"] > 0.3 * sum(patterns.values())


def test_fraud_shift_moves_prices_and_channels(small_table: pa.Table, drifted: pa.Table) -> None:
    profile = get_profile("fraud-shift")
    base_legit = small_table.filter(pc.invert(small_table.column("is_fraud")))
    legit = drifted.filter(pc.invert(drifted.column("is_fraud")))
    ratio = float(np.median(legit.column("amount").to_numpy())) / float(
        np.median(base_legit.column("amount").to_numpy())
    )
    assert ratio == pytest.approx(profile.amount_scale, rel=0.1)

    def card_present_share(t: pa.Table) -> float:
        return float(np.mean(np.asarray(t.column("channel").to_pylist()) == "card_present"))

    assert card_present_share(legit) < 0.6 * card_present_share(base_legit)


def test_fraud_shift_features_drift_past_psi_0_25(small_table: pa.Table, drifted: pa.Table) -> None:
    ref, live = _features(small_table), _features(drifted)
    other_seed = _features(generate(replace(SMALL, seed=SMALL.seed + 1)))
    drifted_psi = {c: _psi(ref[c], live[c]) for c in FEATURE_NAMES}
    baseline_psi = {c: _psi(ref[c], other_seed[c]) for c in FEATURE_NAMES}
    assert sum(v > 0.25 for v in drifted_psi.values()) >= 3, drifted_psi
    assert {"amount", "channel_code"} <= {c for c, v in drifted_psi.items() if v > 0.25}
    assert max(baseline_psi.values()) < 0.1, baseline_psi  # a new seed alone is not drift
