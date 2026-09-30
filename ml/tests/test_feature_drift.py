"""Reference profiles and PSI (ml.evaluation.drift)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.drift import (
    EPSILON,
    MAX_CATEGORIES,
    Profile,
    _generator_config_from_params,
    build_feature,
    build_profile,
    psi,
    psi_feature,
)


def frame(seed: int = 0, n: int = 5000, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.lognormal(3.5 + shift, 1.0, n),
            "is_new_device": (rng.random(n) < 0.1 + shift / 4).astype("float64"),
            "channel_code": rng.integers(0, 3, n).astype("float64"),
            "mostly_zero": np.where(rng.random(n) < 0.8, 0.0, rng.exponential(5, n)),
        }
    )


def test_kinds_and_bins() -> None:
    profile = build_profile(frame())
    amount = profile.features["amount"]
    assert amount.kind == "continuous"
    assert len(amount.points) == 9
    assert len(amount.expected) == 10
    assert all(abs(e - 0.1) < 0.01 for e in amount.expected)
    flag = profile.features["is_new_device"]
    assert flag.kind == "categorical"
    assert flag.points == [0.0, 1.0]
    assert len(flag.expected) == 3
    assert flag.expected[-1] == 0.0  # `other` bin
    # Deduplicated edges: 80% zeros collapse the lower deciles into one edge.
    zero = profile.features["mostly_zero"]
    assert zero.kind == "continuous"
    assert zero.points[0] == 0.0
    assert len(zero.points) < 9
    assert zero.expected[0] == pytest.approx(0.8, abs=0.02)
    for fp in profile.features.values():
        assert sum(fp.expected) == pytest.approx(1.0)


def test_deterministic_and_round_trips() -> None:
    a, b = build_profile(frame()), build_profile(frame())
    assert a == b
    assert Profile.from_dict(json.loads(json.dumps(a.to_dict()))) == a


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="format"):
        Profile.from_dict({"format": 99, "n": 1, "features": {}})


def test_same_distribution_is_stable_and_shift_is_detected() -> None:
    profile = build_profile(frame(seed=0))
    same = psi(profile, frame(seed=1))
    assert max(same.values()) < 0.02
    shifted = psi(profile, frame(seed=1, shift=1.0))
    assert shifted["amount"] > 0.25
    assert shifted["is_new_device"] > 0.1
    assert shifted["channel_code"] < 0.02  # untouched feature stays quiet


def test_unseen_category_lands_in_other_with_epsilon_smoothing() -> None:
    fp = build_feature(pd.Series([0.0, 1.0] * 50))
    live = pd.Series([2.0] * 100)  # never seen in training
    # Reference [0.5, 0.5, other 0] vs live [0, 0, other 1], empty shares floored at EPSILON.
    expected = (1.0 - EPSILON) * np.log(1.0 / EPSILON) + 2 * (EPSILON - 0.5) * np.log(EPSILON / 0.5)
    assert psi_feature(fp, live) == pytest.approx(expected, rel=1e-6)


def test_identical_distribution_has_zero_psi() -> None:
    values = pd.Series(np.arange(1000, dtype="float64"))
    assert psi_feature(build_feature(values), values) == pytest.approx(0.0, abs=1e-12)


def test_out_of_range_values_use_the_open_ended_bins() -> None:
    fp = build_feature(pd.Series(np.arange(100, dtype="float64")))
    assert fp.kind == "continuous"
    # Everything far above the training range lands in the last bin, which held 10%.
    value = psi_feature(fp, pd.Series([1e6] * 50))
    expected = (1.0 - 0.1) * np.log(1.0 / 0.1) + 9 * (EPSILON - 0.1) * np.log(EPSILON / 0.1)
    assert value == pytest.approx(expected, rel=1e-3)


def test_many_distinct_values_are_continuous() -> None:
    fp = build_feature(pd.Series(np.arange(MAX_CATEGORIES + 1, dtype="float64")))
    assert fp.kind == "continuous"
    few = build_feature(pd.Series(np.arange(MAX_CATEGORIES, dtype="float64")))
    assert few.kind == "categorical"


def test_non_finite_values_are_ignored_and_missing_features_skipped() -> None:
    profile = build_profile(frame())
    live = frame(seed=1).drop(columns=["mostly_zero"])
    live.loc[:10, "amount"] = np.nan
    result = psi(profile, live)
    assert "mostly_zero" not in result
    assert np.isfinite(result["amount"])


def test_generator_config_from_logged_params() -> None:
    params = {
        "data.seed": "42",
        "data.customers": "5000",
        "data.days": "90",
        "data.fraud_rate": "0.015",
        "data.start": "2026-01-01T00:00:00+00:00",
        "data.merchants": "None",
        "data.generator_version": "2",
        "n_train": "123",
    }
    config = _generator_config_from_params(params)
    assert (config.seed, config.customers, config.days) == (42, 5000, 90)
    assert config.merchants is None
