"""Drift monitor: champion reference, windowed PSI, exposition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from ml.evaluation import drift_monitor
from ml.evaluation.drift import Profile, build_profile
from ml.evaluation.drift_monitor import (
    ChampionReference,
    DriftCollector,
    ReferenceMissingError,
    compute,
    sync_database_url,
)


def profile() -> Profile:
    rng = np.random.default_rng(0)
    return build_profile(pd.DataFrame({"amount": rng.lognormal(3, 1, 2000)}))


def rows(n: int, shift: float = 0.0, seed: int = 1) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{"amount": float(v)} for v in rng.lognormal(3 + shift, 1, n)]


def scrape(state: drift_monitor.DriftState) -> str:
    collector = DriftCollector()
    collector.update(state)
    registry = CollectorRegistry()
    registry.register(collector)
    return generate_latest(registry).decode()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql+asyncpg://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_sync_database_url(url: str, expected: str) -> None:
    assert sync_database_url(url) == expected


def test_compute_exports_psi_above_min_rows() -> None:
    ref = profile()
    stable = compute(lambda: (2, ref), lambda: rows(1000), min_rows=300, now=lambda: 5.0)
    assert stable.up
    assert stable.window_rows == 1000
    assert stable.reference_version == 2
    assert stable.psi["amount"] < 0.05
    drifted = compute(lambda: (2, ref), lambda: rows(1000, shift=1.5), min_rows=300)
    assert drifted.psi["amount"] > 0.25

    text = scrape(drifted)
    assert "fraud_drift_up 1.0" in text
    assert 'fraud_feature_psi{feature="amount"}' in text
    assert "fraud_drift_reference_version 2.0" in text
    assert "fraud_drift_window_rows 1000.0" in text


def test_too_few_rows_exports_no_psi() -> None:
    state = compute(lambda: (2, profile()), lambda: rows(10), min_rows=300)
    assert state.up
    assert state.psi == {}
    text = scrape(state)
    assert "fraud_feature_psi" not in text
    assert "fraud_drift_window_rows 10.0" in text


def test_failures_report_down_and_keep_row_count() -> None:
    def missing() -> tuple[int, Profile]:
        raise ReferenceMissingError("no profile")

    state = compute(missing, lambda: rows(500), min_rows=300)
    assert not state.up
    assert state.window_rows == 500
    assert "fraud_drift_up 0.0" in scrape(state)

    def db_down() -> list[dict[str, float]]:
        raise ConnectionError("db down")

    state = compute(lambda: (2, profile()), db_down, min_rows=300)
    assert not state.up
    assert state.window_rows is None


class FakeClient:
    def __init__(self) -> None:
        self.version = "2"

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        return SimpleNamespace(version=self.version, run_id=f"run{self.version}")

    def get_run(self, run_id: str) -> Any:
        return SimpleNamespace(
            info=SimpleNamespace(artifact_uri=f"mlflow-artifacts:/4/{run_id}/artifacts")
        )


def test_champion_reference_is_cached_per_version(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []
    data = profile().to_dict()

    def fake_fetch(tracking: str, artifact_uri: str, path: str) -> Any:
        fetched.append(artifact_uri)
        if "run3" in artifact_uri:
            raise OSError("404")
        return data

    monkeypatch.setattr(drift_monitor, "fetch_run_artifact", fake_fetch)
    client = FakeClient()
    ref = ChampionReference(client, "http://mlflow", "fraud-detector")
    assert ref.get()[0] == 2
    assert ref.get()[0] == 2
    assert len(fetched) == 1  # cached

    client.version = "3"  # promoted a model without a profile
    with pytest.raises(ReferenceMissingError, match="backfill"):
        ref.get()


def test_excluded_features_are_not_reported() -> None:
    ref = profile()
    live = [{"amount": r["amount"], "dow_sin": 0.5} for r in rows(500)]
    state = compute(lambda: (2, ref), lambda: live, min_rows=300, exclude=["amount"])
    assert state.psi == {}
