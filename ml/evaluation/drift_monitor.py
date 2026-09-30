"""Drift monitor: PSI of live feature vectors against the champion's training profile.

A long-running exporter (Phase 7). Every `--interval` it:

1. resolves `<MODEL_NAME>@champion` in MLflow and loads that version's
   `reference/feature_profile.json` (cached per version), read through the
   artifact proxy;
2. reads the `features` of payments scored in the last `--window` from the
   payments database;
3. computes PSI per feature (`ml.evaluation.drift`).

and serves the latest result on `/metrics`:

    fraud_feature_psi{feature}                 PSI per feature (absent below --min-rows)
    fraud_drift_window_rows                    payments in the window
    fraud_drift_reference_version              champion version the profile came from
    fraud_drift_up                             1 if the last computation succeeded
    fraud_drift_last_run_timestamp_seconds     when it last ran

    python -m ml.evaluation.drift_monitor --port 9102

Calendar features (`hour_*`, `dow_*`) are excluded by default (`--exclude`):
in a one-hour window their distribution reflects how fast the simulator
replays synthetic time, not a change in the population.

Reads DATABASE_URL (the API's `postgresql+asyncpg://` URL is accepted),
MLFLOW_TRACKING_URI and MODEL_NAME from the environment. Runs from the trainer
image as `deploy/base/drift-monitor`.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from ml.evaluation.drift import PROFILE_ARTIFACT, Profile, fetch_run_artifact, psi

log = logging.getLogger("ml.evaluation.drift_monitor")

CHAMPION = "champion"
DEFAULT_EXCLUDE = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")


class ReferenceMissingError(RuntimeError):
    """The champion was trained before reference profiles existed (backfill it)."""


@dataclass(frozen=True, slots=True)
class DriftState:
    up: bool = False
    last_run: float = 0.0
    window_rows: int | None = None
    reference_version: int | None = None
    psi: dict[str, float] = field(default_factory=dict)


def sync_database_url(url: str) -> str:
    """The API's asyncpg URL, rewritten for the synchronous psycopg driver."""
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


class ChampionReference:
    """The champion's reference profile, cached per registered version."""

    def __init__(self, client: Any, tracking_uri: str, model: str) -> None:
        self.client = client
        self.tracking_uri = tracking_uri
        self.model = model
        self._cache: dict[str, Profile] = {}

    def get(self) -> tuple[int, Profile]:
        mv = self.client.get_model_version_by_alias(self.model, CHAMPION)
        version = str(mv.version)
        if version not in self._cache:
            run = self.client.get_run(mv.run_id)
            try:
                data = fetch_run_artifact(
                    self.tracking_uri, run.info.artifact_uri, PROFILE_ARTIFACT
                )
            except Exception as exc:
                raise ReferenceMissingError(
                    f"{self.model} v{version} has no {PROFILE_ARTIFACT} ({exc}); "
                    f"backfill with `python -m ml.evaluation.drift profile --version {version}`"
                ) from exc
            self._cache[version] = Profile.from_dict(data)
            log.info("loaded reference profile for %s v%s", self.model, version)
        return int(version), self._cache[version]


def compute(
    reference: Callable[[], tuple[int, Profile]],
    live_features: Callable[[], list[dict[str, float]]],
    min_rows: int,
    now: Callable[[], float] = time.time,
    exclude: Sequence[str] = (),
) -> DriftState:
    """One drift evaluation. Never raises: failures come back as up=False."""
    rows: list[dict[str, float]] | None = None
    try:
        rows = live_features()
        version, profile = reference()
    except Exception as exc:
        log.error("drift computation failed: %s", exc)
        return DriftState(up=False, last_run=now(), window_rows=None if rows is None else len(rows))
    if len(rows) < min_rows:
        return DriftState(True, now(), len(rows), version)
    live = pd.DataFrame(rows).drop(columns=list(exclude), errors="ignore")
    return DriftState(True, now(), len(rows), version, psi(profile, live))


class DriftCollector(Collector):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = DriftState()

    def update(self, state: DriftState) -> None:
        with self._lock:
            self._state = state

    def collect(self) -> Iterable[Metric]:
        with self._lock:
            s = self._state
        up = GaugeMetricFamily("fraud_drift_up", "1 if the last drift computation succeeded")
        up.add_metric([], 1.0 if s.up else 0.0)
        yield up
        last = GaugeMetricFamily(
            "fraud_drift_last_run_timestamp_seconds", "When the drift monitor last ran"
        )
        last.add_metric([], s.last_run)
        yield last
        if s.window_rows is not None:
            rows = GaugeMetricFamily(
                "fraud_drift_window_rows", "Scored payments in the drift window"
            )
            rows.add_metric([], float(s.window_rows))
            yield rows
        if s.reference_version is not None:
            ref = GaugeMetricFamily(
                "fraud_drift_reference_version",
                "Champion version whose training profile is the drift reference",
            )
            ref.add_metric([], float(s.reference_version))
            yield ref
        if s.psi:
            fam = GaugeMetricFamily(
                "fraud_feature_psi",
                "Population Stability Index of a feature, live window vs training reference",
                labels=["feature"],
            )
            for name, value in sorted(s.psi.items()):
                fam.add_metric([name], value)
            yield fam


def live_features_query(
    database_url: str, window_seconds: float
) -> Callable[[], list[dict[str, float]]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(sync_database_url(database_url), pool_pre_ping=True)
    query = text(
        "SELECT features FROM payments WHERE created_at >= now() - make_interval(secs => :secs)"
    )

    def read() -> list[dict[str, float]]:
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(query, {"secs": window_seconds})]

    return read


def _duration(value: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600}
    return float(value[:-1]) * units[value[-1]] if value[-1] in units else float(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--interval", default="5m", help="e.g. 300s, 5m")
    parser.add_argument("--window", default="1h", help="live window, e.g. 30m, 1h")
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help="comma-separated features not to report (default: calendar features)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from mlflow import MlflowClient  # heavy; only the running monitor needs it

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model = os.environ.get("MODEL_NAME", "fraud-detector")
    reference = ChampionReference(MlflowClient(tracking_uri), tracking_uri, model)
    live = live_features_query(os.environ["DATABASE_URL"], _duration(args.window))
    interval = _duration(args.interval)
    exclude = [f.strip() for f in args.exclude.split(",") if f.strip()]

    collector = DriftCollector()
    REGISTRY.register(collector)
    start_http_server(args.port)
    log.info("serving :%d/metrics; window %s, every %s", args.port, args.window, args.interval)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        state = compute(reference.get, live, args.min_rows, exclude=exclude)
        collector.update(state)
        if state.psi:
            top = max(state.psi, key=lambda k: state.psi[k])
            log.info("rows=%s max PSI %s=%.3f", state.window_rows, top, state.psi[top])
        else:
            log.info("rows=%s up=%s (no PSI)", state.window_rows, state.up)
        stop.wait(interval)


if __name__ == "__main__":
    main()
