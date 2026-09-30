"""Prometheus exporter for the MLflow registry and training runs (Phase 6).

Grafana's training dashboard needs "metrics over versions", which live in
MLflow, not Prometheus. This process reads them on each scrape (cached for
`--cache-seconds`) and exposes them as gauges:

    fraud_registry_up                                   1 if the last MLflow read worked
    fraud_registry_alias_version{alias}                 version an alias points at
    fraud_registry_version_metric{version,metric}       test-window metrics of each version
    fraud_registry_version_created_timestamp_seconds{version}
    fraud_training_last_run_timestamp_seconds{experiment,status}   newest run per experiment
    fraud_training_last_run_success{experiment}         1 if that run FINISHED

    python -m ml.evaluation.exporter --port 9101

Reads MLFLOW_TRACKING_URI, MODEL_NAME (fraud-detector) and
TRAINING_EXPERIMENTS (fraud-detector,fraud-detector-dev) from the environment.
Runs from the trainer image as `deploy/base/registry-exporter`.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

log = logging.getLogger("ml.evaluation.exporter")

# Run metrics worth charting across versions (the training run logs more).
VERSION_METRICS = (
    "pr_auc",
    "roc_auc",
    "brier",
    "recall_at_fpr_0_5pct",
    "recall_at_fpr_1pct",
    "recall_at_fpr_2pct",
    "recall_account_takeover",
    "recall_card_testing",
    "recall_high_value_new_merchant",
    "recall_impossible_travel",
)
MAX_VERSIONS = 50


class RegistryClient(Protocol):
    """The slice of `mlflow.MlflowClient` the exporter uses (tests pass a fake)."""

    def get_registered_model(self, name: str) -> Any: ...
    def search_model_versions(self, filter_string: str, *, max_results: int) -> Any: ...
    def get_run(self, run_id: str) -> Any: ...
    def get_experiment_by_name(self, name: str) -> Any: ...
    def search_runs(
        self, experiment_ids: list[str], *, order_by: list[str] | None, max_results: int
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class Snapshot:
    aliases: dict[str, int] = field(default_factory=dict)
    version_metrics: dict[tuple[str, str], float] = field(default_factory=dict)
    version_created: dict[str, float] = field(default_factory=dict)
    last_runs: dict[str, tuple[str, float]] = field(default_factory=dict)  # exp -> (status, ts)


def read_snapshot(client: RegistryClient, model: str, experiments: Sequence[str]) -> Snapshot:
    """Everything the exporter publishes, in one pass over MLflow."""
    registered = client.get_registered_model(model)
    raw_aliases = registered.aliases or {}
    if isinstance(raw_aliases, dict):
        aliases = {a: int(v) for a, v in raw_aliases.items()}
    else:  # some client versions return a list of alias objects
        aliases = {a.alias: int(a.version) for a in raw_aliases}

    versions = client.search_model_versions(f"name='{model}'", max_results=MAX_VERSIONS)
    version_metrics: dict[tuple[str, str], float] = {}
    version_created: dict[str, float] = {}
    for mv in versions:
        version = str(mv.version)
        version_created[version] = float(mv.creation_timestamp) / 1000.0
        if not mv.run_id:
            continue
        metrics = client.get_run(mv.run_id).data.metrics
        for name in VERSION_METRICS:
            if name in metrics:
                version_metrics[(version, name)] = float(metrics[name])

    last_runs: dict[str, tuple[str, float]] = {}
    for name in experiments:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            continue
        runs = client.search_runs(
            [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
        )
        if runs:
            info = runs[0].info
            ts = (info.end_time or info.start_time) / 1000.0
            last_runs[name] = (str(info.status), float(ts))

    return Snapshot(aliases, version_metrics, version_created, last_runs)


def to_metrics(snapshot: Snapshot | None) -> Iterator[Metric]:
    up = GaugeMetricFamily("fraud_registry_up", "1 if the last MLflow registry read succeeded")
    up.add_metric([], 0.0 if snapshot is None else 1.0)
    yield up
    if snapshot is None:
        return

    alias = GaugeMetricFamily(
        "fraud_registry_alias_version", "Model version an alias points at", labels=["alias"]
    )
    for alias_name, alias_version in sorted(snapshot.aliases.items()):
        alias.add_metric([alias_name], float(alias_version))
    yield alias

    metric = GaugeMetricFamily(
        "fraud_registry_version_metric",
        "Test-window evaluation metric of a registered version's training run",
        labels=["version", "metric"],
    )
    for (version, name), value in sorted(snapshot.version_metrics.items()):
        metric.add_metric([version, name], value)
    yield metric

    created = GaugeMetricFamily(
        "fraud_registry_version_created_timestamp_seconds",
        "When a version was registered",
        labels=["version"],
    )
    for version, ts in sorted(snapshot.version_created.items()):
        created.add_metric([version], ts)
    yield created

    last = GaugeMetricFamily(
        "fraud_training_last_run_timestamp_seconds",
        "End (or start, if unfinished) of the newest run in an experiment",
        labels=["experiment", "status"],
    )
    success = GaugeMetricFamily(
        "fraud_training_last_run_success",
        "1 if the newest run in an experiment finished successfully",
        labels=["experiment"],
    )
    for exp, (status, ts) in sorted(snapshot.last_runs.items()):
        last.add_metric([exp, status], ts)
        success.add_metric([exp], 1.0 if status == "FINISHED" else 0.0)
    yield last
    yield success


class RegistryCollector(Collector):
    """Reads MLflow at most once per `cache_seconds`; a failed read reports up=0."""

    def __init__(
        self,
        client: RegistryClient,
        model: str,
        experiments: Sequence[str],
        cache_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.model = model
        self.experiments = list(experiments)
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        self._read_at = float("-inf")

    def collect(self) -> Iterable[Metric]:
        with self._lock:
            if time.monotonic() - self._read_at >= self.cache_seconds:
                try:
                    self._snapshot = read_snapshot(self.client, self.model, self.experiments)
                except Exception:
                    log.exception("mlflow read failed")
                    self._snapshot = None
                self._read_at = time.monotonic()
            snapshot = self._snapshot
        return list(to_metrics(snapshot))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--cache-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from mlflow import MlflowClient  # heavy; only the running exporter needs it

    model = os.environ.get("MODEL_NAME", "fraud-detector")
    default_experiments = "fraud-detector,fraud-detector-dev"
    experiments = [
        e.strip()
        for e in os.environ.get("TRAINING_EXPERIMENTS", default_experiments).split(",")
        if e.strip()
    ]
    REGISTRY.register(RegistryCollector(MlflowClient(), model, experiments, args.cache_seconds))
    start_http_server(args.port)
    log.info("serving :%d/metrics for %s (experiments %s)", args.port, model, experiments)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()


if __name__ == "__main__":
    main()
