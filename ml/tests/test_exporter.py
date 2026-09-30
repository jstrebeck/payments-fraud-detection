"""Registry exporter: MLflow state -> Prometheus gauges, with caching and failure handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from ml.evaluation.exporter import RegistryCollector, read_snapshot


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def get_registered_model(self, name: str) -> Any:
        self.calls += 1
        if self.fail:
            raise ConnectionError("mlflow down")
        return SimpleNamespace(aliases={"champion": "2"})

    def search_model_versions(self, filter_string: str, *, max_results: int) -> Any:
        return [
            SimpleNamespace(version="1", run_id="r1", creation_timestamp=1_000_000),
            SimpleNamespace(version="2", run_id="r2", creation_timestamp=2_000_000),
            SimpleNamespace(version="3", run_id=None, creation_timestamp=3_000_000),
        ]

    def get_run(self, run_id: str) -> Any:
        metrics = {"r1": {"pr_auc": 0.8, "best_iteration": 99.0}, "r2": {"pr_auc": 0.92}}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics[run_id]))

    def get_experiment_by_name(self, name: str) -> Any:
        return SimpleNamespace(experiment_id="5") if name == "fraud-detector" else None

    def search_runs(
        self, experiment_ids: list[str], *, order_by: list[str] | None, max_results: int
    ) -> Any:
        return [
            SimpleNamespace(
                info=SimpleNamespace(status="FINISHED", start_time=10_000, end_time=70_000)
            )
        ]


def scrape(collector: RegistryCollector) -> str:
    registry = CollectorRegistry()
    registry.register(collector)
    return generate_latest(registry).decode()


def test_snapshot_reads_aliases_metrics_and_runs() -> None:
    snap = read_snapshot(FakeClient(), "fraud-detector", ["fraud-detector", "missing"])
    assert snap.aliases == {"champion": 2}
    assert snap.version_metrics == {
        ("1", "pr_auc"): 0.8,
        ("2", "pr_auc"): 0.92,
    }  # only charted metrics
    assert snap.version_created == {"1": 1000.0, "2": 2000.0, "3": 3000.0}
    assert snap.last_runs == {"fraud-detector": ("FINISHED", 70.0)}


class AliasListClient(FakeClient):
    def get_registered_model(self, name: str) -> Any:
        return SimpleNamespace(aliases=[SimpleNamespace(alias="champion", version="7")])


def test_alias_objects_list_is_accepted() -> None:
    assert read_snapshot(AliasListClient(), "m", []).aliases == {"champion": 7}


def test_exposition() -> None:
    text = scrape(RegistryCollector(FakeClient(), "fraud-detector", ["fraud-detector"]))
    assert "fraud_registry_up 1.0" in text
    assert 'fraud_registry_alias_version{alias="champion"} 2.0' in text
    assert 'fraud_registry_version_metric{metric="pr_auc",version="2"} 0.92' in text
    assert 'fraud_training_last_run_success{experiment="fraud-detector"} 1.0' in text


def test_failure_reports_down_and_cache_limits_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    collector = RegistryCollector(client, "fraud-detector", [], cache_seconds=60)
    scrape(collector)
    scrape(collector)
    assert client.calls == 1  # second scrape served from cache

    client.fail = True
    collector.cache_seconds = 0
    text = scrape(collector)
    assert "fraud_registry_up 0.0" in text
    assert "fraud_registry_version_metric" not in text
