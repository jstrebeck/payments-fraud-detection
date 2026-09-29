"""MlflowScorer against a throwaway SQLite registry with a model from the real trainer."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from ml.data.generator import GeneratorConfig, generate
from ml.data.io import write_parquet
from ml.training.config import TrainConfig, TrainSettings
from ml.training.train import train
from payments_api.config import Settings
from payments_api.main import create_app

TxnFactory = Callable[..., dict[str, Any]]
MODEL = "fraud-api-test"
EXPERIMENT = "api-tests"


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A registry with two trained versions; v1 is the champion."""
    tmp = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", uri)
        mp.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp / "artifacts").as_uri())
        data = tmp / "data.parquet"
        write_parquet(generate(GeneratorConfig(seed=5, customers=300, days=21)), data)
        base = TrainConfig.load()
        config = base.model_copy(update={"lightgbm": base.lightgbm | {"n_estimators": 50}})
        settings = TrainSettings(
            data_uri=data, mlflow_experiment_name=EXPERIMENT, model_name=MODEL,
            git_sha="test", _env_file=None,
        )  # fmt: skip
        for _ in range(2):
            train(settings, config)
        MlflowClient().set_registered_model_alias(MODEL, "champion", "1")
        yield uri


def _settings(settings: Settings, registry: str, **kw: Any) -> Settings:
    return settings.model_copy(
        update={
            "fraud_scorer": "mlflow",
            "mlflow_tracking_uri": registry,
            "model_name": MODEL,
            "model_refresh_seconds": 0.2,
        }
        | kw
    )


def test_scores_with_champion_and_hot_reloads(
    settings: Settings, registry: str, make_txn: TxnFactory
) -> None:
    with TestClient(create_app(_settings(settings, registry))) as client:
        assert client.get("/readyz").json()["scorer"] == "ok"
        body = client.post("/payments", json=make_txn()).json()
        assert body["scorer"] == "mlflow"
        assert body["model_version"] == f"{MODEL}/1"
        assert 0.0 <= body["score"] <= 1.0
        assert f'fraud_model_version_info{{scorer="mlflow",version="{MODEL}/1"}} 1.0' in (
            client.get("/metrics").text
        )

        # Promotion moves the alias; the API picks it up without a restart.
        MlflowClient(registry).set_registered_model_alias(MODEL, "champion", "2")
        deadline = time.monotonic() + 10
        version = body["model_version"]
        i = 0
        while version != f"{MODEL}/2" and time.monotonic() < deadline:
            time.sleep(0.2)
            i += 1
            version = client.post("/payments", json=make_txn(transaction_id=f"t-{i}")).json()[
                "model_version"
            ]
        assert version == f"{MODEL}/2"
    MlflowClient(registry).set_registered_model_alias(MODEL, "champion", "1")


def test_incompatible_model_is_refused_and_rules_take_over(
    settings: Settings, registry: str, make_txn: TxnFactory
) -> None:
    client_ = MlflowClient(registry)
    client_.set_model_version_tag(MODEL, "1", "feature_version", "999")
    try:
        with TestClient(create_app(_settings(settings, registry))) as client:
            assert client.get("/readyz").json()["scorer"] == "fallback"
            body = client.post("/payments", json=make_txn()).json()
            assert body["scorer"] == "rule"
    finally:
        client_.set_model_version_tag(MODEL, "1", "feature_version", "1")


def test_missing_registry_falls_back(
    settings: Settings, make_txn: TxnFactory, tmp_path: Path
) -> None:
    empty = f"sqlite:///{tmp_path / 'empty.db'}"
    with TestClient(create_app(_settings(settings, empty, model_refresh_seconds=0))) as client:
        assert client.post("/payments", json=make_txn()).json()["scorer"] == "rule"
