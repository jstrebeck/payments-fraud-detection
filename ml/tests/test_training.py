"""End-to-end trainer and gate against a throwaway SQLite MLflow registry."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow import MlflowClient

from ml.data.generator import GeneratorConfig, generate
from ml.data.io import write_parquet
from ml.evaluation.drift import PROFILE_ARTIFACT, Profile
from ml.evaluation.promote import CARD, CHAMPION, GATE, PREVIOUS, run_gate
from ml.features import FEATURE_NAMES, FEATURE_VERSION
from ml.training import train as train_module
from ml.training.config import TrainConfig, TrainSettings
from ml.training.train import TrainResult, train

MODEL = "fraud-test"
EXPERIMENT = "training-tests"


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    tmp = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", uri)
        mp.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp / "artifacts").as_uri())
        write_parquet(
            generate(GeneratorConfig(seed=11, customers=400, days=30, fraud_rate=0.03)),
            tmp / "data.parquet",
        )
        yield tmp


@pytest.fixture(scope="module")
def config() -> TrainConfig:
    base = TrainConfig.load()
    return base.model_copy(update={"lightgbm": base.lightgbm | {"n_estimators": 200}})


def _settings(registry: Path, **kw: Any) -> TrainSettings:
    return TrainSettings(
        data_uri=registry / "data.parquet",
        mlflow_experiment_name=EXPERIMENT,
        model_name=MODEL,
        git_sha="test",
        _env_file=None,
        **kw,
    )


@pytest.fixture(scope="module")
def first(registry: Path, config: TrainConfig) -> TrainResult:
    return train(_settings(registry, promote=True), config)


def test_first_model_is_registered_tagged_and_promoted(first: TrainResult) -> None:
    client = MlflowClient()
    mv = client.get_model_version(MODEL, first.version)
    assert mv.tags["feature_version"] == FEATURE_VERSION
    assert mv.tags["generator_version"] == "2"
    assert mv.tags["gate.outcome"] == "promoted"
    assert str(client.get_model_version_by_alias(MODEL, CHAMPION).version) == first.version
    assert first.report.pr_auc > 0.5
    assert first.promotion is not None
    assert first.promotion.champion_before is None


def test_run_has_report_card_and_gate(first: TrainResult, tmp_path: Path) -> None:
    client = MlflowClient()
    card = Path(client.download_artifacts(first.run_id, CARD, str(tmp_path))).read_text()
    assert "{{" not in card
    assert f"{MODEL} v{first.version}" in card
    assert "promoted" in card
    gate = json.loads(
        Path(client.download_artifacts(first.run_id, GATE, str(tmp_path))).read_text()
    )
    assert gate["promote"] is True


def test_run_logs_the_drift_reference_profile(first: TrainResult, tmp_path: Path) -> None:
    local = MlflowClient().download_artifacts(first.run_id, PROFILE_ARTIFACT, str(tmp_path))
    profile = Profile.from_dict(json.loads(Path(local).read_text()))
    assert set(profile.features) == set(FEATURE_NAMES)
    assert profile.n > 0


def test_registered_model_scores_probabilities(first: TrainResult, registry: Path) -> None:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL}@{CHAMPION}")
    example = model.input_example
    out = model.predict(example)
    assert out.shape == (len(example), 2)
    assert ((out >= 0) & (out <= 1)).all()


def test_identical_retrain_is_rejected(
    first: TrainResult, registry: Path, config: TrainConfig
) -> None:
    """LightGBM is deterministic here, so a retrain on the same data cannot beat the champion."""
    second = train(_settings(registry, promote=True), config)
    assert second.promotion is not None
    assert not second.promotion.result.promote
    assert "gain" in second.promotion.result.reason
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL, CHAMPION).version) == first.version
    assert client.get_model_version(MODEL, second.version).tags["gate.outcome"] == "rejected"


def test_incomparable_champion_is_replaced(first: TrainResult, config: TrainConfig) -> None:
    """A champion trained on another feature version cannot be compared; the floor decides."""
    client = MlflowClient()
    client.set_model_version_tag(MODEL, first.version, "feature_version", "0")
    latest = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL}'"))
    outcome = run_gate(client, MODEL, str(latest), config.gate.rule())
    assert outcome.result.promote
    assert "not comparable" in outcome.result.reason
    assert str(client.get_model_version_by_alias(MODEL, CHAMPION).version) == str(latest)
    assert str(client.get_model_version_by_alias(MODEL, PREVIOUS).version) == first.version


def test_missing_data_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_URI", str(tmp_path / "missing.parquet"))
    assert train_module.main() == train_module.EXIT_DATA_INVALID
