"""Training settings (environment) and configuration (YAML)."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.evaluation.gate import GateRule
from ml.features import FEATURE_NAMES


class TrainSettings(BaseSettings):
    """Where to read data and where to log. MLFLOW_TRACKING_URI is read by MLflow itself."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_uri: Path = Path("data/transactions.parquet")
    train_config: Path | None = None  # default: the packaged config.yaml
    mlflow_experiment_name: str = "fraud-detector-dev"
    model_name: str = "fraud-detector"
    promote: bool = False
    promote_strict: bool = False  # exit non-zero when the gate rejects
    git_sha: str | None = None  # set in images; otherwise read from git


class SplitConfig(BaseModel):
    valid_fraction: float = Field(gt=0, lt=0.5)
    test_fraction: float = Field(gt=0, lt=0.5)


class GateConfig(BaseModel):
    min_pr_auc_gain: float
    max_recall_drop: float
    min_pr_auc: float

    def rule(self) -> GateRule:
        return GateRule(**self.model_dump())


class TrainConfig(BaseModel):
    split: SplitConfig
    lightgbm: dict[str, Any]
    early_stopping_rounds: int = Field(gt=0)
    categorical_features: list[str]
    gate: GateConfig

    def model_post_init(self, _context: Any) -> None:
        unknown = set(self.categorical_features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"categorical_features not in FEATURE_NAMES: {sorted(unknown)}")

    @classmethod
    def load(cls, path: Path | None = None) -> TrainConfig:
        text = (
            path.read_text()
            if path is not None
            else files("ml.training").joinpath("config.yaml").read_text()
        )
        return cls.model_validate(yaml.safe_load(text))
