"""Runtime configuration, read from environment variables (and `.env` in dev)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/payments"
    # Which FraudScorer implementation to use: rules, the MLflow model in-process
    # (compose), or the KServe InferenceService (cluster).
    fraud_scorer: Literal["rule", "mlflow", "kserve"] = "rule"
    # Readiness fails when the primary scorer is unavailable. Off by default:
    # the rule fallback keeps serving, which is the point of having it.
    require_scorer: bool = False

    # MlflowScorer: registry model to load in-process (ADR-0012, ADR-0013).
    mlflow_tracking_uri: str | None = None  # None: MLflow's own default/env
    model_name: str = "fraud-detector"
    model_alias: str = "champion"
    # How often to check whether the alias moved; 0 disables hot reload.
    model_refresh_seconds: float = Field(default=60.0, ge=0.0)
    # KServeScorer: the InferenceService's V2 endpoint. The timeout is the whole
    # request budget; past it the payment is scored by rules (and counted).
    kserve_url: str = "http://fraud-detector-predictor.fraud.svc/v2/models/fraud-detector/infer"
    kserve_timeout_seconds: float = Field(default=0.3, gt=0.0)
    # Decision policy. Held in config so it can be tuned without a retrain.
    review_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    decline_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> Self:
        if self.review_threshold > self.decline_threshold:
            raise ValueError("review_threshold must be <= decline_threshold")
        return self
