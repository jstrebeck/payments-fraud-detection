"""Fraud scorers and the factory that picks one from settings."""

from __future__ import annotations

from payments_api.config import Settings
from payments_api.scoring.base import FraudScorer, ScoreResult
from payments_api.scoring.rule import RuleScorer

__all__ = ["FraudScorer", "RuleScorer", "ScoreResult", "build_scorer"]


def build_scorer(settings: Settings) -> FraudScorer:
    if settings.fraud_scorer == "rule":
        return RuleScorer()
    if settings.fraud_scorer == "mlflow":
        # Imported here so the rule-only configuration never imports MLflow.
        from payments_api.scoring.mlflow_model import MlflowScorer

        return MlflowScorer(
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.model_name,
            alias=settings.model_alias,
            refresh_seconds=settings.model_refresh_seconds,
        )
    from payments_api.scoring.kserve import KServeScorer, mlflow_feature_version_lookup

    return KServeScorer(
        url=settings.kserve_url,
        model_name=settings.model_name,
        timeout_s=settings.kserve_timeout_seconds,
        lookup=mlflow_feature_version_lookup(settings.mlflow_tracking_uri, settings.model_name),
    )
