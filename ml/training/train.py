"""Train, evaluate, log and register the fraud model; optionally run the gate.

`python -m ml.training` (or `make train`). Stages as in README.md. Exit codes:
0 success, 2 data validation failed, 3 gate rejected with PROMOTE_STRICT=true.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pydantic
from mlflow import MlflowClient
from mlflow.models import infer_signature

from ml.evaluation.card import render_model_card
from ml.evaluation.metrics import EvalReport, evaluate
from ml.evaluation.promote import (
    CARD,
    CARD_META,
    IMPORTANCE,
    REPORT,
    TEST_SET,
    PromotionOutcome,
    run_gate,
)
from ml.features import FEATURE_NAMES, FEATURE_VERSION
from ml.training.config import TrainConfig, TrainSettings
from ml.training.data import Split, load_dataset, time_split

# Types skops may deserialise besides its sklearn/numpy defaults (the model itself).
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]

# What a serving runtime needs to load the model, pinned to what trained it.
# Explicit so MLflow does not export the whole uv workspace as model requirements.
MODEL_REQUIREMENTS = ("lightgbm", "scikit-learn", "skops", "numpy", "pandas")

EXIT_DATA_INVALID = 2
EXIT_GATE_REJECTED = 3


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    version: str
    report: EvalReport
    promotion: PromotionOutcome | None


def model_requirements() -> list[str]:
    reqs = [f"{name}=={package_version(name)}" for name in MODEL_REQUIREMENTS]
    return [f"mlflow=={package_version('mlflow-skinny')}", *reqs]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit(split: Split, config: TrainConfig) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**config.lightgbm)
    model.fit(
        split.train.features,
        split.train.label,
        eval_X=(split.valid.features,),
        eval_y=(split.valid.label,),
        eval_metric="average_precision",
        categorical_feature=config.categorical_features,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    return model


def train(settings: TrainSettings, config: TrainConfig) -> TrainResult:
    dataset, data_meta = load_dataset(settings.data_uri)
    split = time_split(dataset, config.split)
    sha = settings.git_sha or git_sha()

    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tags({"git_sha": sha, "feature_version": FEATURE_VERSION})
        mlflow.log_params(
            {
                "data_uri": str(settings.data_uri),
                **{f"data.{k}": v for k, v in data_meta.items()},
                "feature_version": FEATURE_VERSION,
                "n_features": len(FEATURE_NAMES),
                "n_train": len(split.train),
                "n_valid": len(split.valid),
                "n_test": len(split.test),
                "valid_start": split.valid_start.isoformat(),
                "test_start": split.test_start.isoformat(),
                "early_stopping_rounds": config.early_stopping_rounds,
                "categorical_features": ",".join(config.categorical_features),
                **{f"lgbm.{k}": v for k, v in config.lightgbm.items()},
                **{f"gate.{k}": v for k, v in config.gate.model_dump().items()},
            }
        )
        mlflow.log_dict(config.model_dump(), "train_config.json")

        model = fit(split, config)
        mlflow.log_metric("best_iteration", model.best_iteration_)

        scores = np.asarray(model.predict_proba(split.test.features))[:, 1]
        report = evaluate(scores, split.test.label, split.test.pattern)
        mlflow.log_metrics(report.flat_metrics())
        mlflow.log_dict(report.to_dict(), REPORT)

        gains = model.booster_.feature_importance("gain")
        importance = {name: float(g) for name, g in zip(FEATURE_NAMES, gains, strict=True)}
        mlflow.log_dict(importance, IMPORTANCE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(TEST_SET).name
            split.test.to_frame().to_parquet(path, index=False)
            mlflow.log_artifact(str(path), str(Path(TEST_SET).parent))

        example = split.test.features.head(5)
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            pip_requirements=model_requirements(),
            registered_model_name=settings.model_name,
        )
        version = str(info.registered_model_version)
        client = MlflowClient()
        tags = {
            "feature_version": FEATURE_VERSION,
            "generator_version": data_meta.get("generator_version", "unknown"),
            "seed": data_meta.get("seed", "unknown"),
            "git_sha": sha,
            "data_uri": str(settings.data_uri),
        }
        for key, value in tags.items():
            client.set_model_version_tag(settings.model_name, version, key, value)

        card_meta: dict[str, Any] = {
            "model_name": settings.model_name,
            "run_url": _run_url(run_id, run.info.experiment_id),
            "trained_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "git_sha": sha,
            "feature_version": FEATURE_VERSION,
            "data_uri": str(settings.data_uri),
            "seed": data_meta.get("seed", "?"),
            "generator_version": data_meta.get("generator_version", "?"),
            "customers": data_meta.get("customers", "?"),
            "days": data_meta.get("days", "?"),
            "n_train": len(split.train),
            "n_valid": len(split.valid),
            "n_test": len(split.test),
            "valid_start": split.valid_start.strftime("%Y-%m-%d %H:%M"),
            "test_start": split.test_start.strftime("%Y-%m-%d %H:%M"),
            "test_fraud_rate": f"{np.mean(split.test.label):.2%}",
        }
        mlflow.log_dict(card_meta, CARD_META)
        mlflow.log_text(
            render_model_card(report, card_meta | {"version": version}, importance), CARD
        )

    promotion = None
    if settings.promote:
        promotion = run_gate(MlflowClient(), settings.model_name, version, config.gate.rule())
    return TrainResult(run_id, version, report, promotion)


def _run_url(run_id: str, experiment_id: str) -> str:
    base = mlflow.get_tracking_uri().rstrip("/")
    if not base.startswith("http"):
        return f"run {run_id}"
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def summary(settings: TrainSettings, result: TrainResult) -> dict[str, Any]:
    r = result.report
    return {
        "model": f"{settings.model_name}/{result.version}",
        "run_id": result.run_id,
        "pr_auc": round(r.pr_auc, 4),
        "recall_at_1pct_fpr": round(r.recall_at_reference_fpr, 4),
        "per_pattern_recall": {k: round(v, 3) for k, v in r.per_pattern_recall.items()},
        "recommended_thresholds": {k: round(v, 4) for k, v in r.recommended_thresholds.items()},
        "gate": result.promotion.result.reason if result.promotion else "not run",
        "champion": result.promotion.champion_after if result.promotion else None,
    }


def main() -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    settings = TrainSettings()
    config = TrainConfig.load(settings.train_config)
    try:
        result = train(settings, config)
    except (FileNotFoundError, pydantic.ValidationError) as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return EXIT_DATA_INVALID

    print(json.dumps(summary(settings, result), indent=2))
    if result.promotion and not result.promotion.result.promote and settings.promote_strict:
        return EXIT_GATE_REJECTED
    return 0
