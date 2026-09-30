"""Automated retraining on labelled live decisions (Phase 7).

`python -m ml.training.retrain`, run every 30 minutes by the `retrain` CronJob
(deploy/base/retrain/). Each run decides whether to retrain, and if it does:

1. **Trigger.** Retrain when the `FraudFeatureDrift` alert is firing in
   Prometheus, or the champion is older than `RETRAIN_MAX_AGE_DAYS`, unless a
   retraining run started within `RETRAIN_COOLDOWN_HOURS`. `RETRAIN_TRIGGER=force`
   skips the checks.
2. **Data.** A small generated history (the baseline world) plus every labelled
   payment from the last `LIVE_LOOKBACK_DAYS`, using the feature vectors the API
   stored when it scored them: exactly what the model saw online, no
   recomputation. Labels come from delayed feedback (chargebacks, confirmations).
3. **Split.** Labelled live payments in order of scoring time: the most recent
   `LIVE_TEST_FRACTION` is the test window, the slice before it validates (early
   stopping), everything older plus the generated history trains. The gate
   therefore compares challenger and champion on the most recent production
   traffic, which is where a drifted champion loses.
4. **Train, gate, promote.** The same code as `make train` (`train_on_split`),
   with the gate. On a win, `champion` moves and the Job rolls the predictor
   (ml.evaluation.rollout). The new version carries its recommended decision
   thresholds as tags, which the API applies when it serves it.

Exit codes: 0 skipped, rejected or promoted; 1 on errors; 4 promoted but the
predictor rollout failed (the alias has moved; rerun `promote.py --rollout-only`).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.data.generator import GENERATOR_VERSION, GeneratorConfig, generate
from ml.data.io import iter_transactions
from ml.data.schema import DRIFT_FRAUD_PATTERNS, FRAUD_PATTERNS
from ml.features import FEATURE_NAMES
from ml.features.batch import featurize
from ml.training.config import TrainConfig, TrainSettings
from ml.training.data import Dataset, Split

EXIT_ROLLOUT_FAILED = 4
DRIFT_ALERT = "FraudFeatureDrift"
KNOWN_PATTERNS = frozenset(FRAUD_PATTERNS) | frozenset(DRIFT_FRAUD_PATTERNS)


class RetrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/payments"
    prometheus_url: str = "http://kube-prometheus-stack-prometheus.monitoring.svc:9090"
    retrain_trigger: Literal["auto", "force"] = "auto"
    retrain_cooldown_hours: float = Field(default=6.0, ge=0)
    retrain_max_age_days: float = Field(default=7.0, gt=0)
    live_lookback_days: float = Field(default=7.0, gt=0)
    min_labelled: int = Field(default=1000, ge=1)
    min_test_fraud: int = Field(default=20, ge=1)
    min_valid_fraud: int = Field(default=10, ge=1)
    live_test_fraction: float = Field(default=0.4, gt=0, lt=1)
    live_valid_fraction: float = Field(default=0.2, gt=0, lt=1)
    base_customers: int = Field(default=1000, ge=1)
    base_days: int = Field(default=30, ge=1)
    base_seed: int | None = None  # default: time-based, logged
    rollout: bool = True
    kserve_namespace: str = "fraud"
    kserve_isvc: str = "fraud-detector"


@dataclass(frozen=True)
class Decision:
    retrain: bool
    trigger: str  # drift | schedule | force | none
    reason: str


def decide(
    settings: RetrainSettings,
    *,
    drift_firing: bool,
    champion_age: timedelta | None,
    since_last_retrain: timedelta | None,
) -> Decision:
    """Pure trigger policy (tested without Prometheus or MLflow)."""
    if settings.retrain_trigger == "force":
        return Decision(True, "force", "RETRAIN_TRIGGER=force")
    cooldown = timedelta(hours=settings.retrain_cooldown_hours)
    if since_last_retrain is not None and since_last_retrain < cooldown:
        return Decision(
            False, "none", f"last retrain {_ago(since_last_retrain)}, cooldown {cooldown}"
        )
    if drift_firing:
        return Decision(True, "drift", f"{DRIFT_ALERT} is firing")
    max_age = timedelta(days=settings.retrain_max_age_days)
    if champion_age is None:
        return Decision(True, "schedule", "no champion")
    if champion_age > max_age:
        return Decision(True, "schedule", f"champion is {_ago(champion_age)} old (max {max_age})")
    return Decision(False, "none", f"no drift, champion is {_ago(champion_age)} old")


def _ago(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"


def drift_alert_firing(prometheus_url: str) -> bool:
    query = f'ALERTS{{alertname="{DRIFT_ALERT}",alertstate="firing"}}'
    url = f"{prometheus_url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': query})}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        body: dict[str, Any] = json.load(resp)
    return bool(body["data"]["result"])


# ---- data ------------------------------------------------------------------------


def base_dataset(seed: int, customers: int, days: int) -> Dataset:
    """Generated history of the baseline world, featurised with the batch path."""
    rows = list(
        iter_transactions(generate(GeneratorConfig(seed=seed, customers=customers, days=days)))
    )
    return Dataset(
        features=pd.DataFrame(featurize(rows), columns=list(FEATURE_NAMES), dtype="float64"),
        label=np.array([r.is_fraud for r in rows], dtype=np.int64),
        pattern=list[str | None]([r.fraud_pattern for r in rows]),
        timestamp=pd.Series([r.timestamp for r in rows]),
    )


def live_frame(database_url: str, lookback_days: float) -> pd.DataFrame:
    """Labelled payments scored in the lookback window, oldest first."""
    import sqlalchemy as sa

    from ml.evaluation.drift_monitor import sync_database_url

    query = sa.text(
        "SELECT features, label, label_reason, created_at FROM payments "
        "WHERE label IS NOT NULL AND created_at >= now() - make_interval(secs => :secs) "
        "ORDER BY created_at"
    )
    engine = sa.create_engine(sync_database_url(database_url))
    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"secs": lookback_days * 86400}).mappings().all()
    finally:
        engine.dispose()
    return pd.DataFrame(
        [dict(r) for r in rows], columns=["features", "label", "label_reason", "created_at"]
    )


def live_dataset(frame: pd.DataFrame) -> Dataset:
    """Stored online feature vectors plus delayed labels. Timestamp = scoring time."""
    features = pd.DataFrame(
        [[float(f[name]) for name in FEATURE_NAMES] for f in frame["features"]],
        columns=list(FEATURE_NAMES),
        dtype="float64",
    )
    is_fraud = (frame["label"] == "fraud").to_numpy()
    pattern: list[str | None] = [
        (r if isinstance(r, str) and r in KNOWN_PATTERNS else None) if f else None
        for f, r in zip(is_fraud, frame["label_reason"], strict=True)
    ]
    return Dataset(
        features=features,
        label=is_fraud.astype(np.int64),
        pattern=pattern,
        timestamp=pd.to_datetime(frame["created_at"], utc=True).reset_index(drop=True),
    )


class NotEnoughLabelsError(RuntimeError):
    """Too few labelled payments to train and gate responsibly."""


def retrain_split(base: Dataset, live: Dataset, settings: RetrainSettings) -> Split:
    """Base + older live trains; recent live validates; most recent live tests."""
    n = len(live)
    if n < settings.min_labelled:
        raise NotEnoughLabelsError(f"{n} labelled payments, need {settings.min_labelled}")
    n_test = math.ceil(n * settings.live_test_fraction)
    n_valid = math.ceil(n * settings.live_valid_fraction)
    idx = np.arange(n)
    in_test = idx >= n - n_test
    in_valid = (idx >= n - n_test - n_valid) & ~in_test
    test, valid = live.take(in_test), live.take(in_valid)
    if int(test.label.sum()) < settings.min_test_fraud:
        raise NotEnoughLabelsError(
            f"{int(test.label.sum())} fraud labels in the test window, "
            f"need {settings.min_test_fraud}"
        )
    if int(valid.label.sum()) < settings.min_valid_fraud:
        raise NotEnoughLabelsError(
            f"{int(valid.label.sum())} fraud labels to validate, need {settings.min_valid_fraud}"
        )
    train = _concat(base, live.take(~in_test & ~in_valid))
    return Split(
        train=train,
        valid=valid,
        test=test,
        valid_start=valid.timestamp.min().to_pydatetime(),
        test_start=test.timestamp.min().to_pydatetime(),
    )


def _concat(a: Dataset, b: Dataset) -> Dataset:
    return Dataset(
        features=pd.concat([a.features, b.features], ignore_index=True),
        label=np.concatenate([a.label, b.label]),
        pattern=[*a.pattern, *b.pattern],
        timestamp=pd.concat(
            [pd.to_datetime(a.timestamp, utc=True), pd.to_datetime(b.timestamp, utc=True)],
            ignore_index=True,
        ),
    )


# ---- run -------------------------------------------------------------------------


def main() -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient

    from ml.evaluation.promote import champion_version
    from ml.evaluation.rollout import KubeApi, RolloutError, in_cluster, rollout
    from ml.training.train import summary, train_on_split

    settings = RetrainSettings()
    train_settings = TrainSettings(promote=True)
    config = TrainConfig.load(train_settings.train_config)
    client = MlflowClient()
    now = datetime.now(UTC)

    champion = champion_version(client, train_settings.model_name)
    champion_age = (
        now - datetime.fromtimestamp(int(champion.creation_timestamp) / 1000, UTC)
        if champion is not None
        else None
    )
    since_last = _since_last_retrain(client, train_settings.mlflow_experiment_name, now)
    firing = settings.retrain_trigger == "auto" and drift_alert_firing(settings.prometheus_url)
    decision = decide(
        settings, drift_firing=firing, champion_age=champion_age, since_last_retrain=since_last
    )
    _log("decision", retrain=decision.retrain, trigger=decision.trigger, reason=decision.reason)
    if not decision.retrain:
        return 0

    seed = settings.base_seed if settings.base_seed is not None else int(time.time())
    try:
        live = live_dataset(live_frame(settings.database_url, settings.live_lookback_days))
        base = base_dataset(seed, settings.base_customers, settings.base_days)
        split = retrain_split(base, live, settings)
    except NotEnoughLabelsError as exc:
        _log("skipped", reason=str(exc))
        return 0

    data_meta = {
        "source": "retrain",
        "generator_version": GENERATOR_VERSION,
        "seed": str(seed),
        "customers": str(settings.base_customers),
        "days": str(settings.base_days),
    }
    live_fraud = int(live.label.sum())
    # Recorded as the run's data_uri: where this model's data came from.
    train_settings = train_settings.model_copy(
        update={"data_uri": Path(f"generated:seed={seed}+payments:labelled={len(live)}")}
    )
    result = train_on_split(
        train_settings,
        config,
        split,
        data_meta,
        extra_params={
            "retrain.trigger": decision.trigger,
            "retrain.reason": decision.reason,
            "retrain.live_rows": len(live),
            "retrain.live_fraud": live_fraud,
            "retrain.lookback_days": settings.live_lookback_days,
            "retrain.live_test_fraction": settings.live_test_fraction,
        },
        extra_tags={"retrain": "true", "retrain_trigger": decision.trigger},
    )
    _log("trained", **summary(train_settings, result))

    promotion = result.promotion
    if promotion is None or not promotion.result.promote or promotion.champion_after is None:
        return 0
    if not settings.rollout or not in_cluster():
        _log("rollout_skipped", reason="not in cluster or ROLLOUT=false",
             hint="uv run python scripts/promote.py --rollout-only")  # fmt: skip
        return 0
    try:
        rollout(
            KubeApi(), settings.kserve_namespace, settings.kserve_isvc, promotion.champion_after
        )
    except (RolloutError, OSError) as exc:
        _log("rollout_failed", error=repr(exc))
        return EXIT_ROLLOUT_FAILED
    _log("rolled_out", version=promotion.champion_after)
    return 0


def _since_last_retrain(client: Any, experiment: str, now: datetime) -> timedelta | None:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.retrain = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return now - datetime.fromtimestamp(runs[0].info.start_time / 1000, UTC)


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
