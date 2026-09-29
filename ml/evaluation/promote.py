"""The promotion gate against the MLflow registry (ADR-0004, ADR-0012).

The only code path that moves the `champion` alias. Used by the trainer
(`PROMOTE=true`) and by `scripts/promote.py`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ml.evaluation.card import render_model_card
from ml.evaluation.gate import GateResult, GateRule, gate
from ml.evaluation.metrics import EvalReport, evaluate
from ml.features import FEATURE_NAMES

CHAMPION = "champion"
PREVIOUS = "previous"

# Artifact paths inside each training run.
TEST_SET = "evaluation/test_set.parquet"
REPORT = "evaluation/report.json"
IMPORTANCE = "evaluation/feature_importance.json"
CARD_META = "evaluation/card_meta.json"
GATE = "evaluation/gate.json"
CARD = "model_card.md"


@dataclass(frozen=True)
class PromotionOutcome:
    version: str
    result: GateResult
    champion_before: str | None
    champion_after: str | None


def fraud_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Fraud probability from a pyfunc model logged with predict_proba."""
    out = np.asarray(model.predict(features[list(FEATURE_NAMES)]))
    return out[:, 1] if out.ndim == 2 else out


def champion_version(client: MlflowClient, name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None


def _download_json(client: MlflowClient, run_id: str, path: str) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(client.download_artifacts(run_id, path, tmp)).read_text())


def run_gate(
    client: MlflowClient,
    name: str,
    version: str,
    rule: GateRule,
    *,
    apply: bool = True,
) -> PromotionOutcome:
    """Evaluate `version` and the current champion on the challenger's test window;
    move the alias only on a win (and only if `apply`). Records the outcome on the
    version (tags) and its run (gate.json, re-rendered model card)."""
    version = str(version)
    challenger = client.get_model_version(name, version)
    run_id = challenger.run_id
    if run_id is None:
        raise ValueError(f"{name} v{version} has no source run")

    with tempfile.TemporaryDirectory() as tmp:
        test = pd.read_parquet(client.download_artifacts(run_id, TEST_SET, tmp))
    labels, patterns = test["is_fraud"].to_numpy(), list(test["fraud_pattern"])
    challenger_report = EvalReport.from_dict(_download_json(client, run_id, REPORT))

    current = champion_version(client, name)
    # Registry stores disagree on the type of `version` (str over REST, int from SQL).
    current_version = str(current.version) if current is not None else None
    champion_report: EvalReport | None = None
    note = ""
    if current is not None and current_version != version:
        if current.tags.get("feature_version") == challenger.tags.get("feature_version"):
            model = mlflow.pyfunc.load_model(f"models:/{name}/{current_version}")
            champion_report = evaluate(fraud_scores(model, test), labels, patterns)
        else:
            note = (
                f" (champion v{current_version} uses feature version "
                f"{current.tags.get('feature_version')}, not comparable)"
            )

    result = gate(challenger_report, champion_report, rule)
    if note:
        result = GateResult(**{**result.__dict__, "reason": result.reason + note})
    if current_version == version:
        result = GateResult(**{**result.__dict__, "promote": False,
                               "reason": f"v{version} is already the champion"})  # fmt: skip

    champion_before = current_version
    champion_after = champion_before
    if result.promote and apply:
        if current_version is not None:
            client.set_registered_model_alias(name, PREVIOUS, current_version)
        client.set_registered_model_alias(name, CHAMPION, version)
        champion_after = version

    outcome = "promoted" if result.promote and apply else (
        "would promote" if result.promote else "rejected")  # fmt: skip
    client.set_model_version_tag(name, version, "gate.outcome", outcome)
    client.set_model_version_tag(name, version, "gate.reason", result.reason)
    client.set_model_version_tag(name, version, "gate.champion", champion_before or "none")
    client.log_text(run_id, json.dumps(result.to_dict(), indent=2), GATE)
    card = render_model_card(
        challenger_report,
        _download_json(client, run_id, CARD_META) | {"version": version},
        _download_json(client, run_id, IMPORTANCE),
        result,
    )
    client.log_text(run_id, card, CARD)
    return PromotionOutcome(version, result, champion_before, champion_after)
