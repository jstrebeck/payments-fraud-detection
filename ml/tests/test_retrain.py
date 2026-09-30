"""Retraining: trigger policy, live data parsing, the live-window split, and the rollout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.rollout import ANNOTATION, RolloutError, _rolled_out, rollout
from ml.features import FEATURE_NAMES
from ml.training.data import Dataset
from ml.training.retrain import (
    NotEnoughLabelsError,
    RetrainSettings,
    card_bucket,
    decide,
    live_dataset,
    retrain_split,
)

H = timedelta(hours=1)
D = timedelta(days=1)


def settings(**kw: Any) -> RetrainSettings:
    return RetrainSettings(**({"min_labelled": 10, "min_test_fraud": 1, "min_valid_fraud": 1} | kw))


# ---- trigger policy --------------------------------------------------------------


def test_drift_triggers_a_retrain() -> None:
    d = decide(settings(), drift_firing=True, champion_age=1 * D, since_last_retrain=None)
    assert (d.retrain, d.trigger) == (True, "drift")


def test_cooldown_after_a_promotion_blocks_even_when_drifting() -> None:
    d = decide(
        settings(),
        drift_firing=True,
        champion_age=1 * D,
        since_last_retrain=2 * H,
        last_retrain_promoted=True,
    )
    assert not d.retrain


def test_a_rejected_attempt_is_retried_after_an_hour() -> None:
    kw: dict[str, Any] = {"drift_firing": True, "champion_age": 1 * D}
    assert not decide(settings(), since_last_retrain=0.5 * H, **kw).retrain
    assert decide(settings(), since_last_retrain=2 * H, **kw).retrain


def test_old_champion_triggers_a_scheduled_retrain() -> None:
    d = decide(settings(), drift_firing=False, champion_age=8 * D, since_last_retrain=3 * D)
    assert (d.retrain, d.trigger) == (True, "schedule")


def test_quiet_and_fresh_means_no_retrain() -> None:
    d = decide(settings(), drift_firing=False, champion_age=2 * D, since_last_retrain=None)
    assert not d.retrain


def test_force_ignores_everything() -> None:
    s = settings(retrain_trigger="force")
    d = decide(s, drift_firing=False, champion_age=1 * H, since_last_retrain=1 * H)
    assert (d.retrain, d.trigger) == (True, "force")


# ---- data ------------------------------------------------------------------------


def _frame(n: int, fraud_every: int = 5) -> pd.DataFrame:
    start = datetime(2026, 9, 30, tzinfo=UTC)
    return pd.DataFrame(
        {
            "card_token": [f"tok_{i % 37:016x}" for i in range(n)],
            "features": [{name: float(i) for name in FEATURE_NAMES} for i in range(n)],
            "label": ["fraud" if i % fraud_every == 0 else "legit" for i in range(n)],
            "label_reason": ["session_hijack" if i % fraud_every == 0 else None for i in range(n)],
            "created_at": [start + i * timedelta(minutes=1) for i in range(n)],
        }
    )


def test_live_dataset_uses_stored_features_and_labels() -> None:
    live = live_dataset(_frame(10))
    assert list(live.features.columns) == list(FEATURE_NAMES)
    assert live.features.iloc[3, 0] == 3.0
    assert live.label.tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert live.pattern[0] == "session_hijack"
    assert live.pattern[1] is None


def test_unknown_chargeback_reasons_become_no_pattern() -> None:
    frame = _frame(5)
    frame.loc[0, "label_reason"] = "made-up-reason"
    assert live_dataset(frame).pattern[0] is None


def _base(n: int) -> Dataset:
    return Dataset(
        features=pd.DataFrame(np.zeros((n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)),
        label=np.zeros(n, dtype=np.int64),
        pattern=[None] * n,
        timestamp=pd.Series([datetime(2026, 1, 1, tzinfo=UTC)] * n),
    )


def test_split_is_by_card_and_base_only_trains() -> None:
    frame = _frame(400)
    live, cards = live_dataset(frame), list(frame["card_token"])
    split = retrain_split(_base(50), live, cards, settings())
    assert len(split.train) + len(split.valid) + len(split.test) == 50 + 400

    # Each live row's features encode its row index, which maps back to its card.
    test_cards = {cards[int(i)] for i in split.test.features.iloc[:, 0]}
    valid_cards = {cards[int(i)] for i in split.valid.features.iloc[:, 0]}
    live_train = split.train.features.iloc[50:, 0]
    train_cards = {cards[int(i)] for i in live_train}
    assert test_cards
    assert valid_cards
    assert train_cards
    assert not test_cards & valid_cards
    assert not test_cards & train_cards
    # The generated history is only ever training data.
    assert (split.train.features.iloc[:50] == 0).all().all()


def test_card_buckets_are_stable() -> None:
    assert card_bucket("tok_0123456789abcdef") == card_bucket("tok_0123456789abcdef")
    assert 0 <= card_bucket("tok_0123456789abcdef") < 1


def test_too_few_labels_is_a_skip_not_a_crash() -> None:
    with pytest.raises(NotEnoughLabelsError):
        retrain_split(_base(5), live_dataset(_frame(5)), list(_frame(5)["card_token"]), settings())
    frame = _frame(50, fraud_every=10_000)  # a single fraud in the whole window
    live, cards = live_dataset(frame), list(frame["card_token"])
    with pytest.raises(NotEnoughLabelsError, match="test window"):
        retrain_split(_base(5), live, cards, settings(min_test_fraud=5))


# ---- rollout ---------------------------------------------------------------------


def _deployment(
    annotation: str | None, updated: int, available: int, replicas: int
) -> dict[str, Any]:
    ann = {ANNOTATION: annotation} if annotation else {}
    return {
        "metadata": {"generation": 5},
        "spec": {"replicas": 1, "template": {"metadata": {"annotations": ann}}},
        "status": {
            "observedGeneration": 5,
            "updatedReplicas": updated,
            "availableReplicas": available,
            "replicas": replicas,
        },
    }


def test_rolled_out_needs_the_annotation_and_only_new_pods() -> None:
    assert not _rolled_out(_deployment("2", 1, 1, 1), "3")  # KServe hasn't copied it yet
    assert not _rolled_out(_deployment("3", 1, 1, 2), "3")  # old pod still around
    assert not _rolled_out(_deployment("3", 1, 0, 1), "3")  # new pod not ready
    assert _rolled_out(_deployment("3", 1, 1, 1), "3")


class FakeApi:
    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.states = states
        self.patches: list[tuple[str, Any]] = []

    def merge_patch(self, path: str, body: Any) -> Any:
        self.patches.append((path, body))
        return {}

    def get(self, path: str) -> Any:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def test_rollout_patches_the_isvc_and_waits() -> None:
    api = FakeApi([_deployment("2", 1, 1, 1), _deployment("3", 1, 1, 2), _deployment("3", 1, 1, 1)])
    rollout(api, "fraud", "fraud-detector", "3", poll_s=0)  # type: ignore[arg-type]
    ((path, body),) = api.patches
    assert path.endswith("/namespaces/fraud/inferenceservices/fraud-detector")
    assert body == {"spec": {"predictor": {"annotations": {ANNOTATION: "3"}}}}


def test_rollout_times_out() -> None:
    api = FakeApi([_deployment("2", 1, 1, 1)])
    with pytest.raises(RolloutError):
        rollout(api, "fraud", "fraud-detector", "3", timeout_s=0.01, poll_s=0)  # type: ignore[arg-type]
