from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.card import render_model_card
from ml.evaluation.gate import GateRule, gate
from ml.evaluation.metrics import EvalReport, evaluate, threshold_for_fpr

RULE = GateRule(min_pr_auc_gain=0.005, max_recall_drop=0.01, min_pr_auc=0.5)


def _data(n: int = 2000, fraud: int = 100, seed: int = 0) -> tuple[np.ndarray, list[str | None]]:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, fraud, replace=False)] = 1
    patterns: list[str | None] = [
        ("card_testing" if i % 2 else "account_takeover") if label else None
        for i, label in enumerate(y)
    ]
    return y, patterns


def test_perfect_scores() -> None:
    y, patterns = _data()
    report = evaluate(y.astype(float), y, patterns)
    assert report.pr_auc == pytest.approx(1.0)
    assert report.recall_at_reference_fpr == pytest.approx(1.0)
    assert set(report.per_pattern_recall.values()) == {1.0}
    assert report.n_fraud == 100


def test_random_scores_are_near_base_rate() -> None:
    y, patterns = _data()
    report = evaluate(np.random.default_rng(1).random(len(y)), y, patterns)
    assert report.pr_auc < 0.15
    assert report.roc_auc == pytest.approx(0.5, abs=0.1)


def test_threshold_respects_fpr_budget() -> None:
    y, _ = _data(seed=2)
    scores = np.random.default_rng(3).random(len(y)) + y * 0.5
    threshold, _ = threshold_for_fpr(scores, y, 0.01)
    fpr = np.mean(scores[y == 0] >= threshold)
    assert fpr <= 0.01


def test_report_roundtrips_through_dict() -> None:
    y, patterns = _data()
    report = evaluate(np.clip(y * 0.7 + 0.1, 0, 1), y, patterns)
    assert EvalReport.from_dict(report.to_dict()) == report
    assert all(isinstance(k, str) for k in report.per_pattern_recall)


def test_evaluate_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both"):
        evaluate([0.1, 0.2], [0, 0], [None, None])


def _report(pr_auc: float, recall: float) -> EvalReport:
    return EvalReport(
        n=100, n_fraud=10, pr_auc=pr_auc, roc_auc=0.9, brier=0.01,
        recall_at_fpr={"0.5%": recall, "1.0%": recall, "2.0%": recall},
        threshold_at_fpr={"0.5%": 0.9, "1.0%": 0.8, "2.0%": 0.7},
        per_pattern_recall={}, per_pattern_count={},
        recommended_thresholds={"review": 0.7, "decline": 0.9},
    )  # fmt: skip


@pytest.mark.parametrize(
    ("challenger", "champion", "promote", "why"),
    [
        (_report(0.80, 0.9), None, True, "no comparable champion"),
        (_report(0.40, 0.9), None, False, "below the floor"),
        (_report(0.80, 0.90), _report(0.797, 0.90), False, "gain"),
        (_report(0.80, 0.85), _report(0.70, 0.90), False, "recall"),
        (_report(0.80, 0.895), _report(0.70, 0.90), True, "0.7000 -> 0.8000"),
    ],
)
def test_gate(challenger: EvalReport, champion: EvalReport | None, promote: bool, why: str) -> None:
    result = gate(challenger, champion, RULE)
    assert result.promote is promote
    assert why in result.reason


def test_model_card_has_no_unfilled_placeholders() -> None:
    y, patterns = _data()
    report = evaluate(np.clip(y * 0.7 + 0.1, 0, 1), y, patterns)
    meta = dict.fromkeys(
        [
            "model_name",
            "version",
            "run_url",
            "trained_at",
            "git_sha",
            "feature_version",
            "data_uri",
            "seed",
            "generator_version",
            "customers",
            "days",
            "n_train",
            "n_valid",
            "n_test",
            "valid_start",
            "test_start",
            "test_fraud_rate",
        ],
        "x",
    )
    card = render_model_card(report, meta, {"amount": 2.0, "txn_count_1h": 6.0})
    assert "{{" not in card
    assert "| txn_count_1h | 75.0% |" in card
    with pytest.raises(KeyError, match="run_url"):
        render_model_card(report, {k: v for k, v in meta.items() if k != "run_url"}, {})
