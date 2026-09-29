"""Offline model quality metrics. Deterministic given scores and labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

# False-positive-rate budgets we report recall at. 1% is the gate's reference.
FPR_TARGETS: tuple[float, ...] = (0.005, 0.01, 0.02)
REFERENCE_FPR = 0.01
# Policy thresholds are recommended from these budgets (review is looser).
REVIEW_FPR = 0.02
DECLINE_FPR = 0.005


def fpr_key(fpr: float) -> str:
    return f"{fpr:.1%}"


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_score: float
    fraud_rate: float


@dataclass(frozen=True)
class EvalReport:
    """Model quality on one labelled window. The contract with the gate and the card."""

    n: int
    n_fraud: int
    pr_auc: float
    roc_auc: float
    brier: float
    recall_at_fpr: dict[str, float]
    threshold_at_fpr: dict[str, float]
    # Recall per fraud pattern at the reference-FPR threshold.
    per_pattern_recall: dict[str, float]
    per_pattern_count: dict[str, int]
    recommended_thresholds: dict[str, float]
    calibration: list[CalibrationBin] = field(default_factory=list)

    @property
    def recall_at_reference_fpr(self) -> float:
        return self.recall_at_fpr[fpr_key(REFERENCE_FPR)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReport:
        bins = [CalibrationBin(**b) for b in data.get("calibration", [])]
        return cls(**{**data, "calibration": bins})

    def flat_metrics(self) -> dict[str, float]:
        """Scalar metrics for mlflow.log_metrics (names without '%')."""
        out = {"pr_auc": self.pr_auc, "roc_auc": self.roc_auc, "brier": self.brier}
        for fpr in FPR_TARGETS:
            slug = f"{fpr * 100:g}pct".replace(".", "_")
            out[f"recall_at_fpr_{slug}"] = self.recall_at_fpr[fpr_key(fpr)]
        for pattern, recall in self.per_pattern_recall.items():
            out[f"recall_{pattern}"] = recall
        return out


def threshold_for_fpr(scores: np.ndarray, y: np.ndarray, fpr_target: float) -> tuple[float, float]:
    """Lowest threshold whose FPR (score >= threshold) stays within the budget.

    Returns (threshold, recall at that threshold).
    """
    fpr, tpr, thresholds = roc_curve(y, scores)
    ok = np.flatnonzero(fpr <= fpr_target)
    best = ok[-1]  # roc_curve is sorted by increasing FPR
    return float(min(thresholds[best], 1.0)), float(tpr[best])


def evaluate(
    scores: ArrayLike,
    y: ArrayLike,
    patterns: Sequence[str | None],
    n_bins: int = 10,
) -> EvalReport:
    s = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if s.shape != labels.shape or len(patterns) != len(labels):
        raise ValueError("scores, labels and patterns must have the same length")
    if labels.min() == labels.max():
        raise ValueError("evaluation window needs both fraud and legitimate rows")

    recall_at: dict[str, float] = {}
    threshold_at: dict[str, float] = {}
    for target in FPR_TARGETS:
        threshold_at[fpr_key(target)], recall_at[fpr_key(target)] = threshold_for_fpr(
            s, labels, target
        )

    ref_threshold = threshold_at[fpr_key(REFERENCE_FPR)]
    pattern_arr = np.array([p or "" for p in patterns])
    per_pattern_recall: dict[str, float] = {}
    per_pattern_count: dict[str, int] = {}
    for pattern in sorted({str(p) for p in pattern_arr[labels == 1]} - {""}):
        mask = pattern_arr == pattern
        per_pattern_count[pattern] = int(mask.sum())
        per_pattern_recall[pattern] = float(np.mean(s[mask] >= ref_threshold))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    calibration = [
        CalibrationBin(
            lower=float(edges[b]),
            upper=float(edges[b + 1]),
            count=int((idx == b).sum()),
            mean_score=float(s[idx == b].mean()),
            fraud_rate=float(labels[idx == b].mean()),
        )
        for b in range(n_bins)
        if (idx == b).any()
    ]

    return EvalReport(
        n=int(labels.size),
        n_fraud=int(labels.sum()),
        pr_auc=float(average_precision_score(labels, s)),
        roc_auc=float(roc_auc_score(labels, s)),
        brier=float(brier_score_loss(labels, s)),
        recall_at_fpr=recall_at,
        threshold_at_fpr=threshold_at,
        per_pattern_recall=per_pattern_recall,
        per_pattern_count=per_pattern_count,
        recommended_thresholds={
            "review": threshold_at[fpr_key(REVIEW_FPR)],
            "decline": threshold_at[fpr_key(DECLINE_FPR)],
        },
        calibration=calibration,
    )
