"""Champion/challenger promotion gate. Pure: reports in, decision out."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ml.evaluation.metrics import EvalReport


@dataclass(frozen=True)
class GateRule:
    # Challenger must beat the champion's PR-AUC by at least this much...
    min_pr_auc_gain: float = 0.005
    # ...without losing more than this much recall at 1% FPR...
    max_recall_drop: float = 0.01
    # ...and must clear this absolute floor (also the only check with no champion).
    min_pr_auc: float = 0.5

    def describe(self) -> str:
        return (
            f"PR-AUC >= {self.min_pr_auc} and, if a comparable champion exists, "
            f"PR-AUC gain >= {self.min_pr_auc_gain} with recall@1%FPR drop <= "
            f"{self.max_recall_drop}"
        )


@dataclass(frozen=True)
class GateResult:
    promote: bool
    reason: str
    rule: GateRule
    challenger_pr_auc: float
    challenger_recall: float
    champion_pr_auc: float | None = None
    champion_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gate(challenger: EvalReport, champion: EvalReport | None, rule: GateRule) -> GateResult:
    c_auc, c_rec = challenger.pr_auc, challenger.recall_at_reference_fpr

    def result(promote: bool, reason: str) -> GateResult:
        return GateResult(
            promote=promote,
            reason=reason,
            rule=rule,
            challenger_pr_auc=c_auc,
            challenger_recall=c_rec,
            champion_pr_auc=champion.pr_auc if champion else None,
            champion_recall=champion.recall_at_reference_fpr if champion else None,
        )

    if c_auc < rule.min_pr_auc:
        return result(False, f"PR-AUC {c_auc:.4f} is below the floor {rule.min_pr_auc}")
    if champion is None:
        return result(True, f"no comparable champion; PR-AUC {c_auc:.4f} clears the floor")

    gain = c_auc - champion.pr_auc
    drop = champion.recall_at_reference_fpr - c_rec
    if gain < rule.min_pr_auc_gain:
        return result(
            False,
            f"PR-AUC gain {gain:+.4f} ({champion.pr_auc:.4f} -> {c_auc:.4f}) "
            f"is below {rule.min_pr_auc_gain}",
        )
    if drop > rule.max_recall_drop:
        return result(False, f"recall@1%FPR drops by {drop:.4f} (limit {rule.max_recall_drop})")
    return result(
        True,
        f"PR-AUC {champion.pr_auc:.4f} -> {c_auc:.4f} ({gain:+.4f}), "
        f"recall@1%FPR {champion.recall_at_reference_fpr:.4f} -> {c_rec:.4f}",
    )
