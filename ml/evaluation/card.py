"""Model card rendering from the evaluation report and run metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.resources import files

from ml.evaluation.gate import GateResult
from ml.evaluation.metrics import EvalReport

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_model_card(
    report: EvalReport,
    meta: Mapping[str, object],
    importance: Mapping[str, float],
    gate_result: GateResult | None = None,
) -> str:
    """Fill the template. `meta` supplies run and data fields; unknown or missing
    placeholders raise, so the template and the generator cannot drift silently."""
    patterns = "\n".join(
        f"| {p} | {report.per_pattern_count[p]} | {r:.3f} |"
        for p, r in report.per_pattern_recall.items()
    )
    total = sum(importance.values()) or 1.0
    features = "\n".join(
        f"| {name} | {gain / total:.1%} |"
        for name, gain in sorted(importance.items(), key=lambda kv: -kv[1])
    )
    metrics = report.flat_metrics()
    values: dict[str, object] = {
        **meta,
        "pr_auc": _fmt(report.pr_auc),
        "roc_auc": _fmt(report.roc_auc),
        "brier": _fmt(report.brier),
        "recall_at_1pct_fpr": _fmt(metrics["recall_at_fpr_1pct"]),
        "recall_at_0_5pct_fpr": _fmt(metrics["recall_at_fpr_0_5pct"]),
        "recall_at_2pct_fpr": _fmt(metrics["recall_at_fpr_2pct"]),
        "per_pattern_table": "| Pattern | Rows | Recall |\n|---|---|---|\n" + patterns,
        "review_threshold": _fmt(report.recommended_thresholds["review"]),
        "decline_threshold": _fmt(report.recommended_thresholds["decline"]),
        "feature_table": "| Feature | Share of gain |\n|---|---|\n" + features,
        "n_features": len(importance),
        "champion_pr_auc": _fmt(gate_result.champion_pr_auc if gate_result else None),
        "champion_recall_at_1pct_fpr": _fmt(gate_result.champion_recall if gate_result else None),
        "gate_rule": gate_result.rule.describe() if gate_result else "not run",
        "gate_outcome": (
            ("promoted" if gate_result.promote else "not promoted") if gate_result else "not gated"
        ),
        "gate_reason": gate_result.reason if gate_result else "Run `make promote` to gate it.",
    }
    template = files("ml.evaluation").joinpath("model_card_template.md").read_text()

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"model card placeholder {{{{{key}}}}} has no value")
        return str(values[key])

    return _PLACEHOLDER.sub(sub, template)
