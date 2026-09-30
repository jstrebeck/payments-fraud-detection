# ml.evaluation

Offline metrics, the promotion gate and the model card. Imported by the
trainer and `scripts/promote.py`. Needs the `train` extra.

| Module | Contents |
|---|---|
| `metrics.py` | `evaluate(scores, labels, patterns) -> EvalReport`: PR-AUC, ROC-AUC, Brier, recall and threshold at 0.5% / 1% / 2% FPR, per-pattern recall at the 1% FPR threshold, 10-bin calibration table, recommended API thresholds |
| `gate.py` | `gate(challenger, champion, rule) -> GateResult`. Pure. |
| `promote.py` | `run_gate(client, name, version, rule)`: the registry side. The only code that moves `champion` |
| `card.py`, `model_card_template.md` | `render_model_card(...)`; fails on any unfilled placeholder |
| `exporter.py` | Prometheus exporter for the MLflow registry (`python -m ml.evaluation.exporter`): alias targets, per-version test metrics, newest training run per experiment. Runs as `deploy/base/registry-exporter` from the trainer image; feeds the training dashboard |

## Gate rule (`ml/training/config.yaml`)

Promote when PR-AUC >= 0.5 and, if a comparable champion exists, PR-AUC
improves by at least 0.005 without recall at 1% FPR dropping more than 0.01.
Both models are scored on the **challenger's** test window. A champion with a
different `feature_version` is not comparable (ADR-0013); then only the floor
applies. On a win, the old champion gets the `previous` alias.

Every gate run records its outcome on the version (`gate.outcome`,
`gate.reason`, `gate.champion` tags) and on its run (`evaluation/gate.json`,
re-rendered `model_card.md`).

## Planned

- `drift(reference, live) -> DriftReport` (Phase 7): PSI per feature,
  exported as `fraud_feature_psi`.

## Rules

- Deterministic given inputs. No randomness without an explicit seed.
- Reports are plain dataclasses serialisable to JSON; they are the contract
  with the gate, the card and (later) dashboards.
