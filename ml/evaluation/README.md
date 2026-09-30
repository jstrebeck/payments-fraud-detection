# ml.evaluation

Offline metrics, the promotion gate and the model card. Imported by the
trainer and `scripts/promote.py`. Needs the `train` extra.

| Module | Contents |
|---|---|
| `metrics.py` | `evaluate(scores, labels, patterns) -> EvalReport`: PR-AUC, ROC-AUC, Brier, recall and threshold at 0.5% / 1% / 2% FPR, per-pattern recall at the 1% FPR threshold, 10-bin calibration table, recommended API thresholds |
| `gate.py` | `gate(challenger, champion, rule) -> GateResult`. Pure. |
| `promote.py` | `run_gate(client, name, version, rule)`: the registry side. The only code that moves `champion` |
| `card.py`, `model_card_template.md` | `render_model_card(...)`; fails on any unfilled placeholder |
| `drift.py` | Reference profiles and PSI. `build_profile(features)` bins each feature (categorical up to 32 values plus `other`, else training deciles with open ends); `psi(profile, live)`. The trainer logs the profile as `reference/feature_profile.json`. `python -m ml.evaluation.drift profile --version N` backfills a version trained before profiles existed |
| `drift_monitor.py` | Long-running exporter (`python -m ml.evaluation.drift_monitor`): every 5m, PSI of the last hour of payments' feature vectors against the champion's profile; `fraud_feature_psi{feature}`, `fraud_drift_*`. Runs as `deploy/base/drift-monitor` |
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

## Drift

PSI per feature is judged against each feature's own 24h median, not against
0 (`deploy/base/prometheusrules.yaml`, group `fraud.drift`): live traffic
never matches the training world exactly (history features in particular),
so only a jump above the usual level counts. Calendar features are excluded
by the monitor. Runbook: `docs/runbooks/drift-alert.md`.

## Rules

- Deterministic given inputs. No randomness without an explicit seed.
- Reports are plain dataclasses serialisable to JSON; they are the contract
  with the gate, the card and (later) dashboards.
