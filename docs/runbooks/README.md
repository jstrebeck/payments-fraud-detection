# Runbooks

One file per operational procedure, written so that someone who did not
build the system can follow it. Each runbook has: symptom or trigger,
preconditions, steps with exact commands, verification, rollback.

Planned (create as the corresponding phase lands):

| Runbook | Phase | Trigger |
|---|---|---|
| [`rollback-model.md`](rollback-model.md) | 4 | Bad model in production; repoint `champion`, `promote.py --rollout-only`, verify version metric |
| [`trace-a-payment.md`](trace-a-payment.md) | 6 | Explain one decision: follow its `request_id` from simulator to API logs, predictor and Postgres |
| `rollback-api.md` | 5 | Bad API release; revert tag-bump commit, Argo sync |
| [`scorer-fallback-firing.md`](scorer-fallback-firing.md) | 6 | `FraudScorerFallbackHigh`: triage by fallback `reason`, predictor, storage initializer, MLflow |
| `manual-training-run.md` | 5 | Run training outside the schedule; dispatch workflow or apply Job |
| [`drift-alert.md`](drift-alert.md) | 7 | `FraudFeatureDrift` / `FraudDriftMonitorDown`: read PSI against each feature's baseline, decide retrain or threshold change, backfill a missing reference profile |
| `rebuild-after-cluster-reset.md` | 5 | Homelab rebuilt; order of operations across both repos |
