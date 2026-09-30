# Runbooks

One file per operational procedure, written so that someone who did not
build the system can follow it. Each runbook has: symptom or trigger,
preconditions, steps with exact commands, verification, rollback.

Planned (create as the corresponding phase lands):

| Runbook | Phase | Trigger |
|---|---|---|
| [`rollback-model.md`](rollback-model.md) | 4 | Bad model in production; repoint `champion`, `promote.py --rollout-only`, verify version metric |
| `rollback-api.md` | 5 | Bad API release; revert tag-bump commit, Argo sync |
| `scorer-fallback-firing.md` | 6 | `fraud_scorer_fallback_total` alert; check InferenceService, storage initializer, S3 store |
| `manual-training-run.md` | 5 | Run training outside the schedule; dispatch workflow or apply Job |
| `drift-alert.md` | 7 | `fraud_feature_psi` alert; inspect report, decide retrain or threshold change |
| `rebuild-after-cluster-reset.md` | 5 | Homelab rebuilt; order of operations across both repos |
