# Runbook: roll back the served model

**Trigger:** the current `champion` misbehaves in production: a jump in
declines or reviews, an alert on the score distribution, or a bad model
card after the fact.

**Target:** back on the previous version in under five minutes.

## Preconditions

- `kubectl` context with access to namespace `fraud`
- `MLFLOW_TRACKING_URI=http://192.168.2.202` (the Makefile sets it)
- Know which version to go back to. The gate keeps the prior champion as the
  `previous` alias:

```
uv run python -c "from mlflow import MlflowClient as C; c=C(); \
  print({a: c.get_model_version_by_alias('fraud-detector', a).version for a in ('champion','previous')})"
```

## Steps

1. Point `champion` at the good version (here `N`):

   ```
   uv run python -c "from mlflow import MlflowClient as C; \
     C().set_registered_model_alias('fraud-detector', 'champion', 'N')"
   ```

2. Roll the predictor so new pods resolve the alias:

   ```
   uv run python scripts/promote.py --rollout-only
   ```

   This patches the InferenceService annotation
   `fraud-detection/model-version` to `N` and waits for the rolling update
   and `Ready`. Do not use `kubectl rollout restart`: KServe reverts it.

## Verify

```
kubectl -n fraud logs deploy/fraud-detector-predictor -c storage-initializer | tail -1
#   {"event": "done", "name": "fraud-detector", "version": "N"}
make smoke-cluster        # the POST /payments response shows model_version fraud-detector/N
```

In Prometheus, `fraud_model_version_info{version="fraud-detector/N"}` is 1,
and `fraud_scorer_fallback_total` has not started climbing.

## If the predictor will not start

The pod stays in `Init`. Read the reason:

```
kubectl -n fraud logs <predictor-pod> -c storage-initializer
```

Meanwhile the API keeps answering with the rule scorer (counted in
`fraud_scorer_fallback_total`), so payments are still decided. Fix MLflow
reachability or the alias, then rerun step 2.

## Automated retraining

The `retrain` CronJob (ADR-0015) may have made the promotion you are rolling
back. After a retrain it waits 6 hours before retraining again, so it will
not immediately re-promote; if drift keeps firing it will retrain after the
cooldown and promote only through the gate. To stop it while you
investigate: `kubectl -n fraud patch cronjob retrain -p '{"spec":{"suspend":true}}'`
(Argo CD self-heal reverts that within minutes; for longer, set
`suspend: true` in `deploy/base/retrain/cronjob.yaml` through Git).

## Roll forward again

Once the fix is trained and registered, promote it through the gate as usual
(`make promote`); that moves `champion` and rolls the predictor in one step.
