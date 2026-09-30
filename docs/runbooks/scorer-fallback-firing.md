# Runbook: scorer fallback is firing

**Trigger:** `FraudScorerFallbackHigh` (more than 5% of payments decided by the
rule fallback for 2 minutes), or `FraudPaymentsLatencyP99High`, or the
"Scorer fallback share" stat on the **Fraud / payments-api** dashboard is red.

**Impact:** payments are still decided, but by `rules-v1`, which catches far
less fraud than the model. Nothing is lost; each payment records
`scorer = rule`.

## 1. Which failure?

The `reason` label is the error class the KServe scorer raised:

```
sum by (reason) (rate(fraud_scorer_fallback_total[5m]))
```

| reason | Meaning | Go to |
|---|---|---|
| `KServeTimeoutError` | The predictor did not answer within `KSERVE_TIMEOUT_SECONDS` (0.3s) | 2 |
| `KServeHTTPError` | Connection refused or non-2xx: predictor down, restarting or crashing | 2 |
| `KServeResponseError` | 2xx with an unexpected body: wrong runtime or model signature | 3 |
| `IncompatibleModelError` | The served version's `feature_version` differs from the API's, or no `model_version` in responses | 3 |
| `ModelVersionCheckError` | The API could not read the served version's tags from MLflow | 4 |

## 2. Predictor down or slow

```
kubectl -n fraud get isvc fraud-detector
kubectl -n fraud get pods -l serving.kserve.io/inferenceservice=fraud-detector
kubectl -n fraud logs deploy/fraud-detector-predictor -c storage-initializer | tail -3
kubectl -n fraud logs deploy/fraud-detector-predictor -c kserve-container --tail=50
```

- Stuck in `Init`: the MLflow storage initializer cannot resolve or download
  `models:/fraud-detector@champion`; its last log line says why. Check MLflow
  (`curl http://192.168.2.202/health`).
- `OOMKilled` or crash-looping: compare the model size with the runtime's
  memory limit in `deploy/base/serving-runtime.yaml`.
- Running but slow: MLServer p99 on the predictor pod (PodMonitor
  `fraud-detector`) versus the API's `fraud_scorer_latency_seconds`. A slow
  node or CPU throttling shows up in both.

## 3. Wrong model being served

```
kubectl -n fraud get isvc fraud-detector -o jsonpath='{.spec.predictor.annotations}'
kubectl -n fraud logs deploy/payments-api | grep -e incompatible_model -e model_version_changed | tail
```

The champion was trained on a different `feature_version` than the running
API computes. Either deploy the API that matches the model or roll the model
back: `docs/runbooks/rollback-model.md`.

## 4. MLflow unreachable from the API

The API checks each new served version's tags in MLflow once, then caches
the result. A failing check means a new version appeared while MLflow was
down.

```
kubectl -n fraud exec deploy/payments-api -- python -c \
  "import urllib.request; print(urllib.request.urlopen('http://mlflow.mlops.svc/health').read())"
```

It recovers by itself once MLflow answers; no restart needed.

## Verify

The fallback share on the dashboard drops to 0, `FraudScorerFallbackHigh`
resolves (Prometheus `ALERTS{alertname="FraudScorerFallbackHigh"}` disappears),
and new rows in `payments` have `scorer = kserve`.

## Drill (2026-09-30)

Exercised end to end through GitOps, as a Phase 6 acceptance test:

| Time (UTC) | Event |
|---|---|
| 05:02 | `3491307 chore(drill)`: `KSERVE_TIMEOUT_SECONDS=0.001` in `deploy/overlays/homelab`, so every model call times out |
| 05:04 | Argo CD rolled it out; alert `pending` |
| 05:06 | `FraudScorerFallbackHigh` **firing** (72% of payments on the rule fallback), received by Alertmanager |
| 05:07 | `816e6d1` reverts the drill commit |
| 05:14 | Alert **resolved**; fallback ratio back to 0 |

Payments kept being decided throughout (by `rules-v1`). The drill also
caught a false positive: `FraudModelVersionChanged` fired because a starting
API pod reports version `unknown`; fixed in `a148e38`.

Alertmanager routes these to the homelab's `null` receiver today, so they
are visible in the Prometheus and Alertmanager UIs (and as annotations on the
fraud dashboards) but page nobody. Add a receiver in the homelab repo's
kube-prometheus-stack values to get notified.

To repeat it: the same two commits (set the timeout, then `git revert`),
while watching `ALERTS{alertname="FraudScorerFallbackHigh"}`.
