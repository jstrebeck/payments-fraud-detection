# Architecture

## Purpose

Score card payments for fraud in real time and keep the model healthy over
time with as little manual work as possible. The business logic is
deliberately thin; the interesting part is the machinery around the model.

## Components

| Component | Where | Runtime | Responsibility |
|---|---|---|---|
| Synthetic generator | `ml/data` | library + CLI | Deterministic customers, merchants, cards, transactions, injected fraud patterns. Single source of truth for data. |
| Feature library | `ml/features` | library | Pure functions: transaction (+ rolling context) to feature vector. Shared by training and serving. |
| Training | `ml/training` | K8s Job / CronJob image | Featurise with `ml.features`, train LightGBM, evaluate, log to MLflow, register. |
| Evaluation and gate | `ml/evaluation`, `scripts/promote.py` | in training image | Champion vs challenger; set `champion` alias only on a win. |
| MLflow | homelab repo | Deployment | Tracking server + model registry. Postgres backend, S3 artifact store (SeaweedFS, ADR-0011). |
| KServe | homelab repo (install), `deploy/` (InferenceService) | RawDeployment | Serves the `champion` model version using the MLflow/MLServer runtime, V2 inference protocol. |
| payments-api | `services/payments-api` | Deployment | `POST /payments`: validate, featurise, score via KServe, persist decision, respond. |
| simulator | `services/simulator` | Deployment (cluster) / CLI (local) | Drives traffic and delayed label feedback. |
| Postgres (payments) | homelab repo | StatefulSet | Decision log and labels. Source for retraining. |
| Drift monitor | `ml/evaluation/drift_monitor.py` | Deployment (exporter) | Every 5 minutes, PSI of the last hour of live feature vectors against the champion's training profile; Prometheus metrics. A long-running exporter rather than a CronJob, so Prometheus can scrape it without a Pushgateway. |
| Prometheus / Grafana | homelab repo (stack), `dashboards/` (this repo) | existing | Metrics, dashboards, alerts. |
| Argo CD | homelab repo | existing after Phase 5 | Syncs `deploy/overlays/homelab` into namespace `fraud`. |
| GitHub Actions | `.github/workflows` | self-hosted runner VM | Lint, test, build, push, bump tag, trigger training. |

## Request path (online)

1. Simulator sends `POST /payments` with a transaction (card token, amount,
   merchant, timestamp, device, geo, channel).
2. API validates with Pydantic, loads the card's recent history from
   Postgres (last 7 days, at most 100 rows; ADR-0010), and calls
   `ml.features.build_features`.
3. API calls the `FraudScorer`. In the cluster this is `KServeScorer`, which
   POSTs to the `InferenceService` V2 endpoint with a 300 ms timeout. The
   response names the registry version that scored it; the first time a
   version appears, its `feature_version` tag is checked against the API's
   (ADR-0013). On timeout, error or an incompatible version it falls back to
   `RuleScorer` and increments `fraud_scorer_fallback_total{reason}`.
4. Decision policy maps score to `approved | review | declined` using
   thresholds held in config (so they can be tuned without a retrain).
5. Decision, score, model version and features are written to Postgres.
   Response returns the decision and a payment ID.
6. Later, the simulator posts `POST /payments/{id}/feedback` with
   `fraud | legit` for a sample, simulating chargebacks. Labels land in the
   same table.

## Tracing a payment

One correlation ID follows a payment through every hop:

1. **simulator** generates it (uuid4 hex) and sends it as `x-request-id`; it
   logs it on `payment_failed` and on `payment_outcome` for flagged or
   fraudulent payments.
2. **payments-api** keeps a well-formed caller ID (or makes one), binds it to
   structlog, so every API log line for the request (`request`,
   `payment_scored`, `scorer_failed_using_fallback`) has `request_id`, and
   echoes it in the response header.
3. **KServeScorer** sends it to the predictor as the V2 request `id` and as
   `x-request-id`. MLServer echoes the `id` in its response. MLServer's own
   access log does not print it, so that hop is matched by time.
4. **Postgres**: stored as `payments.request_id` (indexed) and returned by
   `GET /payments/{id}`.

```
kubectl -n fraud logs deploy/simulator | grep '"decision": "declined"' | tail -1   # pick a request_id
kubectl -n fraud logs deploy/payments-api -c payments-api | grep <request_id>
kubectl -n fraud exec payments-postgres-0 -- psql -U payments -d payments \
  -c "select payment_id, decision, score, scorer, model_version from payments where request_id = '<request_id>'"
```

Step by step: `docs/runbooks/trace-a-payment.md`.

## Training path (offline)

1. Training image starts as a Kubernetes Job (manual, scheduled, or from a
   GitHub workflow dispatch).
2. Data source is either generator output (Phases 1 to 6) or labelled
   decisions from Postgres (Phase 7). Both produce the same Parquet schema.
3. Features computed with the same `ml.features` code the API uses, then a
   time-based split. LightGBM is trained on the feature vector; the
   registered model takes named features and returns `predict_proba`, and is
   tagged with its `feature_version` (ADR-0013).
4. Metrics, parameters, evaluation report, model card and the model are
   logged to MLflow. The model is registered as `fraud-detector`.
5. The gate compares the new version to the `champion` alias on the most
   recent held-out window. Win means the alias moves; otherwise the version
   stays registered but unaliased.
6. The `InferenceService` references the model by alias-resolved URI. Moving
   the alias plus a rollout (or a KServe model reload) puts the new version
   into service. Exact mechanism is decided in Phase 4 and recorded in
   ADR-0006.

## Delivery path

1. PR opened. CI runs lint, type checks, tests on the self-hosted runner.
2. Merge to `main`. Build workflow builds `payments-api`, `simulator`,
   `trainer` images, tags them with the git SHA, pushes to the in-cluster
   registry.
3. Workflow updates `deploy/overlays/homelab/kustomization.yaml` image tags
   and commits (or opens an auto-merged PR).
4. Argo CD notices the change and syncs namespace `fraud`.

## Namespaces and naming

- Namespace: `fraud`
- Model name in MLflow registry: `fraud-detector`; aliases `champion` and
  `challenger`
- InferenceService: `fraud-detector`
- Images: `192.168.2.203:5000/fraud/payments-api`, `.../fraud/simulator`,
  `.../fraud/trainer`
- Metric prefix: `fraud_`

## Key metrics

| Metric | Type | Why |
|---|---|---|
| `fraud_payments_total{decision}` | counter | Decision mix; sudden change means model or traffic changed |
| `fraud_score` | histogram | Score distribution; drift signal without labels |
| `fraud_scorer_latency_seconds{scorer}` | histogram | Model latency budget |
| `fraud_scorer_fallback_total{reason}` | counter | Serving health; `reason` is the error class (`KServeTimeoutError`, `KServeHTTPError`, `IncompatibleModelError`, ...) |
| `fraud_model_version_info{scorer,version}` | gauge | Which version is live (for KServe: the last version seen in a verified response) |
| `fraud_payments_replayed_total` | counter | Idempotent replays of an already-scored transaction |
| `fraud_http_requests_total{method,route,status}`, `fraud_http_request_duration_seconds` | counter, histogram | RED metrics for every route |
| `fraud_sim_decisions_total{decision,truth}` | counter | Simulator side: live confusion counts against ground truth |
| `fraud_registry_alias_version{alias}`, `fraud_registry_version_metric{version,metric}`, `fraud_training_last_run_timestamp_seconds{experiment,status}` | gauge | MLflow registry and training runs, via the registry exporter (`ml/evaluation/exporter.py`); the training dashboard |
| `fraud:flagged_share:ratio_rate1h` / `_rate24h`, `fraud:scorer_fallback:ratio_rate5m`, `fraud:payments_post_latency_seconds:p99_5m` | recording rules | Inputs to the `Fraud*` alerts (`deploy/base/prometheusrules.yaml`) |
| `fraud_feature_psi{feature}` | gauge | Drift per feature: PSI of the last hour of payments against the champion's training profile (drift monitor, calendar features excluded) |
| `fraud_drift_window_rows`, `fraud_drift_reference_version`, `fraud_drift_up`, `fraud_drift_last_run_timestamp_seconds` | gauge | Drift monitor health: window size (PSI needs 300), which champion's profile, last computation |
| `fraud:feature_psi:max`, `fraud:feature_psi:baseline_24h`, `fraud:feature_psi:drifting` | recording rules | Per-feature PSI, its 24h median, and features above both 0.25 and twice their median: inputs to `FraudFeatureDrift` |
| `fraud_labels_total{label,decision}` | counter | Delayed labels by label and the decision originally made. Precision = `sum(rate(fraud_labels_total{label="fraud",decision!="approved"}[1h])) / sum(rate(fraud_labels_total{decision!="approved"}[1h]))`; recall = flagged fraud labels over all fraud labels. Legit labels are a sample, so precision is an estimate. |
| `fraud_label_delay_seconds` | histogram | Time from scoring to label (chargeback delay) |
| `fraud_sim_feedback_total{label,outcome}` | counter | Simulator side: labels scheduled, sent, errored, dropped |

## Data schema (transactions)

Defined once in `ml/data/schema.py` as a Pydantic model, imported everywhere
else. The matching PyArrow schema for Parquet is in `ml/data/io.py` (kept
separate so the API does not need pyarrow). Fields (initial): `transaction_id`, `timestamp`,
`card_token`, `customer_id`, `merchant_id`, `merchant_category`, `amount`,
`currency`, `channel` (`card_present | ecommerce | recurring`),
`device_id`, `ip_country`, `billing_country`, `is_fraud` (training only),
`fraud_pattern` (training only, for per-pattern recall).

## Non-goals

- Real PCI-scope handling. Cards are opaque tokens.
- A UI. Grafana and MLflow are the UI.
- Multi-cluster or cloud deployment. The homelab is the target; the Terraform
  for EKS in the homelab repo is unrelated.
