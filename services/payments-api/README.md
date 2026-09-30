# payments-api

FastAPI service. Receives a transaction, computes features, asks a fraud
scorer, records the decision, returns it. The integration point for every
serving concern in this project (ADR-0001). No authentication: it is a demo.

## Interface

| Method | Path | Notes |
|---|---|---|
| `POST` | `/payments` | Body: `ml.data.schema.Transaction` (label fields are rejected with 422). Returns `{payment_id, transaction_id, decision, score, scorer, model_version}`. `201` when scored, `200` with the stored decision when the `transaction_id` was seen before (idempotent). |
| `GET` | `/payments/{payment_id}` | Decision record plus the transaction, the features used, the `request_id` that created it, and its label if feedback arrived (`label`, `label_reason`, `label_source`, `labelled_at`). |
| `POST` | `/payments/{payment_id}/feedback` | Delayed label. Body `{label: "fraud" \| "legit", reason, source}`: `reason` (chargeback reason, e.g. the fraud pattern) only with `fraud`; `source` like `simulator`, `chargeback`, `manual`. Last write wins; an identical re-post changes nothing. Returns the record. `404` for an unknown payment. |
| `GET` | `/healthz` | Process up. |
| `GET` | `/readyz` | `200` when the database answers. `scorer` is `ok`, or `fallback` while the primary scorer is unavailable and rules are serving; set `REQUIRE_SCORER=true` to make that a `503`. |
| `GET` | `/metrics` | Prometheus. Names in `docs/architecture.md` "Key metrics". |

Every response carries `x-request-id`: the caller's value if it is 1 to 64
characters of `[A-Za-z0-9._:-]`, otherwise a fresh one. That correlation ID is
bound to every log line for the request, sent to the KServe predictor (V2
request `id` and `x-request-id` header), and stored on the payment as
`request_id` (`POST` and `GET` responses include it; a replayed submission
returns the original request's ID). See `docs/runbooks/trace-a-payment.md`.

## Configuration (environment)

| Variable | Default | |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://fraud:fraud@localhost:5432/payments` | In the cluster, from Secret `payments-db` |
| `FRAUD_SCORER` | `rule` | `rule`, `mlflow` (compose default), `kserve` (cluster) |
| `REVIEW_THRESHOLD` / `DECLINE_THRESHOLD` | `0.5` / `0.8` | Decision policy for the rule fallback and for model versions without recommended thresholds. A served KServe version tagged `recommended_review_threshold` / `recommended_decline_threshold` (every version trained since Phase 7, and v2 by backfill) is decided with its own thresholds (ADR-0015) |
| `KSERVE_URL` | `http://fraud-detector-predictor.fraud.svc/v2/models/fraud-detector/infer` | `KServeScorer` V2 endpoint; `/ready` is derived from it |
| `KSERVE_TIMEOUT_SECONDS` | `0.3` | Whole predictor request budget; past it the payment is scored by rules |
| `MLFLOW_TRACKING_URI` | MLflow default | `MlflowScorer` registry; `KServeScorer` feature-version lookups |
| `MODEL_NAME` / `MODEL_ALIAS` | `fraud-detector` / `champion` | |
| `MODEL_REFRESH_SECONDS` | `60` | How often to check whether the alias moved; `0` disables hot reload |
| `REQUIRE_SCORER` | `false` | Fail readiness while rules are standing in |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `console` | The image sets `LOG_FORMAT=json` |

## Labels (Phase 7)

Labels arrive later than decisions: chargebacks for fraud, confirmations for
a sample of legitimate payments (the simulator imitates both). They are
stored on the payment row (`label`, `label_reason`, `label_source`,
`labelled_at`; migration `0003`, with a partial index on `created_at` for
labelled rows) and are the training data for retraining. Each accepted label
counts in `fraud_labels_total{label, decision}` (the decision originally
made), and the wait in `fraud_label_delay_seconds`. Precision and recall of
the served decisions are PromQL over that counter, e.g. recall =
flagged fraud labels / all fraud labels.

## Layout

```
payments_api/
  main.py          app factory, request middleware (request id, access log, RED metrics)
  routes.py        HTTP handlers; thin
  service.py       context -> features -> score -> decision -> persist; scorer fallback
  repository.py    all SQL: lookups, idempotency, card history window
  models.py        SQLAlchemy `payments` table
  schemas.py       response models
  policy.py        score -> approved | review | declined
  scoring/
    base.py        FraudScorer protocol, ScoreResult
    rule.py        RuleScorer: additive red flags; baseline and permanent fallback
    errors.py      ModelUnavailableError, IncompatibleModelError (names are fallback `reason` labels)
    mlflow_model.py MlflowScorer: `MODEL_NAME@MODEL_ALIAS` in-process, feature_version check, hot reload
    kserve.py      KServeScorer: V2 call to the InferenceService, feature_version check per served version
  metrics.py       prometheus-client objects
  config.py        pydantic-settings
  logs.py          structlog (JSON or console)
  migrate.py       `payments-api-migrate` entry point
  migrations/      Alembic; env.py reads DATABASE_URL
tests/             SQLite-backed API contract tests, including feature parity with ml.features.batch
alembic.ini        for `uv run alembic revision --autogenerate` during development
Dockerfile
```

## Scorers

| Scorer | Where | Model version reported |
|---|---|---|
| `RuleScorer` | always available; the fallback | `rules-v1` |
| `MlflowScorer` | compose and local (Phase 2) | `fraud-detector/<n>` |
| `KServeScorer` | cluster | `fraud-detector/<n>`, from the V2 response's `model_version` |

`MlflowScorer` loads the aliased version at startup and polls the alias every
`MODEL_REFRESH_SECONDS`; a promotion goes live without a restart and shows
up in `fraud_model_version_info`. It refuses versions whose `feature_version`
tag differs from this build's `ml.features.FEATURE_VERSION` (ADR-0013). While
no compatible model is loaded, requests are scored by rules and counted in
`fraud_scorer_fallback_total{reason="ModelUnavailableError"}`.

`KServeScorer` sends the feature vector as one named `FP64` input per
feature and reads the fraud probability from column 1 of `predict_proba`.
The storage initializer that resolved `models:/fraud-detector@champion`
tells MLServer the registry version (ADR-0006), so every response says which
version scored it. The first time a version is seen, its `feature_version`
tag is read from MLflow (bounded to 2 s, shared by concurrent requests) and
compared with this build's; a mismatch or a missing version is refused and
remembered. Lookup failures are not cached. A promotion therefore needs no
API change: the predictor restarts, the new version appears in responses,
and `fraud_model_version_info` follows. Failures map to fallback reasons
`KServeTimeoutError`, `KServeHTTPError` (unreachable or non-2xx),
`KServeResponseError`, `IncompatibleModelError` and `ModelVersionCheckError`.
`/readyz` reports `fallback` while the predictor's `/ready` does not answer 200.

## Rules

- Feature computation is imported from `ml.features`; never reimplemented here.
- Scorer failures never fail the request. Fall back to `RuleScorer`, count it
  in `fraud_scorer_fallback_total{reason}`, log it.
- The model version that produced a score is stored with the decision and
  exported in `fraud_model_version_info`.
- Schema changes go through an Alembic migration; `tests` run migrations, and
  `alembic check` must report no drift.

## Running

- In compose: `make dev` (runs the `migrate` one-shot, then the API on :8000).
- On the host with hot reload: `make dev`, `docker compose stop payments-api`,
  then `make api`.
- Tests: `uv run pytest services/payments-api`.

## Container image (workspace pattern, ADR-0009)

Built from the repo root: `docker build -f services/payments-api/Dockerfile .`

1. Copy `pyproject.toml`, `uv.lock` and every member's `pyproject.toml`
   (uv needs all members to resolve the workspace).
2. `uv sync --frozen --no-dev --no-install-workspace --package payments-api`
   installs third-party dependencies only; this layer is cached until the
   lockfile changes.
3. Copy only `ml/` and this service, then `uv sync --no-editable` installs
   them into the venv.
4. The runtime stage adds `libgomp1` (LightGBM), copies just the venv onto
   `python:3.12-slim`, runs as uid 10001, and starts `uvicorn --factory`.

The simulator image follows the same pattern. Copy it for new images.
