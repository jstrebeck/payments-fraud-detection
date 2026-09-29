# payments-api

FastAPI service. Receives a transaction, computes features, asks a fraud
scorer, records the decision, returns it. The integration point for every
serving concern in this project (ADR-0001). No authentication: it is a demo.

## Interface

| Method | Path | Notes |
|---|---|---|
| `POST` | `/payments` | Body: `ml.data.schema.Transaction` (label fields are rejected with 422). Returns `{payment_id, transaction_id, decision, score, scorer, model_version}`. `201` when scored, `200` with the stored decision when the `transaction_id` was seen before (idempotent). |
| `GET` | `/payments/{payment_id}` | Decision record plus the transaction and the features used. |
| `POST` | `/payments/{payment_id}/feedback` | Phase 7. Body `{label: "fraud" \| "legit", source}`. |
| `GET` | `/healthz` | Process up. |
| `GET` | `/readyz` | `200` when the database answers. `scorer` is `ok`, or `fallback` while the primary scorer is unavailable and rules are serving; set `REQUIRE_SCORER=true` to make that a `503`. |
| `GET` | `/metrics` | Prometheus. Names in `docs/architecture.md` "Key metrics". |

Every response carries `x-request-id` (taken from the request if present).

## Configuration (environment)

| Variable | Default | |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://fraud:fraud@localhost:5432/payments` | In the cluster, from Secret `payments-db` |
| `FRAUD_SCORER` | `rule` | `rule`, `mlflow` (compose default), `kserve` (Phase 4; fails fast until then) |
| `REVIEW_THRESHOLD` / `DECLINE_THRESHOLD` | `0.5` / `0.8` | Decision policy for every scorer; the model card recommends model-specific values |
| `MLFLOW_TRACKING_URI` | MLflow default | `MlflowScorer` registry |
| `MODEL_NAME` / `MODEL_ALIAS` | `fraud-detector` / `champion` | |
| `MODEL_REFRESH_SECONDS` | `60` | How often to check whether the alias moved; `0` disables hot reload |
| `REQUIRE_SCORER` | `false` | Fail readiness while rules are standing in |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `console` | The image sets `LOG_FORMAT=json` |

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
    mlflow_model.py MlflowScorer: `MODEL_NAME@MODEL_ALIAS` in-process, feature_version check, hot reload
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
| `KServeScorer` | cluster (Phase 4) | from the InferenceService |

`MlflowScorer` loads the aliased version at startup and polls the alias every
`MODEL_REFRESH_SECONDS`; a promotion goes live without a restart and shows
up in `fraud_model_version_info`. It refuses versions whose `feature_version`
tag differs from this build's `ml.features.FEATURE_VERSION` (ADR-0013). While
no compatible model is loaded, requests are scored by rules and counted in
`fraud_scorer_fallback_total{reason="ModelUnavailableError"}`.

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
