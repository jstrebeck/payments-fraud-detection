# Local development

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.11+ (it installs Python 3.12 if needed)
- Docker with Compose v2

## First run

```
make install                 # uv sync + pre-commit hooks
make check                   # lint, mypy, tests: what CI runs
make dev                     # postgres + API, wait until healthy
make smoke                   # health, readiness, one payment end to end
make train                   # generate data if needed, train, register a version in MLflow
make promote                 # gate it; moves `champion` on a win
make simulate                # 60 s of synthetic traffic at 20 rps, scored by the champion
make psql                    # look at the decisions:  select decision, count(*) from payments group by 1;
curl -s localhost:8000/metrics | grep ^fraud_
make down                    # stop (keeps volumes); make clean-dev also deletes them
```

## Stack (docker-compose)

| Service | Port | Purpose |
|---|---|---|
| `postgres` | 5432 | `payments` db (decision log); also the `mlflow` db for the local MLflow |
| `s3` | 8333 | Profile `local-mlflow` only. SeaweedFS S3 gateway (ADR-0011), bucket `mlflow-artifacts` |
| `mlflow` | 5000 | Profile `local-mlflow` only. Tracking server + registry, artifacts proxied to `s3` |
| `migrate` | none | One-shot `payments-api-migrate`; the API waits for it |
| `payments-api` | 8000 | The API (`/docs` for OpenAPI) |
| `simulator` | none | Profile `sim`, runs on demand via `make simulate` |

MLflow is the homelab server by default (`http://192.168.2.202`, MLflow
3.16.1; ADR-0012). It proxies artifacts, so setting `MLFLOW_TRACKING_URI`
is all a client needs. Without homelab access, `make dev-mlflow` also starts
a local MLflow and S3 store; point `MLFLOW_TRACKING_URI` at
`http://localhost:5000`. Keep the client on the server's major version
(MLflow 3.x).

The compose API runs `FRAUD_SCORER=mlflow`: it loads `fraud-detector@champion`
from that MLflow, reloads within 30 s of a promotion, and scores with rules
while there is no champion (`/readyz` then reports `"scorer": "fallback"`).
`make dev API_FRAUD_SCORER=rule` forces rules. KServe is not part of the
local stack.

The compose API runs the built image, without hot reload. For hot reload,
`docker compose stop payments-api` and `make api` (runs uvicorn on the host
against the compose Postgres).

## Make targets

```
make install     uv sync --all-packages, pre-commit install
make lint        ruff check, ruff format --check, mypy --strict per member
make fmt         ruff format + ruff check --fix
make test        pytest with coverage
make check       lint + test
make dev         docker compose up (postgres, migrate, api) and wait for health
make dev-mlflow  same plus a local MLflow and S3 store (profile local-mlflow)
make down        stop the stack; clean-dev also removes volumes
make logs        follow API logs
make psql        psql into the payments database
make generate    ml.data CLI -> data/transactions.parquet (SEED=42)
make simulate    simulator in compose against the API (SEED, SIM_ARGS="--rps 20 --duration 60s")
make api         API on the host with --reload
make smoke       scripts/smoke.sh against API_URL (default http://localhost:8000)
make build       build service images tagged with the git SHA
make push        push them to 192.168.2.203:5000/fraud (Phase 5 automates this)
make train       train on data/transactions.parquet, register in MLflow (generates the data if missing)
make promote     gate the latest version; PROMOTE_ARGS="--version 7 --dry-run"
```

## Environment

Compose sets everything it needs. `.env` is only read by tools you run on
the host (`make api`, the Phase 2 trainer). Copy `.env.example` to `.env`.
Never commit `.env`.

## Pointing local tools at the homelab

`.env.example` already points `MLFLOW_TRACKING_URI` at the homelab MLflow.
Endpoints are listed in `docs/homelab-integration.md`. Development runs log
to the `fraud-detector-dev` experiment; only the promotion gate moves the
`champion` alias (ADR-0012).
