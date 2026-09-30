# scripts

Small, single-purpose helpers. Each has a header comment saying what it
does and which phase introduced it. Prefer a Makefile target that calls the
script over asking people to remember arguments.

| Script | Phase | Make target | Does |
|---|---|---|---|
| `smoke.sh [url]` | 1 | `make smoke` | `/healthz`, `/readyz`, one `POST /payments`, read it back, check `/metrics`. `EXPECT_SCORER=<name>` also requires that scorer and a `fraud-detector/<n>` version |
| `smoke-cluster.sh [namespace]` | 4 | `make smoke-cluster` | Port-forwards `svc/payments-api` (default namespace `fraud`), runs `smoke.sh` with `EXPECT_SCORER=kserve`, stops the port-forward on exit |
| `promote.py [--version N] [--dry-run] [--strict]` | 2 | `make promote` | Run the gate (`ml.evaluation.promote`) for a registered version, move `champion` on a win, print before/after. Serving refresh: the compose API reloads within `MODEL_REFRESH_SECONDS`; KServe in Phase 4 (ADR-0006) |

Planned:

- `demo.sh` (Phase 8): generate, train, promote, simulate, show URLs.

Database and bucket setup for the local stack is not a script: the compose
Postgres creates the `mlflow` database from `docker/postgres/init.sql`, and
the MLflow container creates its bucket on start.
