# scripts

Small, single-purpose helpers. Each has a header comment saying what it
does and which phase introduced it. Prefer a Makefile target that calls the
script over asking people to remember arguments.

| Script | Phase | Make target | Does |
|---|---|---|---|
| `smoke.sh [url]` | 1 | `make smoke` | `/healthz`, `/readyz`, one `POST /payments`, read it back, check `/metrics`. `EXPECT_SCORER=<name>` also requires that scorer and a `fraud-detector/<n>` version |
| `smoke-cluster.sh [namespace]` | 4 | `make smoke-cluster` | Port-forwards `svc/payments-api` (default namespace `fraud`), runs `smoke.sh` with `EXPECT_SCORER=kserve`, stops the port-forward on exit |
| `promote.py [--version N] [--dry-run] [--strict] [--no-rollout] [--rollout-only]` | 2, 4 | `make promote` | Run the gate (`ml.evaluation.promote`) for a registered version, move `champion` on a win, then roll the KServe predictor via the `fraud-detection/model-version` annotation (ADR-0006). `--rollout-only` just rolls to the current champion (rollback runbook). The compose API reloads within `MODEL_REFRESH_SECONDS` |
| `release.sh` | 5 | `make release` | Build and push payments-api, simulator and trainer tagged with HEAD's short SHA; pin the SHA in `deploy/overlays/homelab`. `make release` also commits and pushes, which Argo CD deploys. Also run by `build-push.yml` |
| `train-cluster.sh` | 5 | `make train-cluster` | Build/push `trainer:<sha>` if missing, create a Job from `deploy/jobs/train.yaml` (`SEED`, `CUSTOMERS`, `DAYS`), follow its logs, exit with its result. Also run by `train.yml` |

Planned:

- `demo.sh` (Phase 8): generate, train, promote, simulate, show URLs.

Database and bucket setup for the local stack is not a script: the compose
Postgres creates the `mlflow` database from `docker/postgres/init.sql`, and
the MLflow container creates its bucket on start.
