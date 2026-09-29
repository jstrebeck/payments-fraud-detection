# docker

Support files for the local `docker-compose.yml` stack only. Nothing here is
deployed to the cluster; the cluster's MLflow, Postgres and S3 store come
from the homelab repo (ADR-0002). Service images for this project's own code
live next to the code (`services/*/Dockerfile`).

The `s3` and `mlflow` services run only with the `local-mlflow` compose
profile (`make dev-mlflow`); by default development uses the homelab MLflow
(ADR-0012).

| Path | Used by | Purpose |
|---|---|---|
| `postgres/init.sql` | `postgres` | Creates the `mlflow` database on first start (`payments` comes from `POSTGRES_DB`) |
| `seaweedfs/s3.json` | `s3` | Static S3 identity for local dev (ADR-0011). Throwaway credentials. |
| `mlflow/Dockerfile` | `mlflow` | Upstream MLflow image plus `psycopg2` and `boto3` |
| `mlflow/start.sh` | `mlflow` | Creates the `mlflow-artifacts` bucket if missing, then `mlflow server` with proxied artifacts |
