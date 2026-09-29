# ADR-0007: Kubernetes Jobs and CronJobs before any workflow engine

**Status:** Accepted
**Date:** 2026-09-28

## Context

Training, evaluation, promotion, drift checks are batch steps. Kubeflow
Pipelines, Argo Workflows, Prefect, Dagster and Airflow all orchestrate such
steps, at the cost of another platform component and its learning curve.

## Decision

Each batch step is a container entrypoint. Training plus evaluation plus the
gate run as **one** Kubernetes `Job` (a single process with clear stages and
MLflow as the record). Scheduled retraining and drift checks are
`CronJob`s. GitHub Actions `workflow_dispatch` is the manual trigger.

## Consequences

- Nothing new to install for Phases 2 to 7.
- When steps need fan-out, retries per step or a DAG view, adopt Argo
  Workflows (it fits the Argo CD choice) and supersede this ADR. Do not
  adopt Kubeflow.
- Job manifests live in `deploy/jobs/` and are applied by CI or by a
  `CronJob`, not by Argo CD sync (they are not long-lived).
