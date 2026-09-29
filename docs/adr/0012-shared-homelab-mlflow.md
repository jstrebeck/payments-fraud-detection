# ADR-0012: Development uses the homelab MLflow; local MLflow is optional

**Status:** Accepted
**Date:** 2026-09-29

## Context

Phase 1 shipped a local MLflow (plus Postgres database and S3 store) in
docker-compose, because the roadmap assumed MLflow would only appear in the
cluster in Phase 3. MLflow already runs in the homelab (`mlops` namespace,
shared with demand-forecast-mlops) and was upgraded to 3.16.1 on 2026-09-29
so 3.x clients can log models to it. It proxies artifacts
(`--serve-artifacts`), so clients need only a tracking URI.

Alternatives: keep a separate local registry for development and a cluster
registry for serving (two sources of truth, and model versions trained on a
laptop would have to be copied across), or drop the local MLflow entirely
(nobody without the homelab could run Phase 2).

## Decision

Development and the cluster use the same registry: the homelab MLflow at
`http://192.168.2.202` (in-cluster `http://mlflow.mlops.svc`). The compose
`mlflow` and `s3` services move behind the `local-mlflow` profile
(`make dev-mlflow`) for working without the homelab.

## Consequences

- One registry: a version trained on a laptop is the same version KServe can
  serve later. Phase 3's "training run from a laptop logs to the cluster
  MLflow" is already true.
- The registry is shared, so laptop experiments are visible to everything
  that reads it. Phase 2 must keep this safe: development runs log to the
  `fraud-detector-dev` experiment, and only `scripts/promote.py` (the gate)
  may set the `champion` alias on `fraud-detector`. This matters from Phase 4,
  when KServe serves whatever `champion` points at.
- Client and server major versions must match (MLflow 3.x); the `train`
  dependency group pins accordingly.
- KServe still cannot read artifacts from the MLflow PVC. That is the pending
  SeaweedFS move in the homelab repo and does not block Phase 2.
- `make dev` no longer starts MLflow, so it is faster and does not need the
  MLflow image built.
