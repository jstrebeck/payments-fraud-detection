# ADR-0011: SeaweedFS instead of MinIO for S3-compatible storage

**Status:** Accepted
**Date:** 2026-09-28

## Context

MLflow artifacts (and therefore the models KServe loads) need an
S3-compatible object store, locally in docker-compose and in the cluster. The
original plan (ADR-0004, homelab-integration) named MinIO. MinIO no longer
publishes community container images: `minio/minio` and `quay.io/minio/minio`
could not be pulled when Phase 1 was built. Pinning an old image would mean
running unpatched software.

Alternatives: build MinIO from source (maintenance burden), Garage, RustFS
(young), Ceph RGW via Rook (already in the cluster, heavy for a laptop), and
SeaweedFS (Apache-2.0, single container with an S3 gateway, versioned images).

## Decision

Use SeaweedFS's S3 gateway. Locally, `chrislusf/seaweedfs:4.48` runs as the
`s3` service in `docker-compose.yml` (profile `local-mlflow` since ADR-0012)
with a static identity file. In the
cluster, the homelab repo is replacing MinIO with SeaweedFS as part of its
own work; this repo only needs an S3 endpoint and credentials.

## Consequences

- Clients are unchanged: MLflow, boto3 and the KServe storage initializer all
  talk plain S3 (`MLFLOW_S3_ENDPOINT_URL`, `AWS_*` credentials).
- The local MLflow container creates the `mlflow-artifacts` bucket on start,
  so no separate bucket-init job is needed.
- ADR-0004 and ADR-0005 still say MinIO in their context; read that as "the
  S3 store". This ADR supersedes only the choice of product.
