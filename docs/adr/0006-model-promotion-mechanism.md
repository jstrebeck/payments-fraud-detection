# ADR-0006: How a promoted model reaches the InferenceService

**Status:** Accepted
**Date:** 2026-09-28 (proposed), 2026-09-30 (accepted)

## Context

After the gate moves the `champion` alias, the running predictor must pick
up the new version. Candidate mechanisms:

1. `storageUri: models:/fraud-detector@champion`, resolved by the storage
   initializer at pod start; promotion then restarts the predictor.
2. Promote script resolves the alias to a concrete `s3://` URI and patches
   the `InferenceService` (drift between Git and cluster; Argo CD would
   revert it unless the field is ignored).
3. Promote script writes the resolved URI into `deploy/overlays/homelab`
   and commits; Argo CD deploys it. Fully GitOps, but every promotion
   becomes a Git commit made by a job, and automated retraining (Phase 7)
   needs a bot with push rights.

Findings while building option 1:

- KServe's built-in storage initializer does not understand `models:/`.
  KServe does let a `ClusterStorageContainer` claim a URI prefix and supply
  its own init container.
- The MLflow Python client's parallel downloader stalled against the
  homelab server and left zero-filled files, so the initializer should not
  use it.
- `kubectl rollout restart` on the predictor Deployment does nothing: the
  KServe controller owns the Deployment and reverts the restart annotation
  within seconds, leaving the old pod in place.
- MLServer reports a model version in V2 responses only if its model
  settings carry one.

## Decision

Option 1.

- The homelab repo provides a `ClusterStorageContainer` named `mlflow` for
  the `models:/` prefix (`Kubernetes/kserve/mlflow-storage-initializer`). Its
  image, standard library only, resolves the alias over MLflow's REST API,
  downloads the files through the artifact proxy (no S3 credentials) with
  size checks, and writes `model-settings.json` with the registry version.
  Every V2 response then carries `model_version`, which the API records per
  payment and checks against the model's `feature_version` tag (ADR-0013).
- `deploy/base/inferenceservice.yaml` uses
  `storageUri: models:/fraud-detector@champion`.
- `scripts/promote.py` moves the alias and then sets the predictor pod
  annotation `fraud-detection/model-version: <n>` on the InferenceService,
  which KServe turns into a rolling update. `--rollout-only` repeats just
  that step.

## Consequences

- Git holds no model-version churn. "What is live" is answered by the
  `champion` alias, the `fraud-detection/model-version` annotation, and the
  API's `fraud_model_version_info` metric.
- Promotion and rollback take one command and about a minute (see
  `docs/runbooks/rollback-model.md`).
- The annotation is not declared in Git, so Argo CD (Phase 5) leaves it
  alone: its diff only covers fields the manifests set. Verified on the
  cluster with self-heal on. If `inferenceservice.yaml` ever declares
  `spec.predictor.annotations`, add an `ignoreDifferences` entry for
  `/spec/predictor/annotations/fraud-detection~1model-version`.
- Any predictor pod that starts for another reason (node drain, HPA scale
  up) resolves the alias at that moment. If the alias moved without a
  rollout, replicas can briefly serve different versions. The API records
  the version that produced each score, so this stays visible.
- The predictor depends on MLflow being reachable at pod start. Running
  pods are unaffected by an MLflow outage; the API's rule fallback covers
  a predictor that cannot start.
