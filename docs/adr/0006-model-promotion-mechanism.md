# ADR-0006: How a promoted model reaches the InferenceService

**Status:** Proposed (decide in Phase 4)
**Date:** 2026-09-28

## Context

After the gate moves the `champion` alias, the running predictor must pick
up the new version. Candidate mechanisms:

1. `storageUri: models:/fraud-detector@champion` with KServe's MLflow-aware
   storage initializer resolving the alias at pod start; promotion then
   triggers a `rollout restart`.
2. Promote script resolves the alias to a concrete `s3://` URI and patches
   the `InferenceService` (drift between Git and cluster; Argo CD would
   revert it unless the field is ignored).
3. Promote script writes the resolved URI into
   `deploy/overlays/homelab` and commits; Argo CD deploys it. Fully GitOps,
   but a model promotion becomes a Git commit made by a job.

## Decision

Deferred. Try option 1 first because it keeps Git free of model-version
churn and keeps "what is live" answerable from MLflow plus the version
metric. Fall back to option 3 if alias resolution in the storage initializer
proves unreliable.

## Consequences

To be filled in when accepted. Whichever option wins, the rollback runbook
(`docs/runbooks/`) must describe reversing it in under five minutes.
