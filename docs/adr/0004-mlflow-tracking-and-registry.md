# ADR-0004: MLflow for tracking and registry, alias-based promotion

**Status:** Accepted; the model format paragraph is superseded by [ADR-0013](0013-model-input-is-the-feature-vector.md)
**Date:** 2026-09-28

## Context

Need experiment tracking, artifact storage and a model registry with a
notion of "the version in production". Options: MLflow, Weights & Biases
(hosted), a bare S3 bucket with a JSON manifest. Self-hosting is a goal.

## Decision

MLflow tracking server with Postgres backend and MinIO artifact store,
deployed from the homelab repo. Model registered as `fraud-detector`.
Promotion uses registry **aliases** (`champion`, `challenger`), not the
deprecated stages. The evaluation gate is the only code path allowed to move
`champion`. Models are logged as MLflow `sklearn` flavour (pipeline with
feature transformer) so the MLServer MLflow runtime in KServe can load them
without custom code.

## Consequences

- One URL for clients thanks to `--serve-artifacts`.
- Alias moves are not Git commits; observability of "what is live" comes
  from the `fraud_model_version_info` metric and the promote script's rollout.
- If feature logic cannot be serialised inside the pipeline, a KServe
  transformer is needed (`ml/serving`); avoid this if possible.
