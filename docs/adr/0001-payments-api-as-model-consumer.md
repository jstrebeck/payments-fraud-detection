# ADR-0001: The payments API is the model's consumer, not its subject

**Status:** Accepted
**Date:** 2026-09-28

## Context

The original idea was "build a dummy payments API with FastAPI and build the
model around it". Two readings exist: the API is the system that produces data
the model learns from, or the API is the system that calls the model. For an
MLOps portfolio the interesting story is the second one: a production service
depends on a model, so the model must be versioned, served, monitored and
replaced safely.

## Decision

`services/payments-api` is a thin FastAPI service that receives transactions,
computes features, calls a `FraudScorer` abstraction (KServe in the cluster),
records the decision, and returns it. It does not own the data-generating
process; `ml/data` does. The API is the integration point that makes every
MLOps concern visible: latency budgets, fallbacks, model version in logs and
metrics, delayed labels.

## Consequences

- The `FraudScorer` interface has three implementations over the project's
  life (`RuleScorer`, `MlflowScorer`, `KServeScorer`), which is a natural
  way to show the progression.
- Feature computation must be shared between the API and training
  (`ml/features`), which forces the training/serving skew problem into the
  open rather than hiding it.
- The API stays small. Resist adding accounts, auth or a UI.
