# ADR-0003: Synthetic, seeded transaction data

**Status:** Accepted
**Date:** 2026-09-28

## Context

Public fraud datasets (Kaggle credit card, IEEE-CIS, PaySim) exist but come
with licence questions, anonymised opaque features, no way to generate new
traffic, and no way to produce delayed labels or controlled drift. The
project needs a stream of transactions to serve, not just a table to train on.

## Decision

`ml/data` is a deterministic generator (seeded, versioned) producing
customers, merchants, card tokens and transactions with explicit injected
fraud patterns (card testing bursts, impossible travel, high-value first-time
merchant, account takeover). The same generator feeds training (batch Parquet)
and the simulator (live traffic), and exposes knobs for fraud rate and drift.

## Consequences

- Fully reproducible; tests can assert exact counts for a seed.
- Drift and retraining can be demonstrated on purpose (Phase 7).
- Model quality numbers are not comparable to published benchmarks. State
  this in the model card; the point is the pipeline, not the AUC.
- The generator must stay honest: the fraud patterns should be learnable
  from features but not trivially so (no leaking the pattern label).
