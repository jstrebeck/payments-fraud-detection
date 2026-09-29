# ADR-0010: One definition of card history for training and serving

**Status:** Accepted
**Date:** 2026-09-28

## Context

Most useful fraud features (velocity, novelty, travel speed) depend on the
card's recent history, not just the transaction in hand. Training computes
them over a Parquet file; the API must compute them per request from what it
has stored. If the two paths disagree about which prior transactions count,
the model is trained on one distribution and served another (training/serving
skew), and nothing fails loudly.

Alternatives considered:

- **Precomputed aggregates** (counters in Redis or a feature store). Fast, but
  a second implementation of every window, and a new platform component.
- **Separate batch (SQL/pandas window functions) and online (Python)
  implementations.** The usual source of skew; ruled out by CLAUDE.md.
- **Raw history window, one function.** Store every scored transaction, load
  the card's recent rows, and pass them through the same Python code in both
  paths.

## Decision

Card history is the raw window: transactions on the same card strictly before
the current one, no older than 7 days, capped at the most recent 100, ordered
by `(timestamp, transaction_id)`. `ml.features.CardContext.from_history`
applies these rules and is the only place they exist. The API loads
candidates from its `payments` table with an indexed query and passes them
through `from_history`; the batch featuriser keeps a sliding window per card
and passes it through the same function. A test posts a generated stream
through the API and asserts the stored features equal the batch output
exactly.

The simulator routes each card's transactions through a single worker, in
time order, so the API sees history in the order it happened.

## Consequences

- No feature store, no Redis. One indexed query per request
  (`ix_payments_card_token_timestamp`).
- Features only see what the API has scored. A card's first transaction in
  the API has empty history even if it existed before; the same is true in
  training for the start of the generated window, so both paths agree.
- Concurrent requests for the same card can race: the second may not see the
  first if it has not committed. Acceptable for a demo; noted as a known skew
  source. The simulator avoids it by design.
- Window size and cap are part of the feature definition. Changing them bumps
  `FEATURE_VERSION` and needs a retrain.
- Revisit if per-request latency from the history query becomes significant,
  or if features need windows longer than 7 days (a Feast-style store is on
  the parked-ideas list in ROADMAP.md).
