# ADR-0015: Alert-driven retraining on labelled live decisions

**Status:** Accepted
**Date:** 2026-09-30

## Context

Phase 7 asks that a drifted stream of payments leads, without a human, to a
retrained model serving in the cluster, and only if it is better. Choices:

- **What triggers a retrain.** A fixed schedule only (simple, but reacts a
  week late or retrains for nothing); an Alertmanager webhook calling a
  service that creates a Job (event-driven, but a new always-on component and
  a receiver in the homelab's Alertmanager); or a frequent CronJob that asks
  Prometheus whether the drift alert is firing.
- **What data.** Regenerated synthetic history only (misses what changed);
  the labelled live decisions only (few rows, few frauds); or both.
- **What the gate compares on.** The usual time split of the combined data
  (the test window would be mostly synthetic history, where the old champion
  still looks fine), or the most recent labelled live traffic.
- **How a promotion reaches serving.** Through Git (a bot commit per model,
  ADR-0006 option 3), or the alias plus a predictor rollout (ADR-0006).
- **Decision thresholds.** Fixed in the overlay (they belong to one model and
  go stale on promotion; they also applied to the rule fallback, whose scores
  are on a different scale), or carried by the model version.

## Decision

- **Trigger:** the `retrain` CronJob runs every 30 minutes and retrains when
  `FraudFeatureDrift` is firing (asked from Prometheus), or when the champion
  is older than 7 days, with a 6 hour cooldown between retrains. Checking is
  cheap; most runs exit in seconds.
- **Data:** a small generated history (1,000 customers, 30 days) plus every
  labelled payment of the last 7 days, using the feature vectors the API
  stored when it scored them. That is exact online/offline parity, with no
  recomputation, and it is what the drift was about.
- **Split and gate:** labelled live payments in scoring order; the newest 40%
  is the test window, the 20% before it validates (early stopping); the rest,
  plus the history, trains. The unchanged gate (ADR-0006) compares challenger
  and champion on that window, so promotion means "better on current
  traffic". Too few labels (under 1,000, or under 20 frauds to test on) is a
  skip, not a failure.
- **Promotion:** the gate moves `champion`; the Job then rolls the predictor
  through the Kubernetes API (ServiceAccount `retrainer`, a Role limited to
  patching `fraud-detector` and reading its Deployment). Git does not change.
- **Thresholds:** training tags each version with its recommended review and
  decline thresholds (from its evaluation report). The API reads them with
  the feature-version check and decides with them; the configured 0.5/0.8
  remain for untagged versions and the rule fallback.

## Consequences

- No new components: a CronJob, a Role, and the trainer image already used by
  the registry exporter and drift monitor.
- Reaction time is the alert's 15 minutes plus up to 30 minutes to the next
  CronJob tick plus training (a few minutes).
- Retraining needs labels. Delayed feedback (chargebacks, confirmations) is
  the bottleneck: with a 5 minute delay and 20% of legit payments confirmed,
  an hour of traffic yields about 1,100 labels.
- A retrain can only promote through the gate; a worse challenger is logged
  and rejected. Rollback stays `rollback-model.md`; within the cooldown the
  CronJob will not immediately re-promote.
- Live labels are not a random sample (all frauds, 20% of legits), so fraud
  is over-represented in live rows. That helps the model learn new fraud but
  makes live-window precision look better than production; recall is
  unaffected.
