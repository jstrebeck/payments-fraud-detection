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
  is older than 7 days. It waits 6 hours after a retrain that promoted (no
  promotion loops) but only 1 hour after one the gate rejected, because more
  labels will have arrived. Checking is cheap; most runs exit in seconds.
- **Data:** a small generated history (1,000 customers, 30 days) plus every
  labelled payment of the last 7 days, using the feature vectors the API
  stored when it scored them. That is exact online/offline parity, with no
  recomputation, and it is what the drift was about.
- **Split and gate:** labelled live payments are split **by card** (hash of
  the card token): 20% of cards test, 20% validate (early stopping), 60% plus
  the history train. All of one card's payments, and so a whole fraud
  incident, stay on one side. The unchanged gate (ADR-0006) compares
  challenger and champion on the held-out cards, so promotion means "better
  on current traffic". Too few labels (under 1,000, or under 50 frauds to test
  on) is a skip, not a failure.

  A time split (newest 40% of labels as the test window) was the first
  version. The Phase 7 drill showed its flaw: an hour after drift began, the
  drifted labels were all in validation and test, the challenger never trained
  on the new pattern (recall 0.0 on it) and the gate rightly rejected it.
  Replayed offline on the same labels, the card split promoted the challenger
  (PR-AUC 0.54 to 0.57 against the champion's 0.45, three seeds). With 32
  test frauds one seed was still rejected on recall at 1% FPR, where one
  fraud is worth 0.03; hence the 50-fraud minimum.
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
