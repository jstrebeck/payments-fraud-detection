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

## Results (Phase 7 drill, 2026-09-30)

The simulator was switched to the `fraud-shift` profile through Git at
06:50 UTC. With no further manual step: `FraudFeatureDrift` fired at 07:54
(amount, log_amount, channel_code, amount_sum_24h); the 08:00 CronJob
retrained, and the gate rejected the challenger (the time-split flaw above);
after the card-split fix, the 10:00 CronJob retrained v6, the gate promoted it
(PR-AUC 0.42 -> 0.64 on held-out live cards, recall at 1% FPR 0.29 -> 0.33),
the Job rolled the predictor, and the API switched to v6 and its thresholds
(0.360 / 0.638).

## Known limitation: forgetting what the live window does not show

Measured afterwards on fresh generated traffic (2,000 customers, 30 days):

| Traffic | Model | PR-AUC | card_testing recall | session_hijack recall |
|---|---|---|---|---|
| fraud-shift | v2 (old champion) | 0.31 | 0.71 | 0.01 |
| fraud-shift | v6 (retrained) | 0.54 | 0.39 | 0.45 |
| normal | v2 | 0.91 | 0.84 | n/a |
| normal | v6 | 0.82 | 0.68 | n/a |

v6 is the better model for the drifted world, but it partly forgot
`card_testing`. The held-out live cards contained too few card-testing
frauds for the gate to see that. Options, not yet implemented:

- a larger generated history in retraining (the drill data showed 1,000 x 30
  beats 500 x 14 and no history at all), or weighting live rows instead of
  shrinking the history;
- a second gate check on a generated benchmark with every known pattern:
  reject a challenger whose per-pattern recall regresses beyond a margin
  there, even if it wins on live traffic.

Until then, when traffic returns to normal after a drift promotion, roll back
with `docs/runbooks/rollback-model.md` (done at the end of the drill: v2 was
serving again 18 seconds later).

## Incident: a false drift alarm promoted a worse model (2026-09-30)

After the drill the traffic was back to normal and v2 was serving again, yet
`FraudFeatureDrift` fired on and off from 10:15 to 18:00 and the 18:00 run
promoted v7 (gate: PR-AUC 0.77 -> 0.82 over held-out cards from 7 days). On
fresh normal traffic v7 is worse than v2 (PR-AUC 0.84 vs 0.91). Two causes:

- **The alert counted card-history features** (`txn_count_7d`,
  `amount_sum_7d`, `is_new_*`, distance and speed from the last payment).
  The simulator starts each pass with fresh cards, so these swing with its
  replay cycle, and a 24h-median baseline that was only hours old (and partly
  drift) could not absorb that. With 500 customers per pass, pass-to-pass
  noise also moved `ip_billing_mismatch`.
- **The gate tested on 7 days of held-out labels**, which still held the
  drift period, so a drift-fitted challenger won against a champion that is
  better on today's traffic.

Fixes:

- `fraud:feature_psi:drifting` only counts population features (amount,
  log_amount, channel, merchant category and risk tier, IP/billing mismatch);
  the history features stay on the dashboard. Replayed over the day's
  Prometheus data, the new rule is true from 07:40 to 10:15 (the drift) and
  never after.
- The simulator's world grew from 500 to 2,000 customers (passes of ~8.5h
  instead of ~2h).
- The gate's test set is held-out cards from the last 6 hours
  (`LIVE_TEST_WINDOW_HOURS`). On the real labels: at 18:00 v2 beats v7 there
  (0.944 vs 0.914, so v7 would have been rejected); at 10:00, during the
  drift, v6 still beats v2 (0.621 vs 0.373).
- v2 was put back with the rollback runbook.

