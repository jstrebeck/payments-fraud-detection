# Runbook: feature drift alert

**Trigger:** `FraudFeatureDrift`: two or more features (or one above 0.5)
have a PSI over 0.25 against the champion's training profile **and** more
than twice their own usual level (the 24h median), for 15 minutes. Or
`FraudDriftMonitorDown`: the drift monitor has not produced PSI for 20
minutes.

**What it means:** the payments the model sees now are distributed
differently from what it was trained on. Its scores (and the decision
thresholds from its model card) may no longer mean what they did. It is not
proof the model got worse; labels decide that.

## 1. Look

Grafana **Fraud / drift** (`uid fraud-drift`, `http://192.168.2.204`):

- **PSI vs 24h baseline** table: which features, how far above their usual
  level (`ratio`). A feature that is always high (history features such as
  `txn_count_7d`: the simulator's world differs from the training world) only
  matters if its ratio jumped.
- **PSI per feature** over time: a step at a specific time points to a
  traffic change (a new simulator drift profile, a new merchant mix); a slow
  climb points to gradual drift.

```
# Prometheus
fraud:feature_psi:drifting          # drifting features and their PSI
fraud:feature_psi:max               # every monitored feature
fraud:feature_psi:baseline_24h      # each feature's usual level
```

Then **Fraud / model**: did the flagged share or the score distribution move
too (`FraudDecisionMixDrift`)? Did precision or recall against labels drop?

## 2. Decide

| Situation | Action |
|---|---|
| Traffic changed for good (new pattern, new population) and labels show recall or precision falling | Usually nothing to do: the `retrain` CronJob reacts to this alert within 30 minutes (ADR-0015), retrains on the labelled live decisions and promotes only if the gate passes. Check its logs (below). To run the check now instead of waiting: `kubectl -n fraud create job --from=cronjob/retrain retrain-manual-$(date +%s)` |
| Distribution moved but model quality against labels is unchanged | Nothing, or refresh the reference by retraining when convenient |
| Only the flagged share moved, scores are sane | **Threshold change**: thresholds travel with the model (version tags `recommended_review_threshold` / `recommended_decline_threshold`, read by the API). Set new values on the champion's version in MLflow; the API applies them after a restart |
| A single upstream bug (a field suddenly constant or null) | Fix the producer; do not retrain on broken data |

## 3. What automated retraining did

```
kubectl -n fraud get jobs -l app.kubernetes.io/name=retrain
kubectl -n fraud logs job/<retrain-...>     # JSON lines: decision, skipped | trained, rolled_out
```

- `decision ... retrain: false`: no drift firing, champion young, or waiting
  (6h after a promotion, 1h after a rejected attempt).
- `skipped`: not enough labels yet (needs 1,000 labelled payments and 50
  frauds among the held-out test cards). Labels arrive about 5 minutes after
  scoring.
- `trained` with `gate: ...`: the gate's verdict; on a win, `rolled_out`
  follows and **Fraud / model** shows the new served version.
- A failed Job raises `FraudRetrainFailed`. Exit 4 = champion moved but the
  rollout failed: `uv run python scripts/promote.py --rollout-only`.

## 4. If the monitor is down (`FraudDriftMonitorDown`)

```
kubectl -n fraud logs deploy/drift-monitor --tail=30
```

- `has no reference/feature_profile.json`: the champion was trained before
  profiles existed. Backfill it (regenerates the version's training data from
  its logged generator params and logs the profile to its run):

  ```
  MLFLOW_TRACKING_URI=http://192.168.2.202 uv run python -m ml.evaluation.drift profile --version <N>
  ```

  The monitor picks it up on its next pass (every 5 minutes).
- Database or MLflow errors: check `payments-postgres` in `fraud` and
  `http://mlflow.mlops.svc`.

## Notes

- PSI is computed over the last hour of scored payments and is not reported
  with fewer than 300 of them.
- Calendar features (`hour_*`, `dow_*`) are excluded: in a one-hour window
  they reflect the simulator's replay speed, not the population.
- The baseline needs an hour of history, so a fresh drift monitor (or a new
  champion) cannot alert in its first hour.
