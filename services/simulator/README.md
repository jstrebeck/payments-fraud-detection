# simulator

Drives traffic at the payments API. Locally a CLI (or `make simulate`, which
runs it in compose); in the cluster a `Deployment` with a low steady rate
(Phase 4).

## Behaviour

- Generates a stream with `ml.data.generate` for a seed and fraud rate, then
  replays it in time order at a target rate (`--rps`, 0 = unlimited).
- `--seed auto` takes the seed from the clock; `--loop` continues with
  seed+1, seed+2, ... when a pass ends. Each pass is deterministic for its
  seed (and logged), but new seeds mean new transaction IDs and cards, so a
  long-running or restarted simulator keeps producing fresh scoring work
  instead of idempotent replays. The in-cluster `Deployment` uses both.
- Label fields are stripped before sending; ground truth stays in the
  simulator.
- Each card's transactions go through one worker, in order (hashed by card
  token), so the API sees card history as it happened (ADR-0010). Different
  cards run concurrently (`--concurrency`).
- Records the API's decision against ground truth and prints a summary
  (decision mix, confusion counts, precision/recall treating `review` and
  `declined` as flagged).
- Sends a fresh correlation ID (`x-request-id`, uuid4 hex) with every payment.
  Logs `payment_failed` (warning) and `payment_outcome` with that
  `request_id`: at info for flagged or fraudulent payments (including missed
  fraud), at debug for the rest, so a steady run does not log every approval.
  `LOG_LEVEL` (default `INFO`) and `LOG_FORMAT` (`console`; the image sets
  `json`) as in the API.
- Optional `/metrics` (`--metrics-port`): `fraud_sim_requests_total{outcome}`,
  `fraud_sim_request_duration_seconds`, `fraud_sim_decisions_total{decision,truth}`,
  `fraud_sim_feedback_total{label,outcome}` (outcome: scheduled, sent, error,
  dropped).

## Delayed label feedback

After a payment is scored, the simulator may schedule one
`POST /payments/{payment_id}/feedback` for later, imitating chargebacks and
confirmations:

| Ground truth | Label sent | Probability | `reason` |
|---|---|---|---|
| fraud | `fraud` (a chargeback) | `--chargeback-rate` (default 1.0) | the generator's fraud pattern, e.g. `card_testing` |
| legit | `legit` (a confirmation) | `--legit-label-rate` (default 0.2) | none |

Each label is sent `--feedback-delay` (default `5m`, +/-20% jitter) after
scoring, by one background task, so the send loop never waits on it. Sampling
and jitter use an RNG seeded from `--seed`. Pending labels are held in memory,
capped at `--feedback-max-pending` (default 50,000; when full the earliest-due
one is dropped and counted as `dropped`), and are **lost if the process
stops** before they are due. A finite run drops whatever is still pending
unless `--wait-for-feedback` is given. `--no-feedback` sends no labels.

## Drifted traffic

`--drift <profile>` generates traffic from a drift profile
(`ml/data/drift.py`, documented in `ml/data/README.md`). `fraud-shift` raises
prices x2.5, moves 60% of card-present spend online, and makes half of all
fraud incidents a new pattern, `session_hijack`, which the baseline champion
misses (recall about 8% at the review threshold, against 97% for baseline
fraud). Several features pass a PSI of 0.25, and the chargebacks for the new
pattern are exactly what retraining needs.

**In the cluster, change the Deployment through Git** (recommended): set
`--drift=fraud-shift` in `deploy/base/simulator/deployment.yaml`, commit and
push; Argo CD replaces the steady traffic with drifted traffic. Revert the
commit to go back. A `kubectl edit` would be reverted by Argo CD's self-heal,
and a second, one-off simulator (`kubectl -n fraud run ...`) would mix drifted
with normal traffic and dilute the drift signal.

## CLI

```
uv run simulator run --api http://localhost:8000 --seed 42 --rps 20 --duration 10m
uv run simulator run --customers 2000 --days 30 --fraud-rate 0.03 --rps 0 --limit 5000
uv run simulator run --seed auto --loop --rps 1.5 --metrics-port 9100   # what the cluster runs
uv run simulator run --drift fraud-shift --feedback-delay 10s --wait-for-feedback --limit 2000
make simulate SIM_ARGS="--rps 50 --duration 2m"      # in compose, against the compose API
```

`--api` defaults to `$SIMULATOR_API_URL`, then `http://localhost:8000`.
Re-running with the same seed replays the same transaction IDs, which the API
answers idempotently; use a different `--seed` for fresh traffic.

## Rules

- Same schema as training data. If the generator changes, the simulator
  changes with it (they share code).
- Never hardcodes decisions; the API is the system under test.
