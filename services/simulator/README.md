# simulator

Drives traffic at the payments API. Locally a CLI (or `make simulate`, which
runs it in compose); in the cluster a `Deployment` with a low steady rate
(Phase 4).

## Behaviour

- Generates a stream with `ml.data.generate` for a seed and fraud rate, then
  replays it in time order at a target rate (`--rps`, 0 = unlimited).
- Label fields are stripped before sending; ground truth stays in the
  simulator.
- Each card's transactions go through one worker, in order (hashed by card
  token), so the API sees card history as it happened (ADR-0010). Different
  cards run concurrently (`--concurrency`).
- Records the API's decision against ground truth and prints a summary
  (decision mix, confusion counts, precision/recall treating `review` and
  `declined` as flagged).
- Optional `/metrics` (`--metrics-port`): `fraud_sim_requests_total{outcome}`,
  `fraud_sim_request_duration_seconds`, `fraud_sim_decisions_total{decision,truth}`.
- Phase 7: posts delayed `feedback` for a sample of past transactions; drift
  profiles via `--drift`.

## CLI

```
uv run simulator run --api http://localhost:8000 --seed 42 --rps 20 --duration 10m
uv run simulator run --customers 2000 --days 30 --fraud-rate 0.03 --rps 0 --limit 5000
make simulate SIM_ARGS="--rps 50 --duration 2m"      # in compose, against the compose API
```

`--api` defaults to `$SIMULATOR_API_URL`, then `http://localhost:8000`.
Re-running with the same seed replays the same transaction IDs, which the API
answers idempotently; use a different `--seed` for fresh traffic.

## Rules

- Same schema as training data. If the generator changes, the simulator
  changes with it (they share code).
- Never hardcodes decisions; the API is the system under test.
