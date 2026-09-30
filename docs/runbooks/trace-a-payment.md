# Runbook: trace a payment

**Trigger:** someone asks "why was this payment declined?", or an alert
points at a spike and you want one concrete example.

**Idea:** every payment carries one correlation ID (`request_id`) from the
simulator (or any caller sending `x-request-id`) through the API, the KServe
predictor and the `payments` row. See `docs/architecture.md`, "Tracing a
payment".

## Preconditions

- `kubectl` access to namespace `fraud`
- Something to start from: a `request_id`, a `payment_id`, or a
  `transaction_id`

## Steps

1. **Get the `request_id`.**

   From the simulator (flagged and fraudulent payments are logged at info):

   ```
   kubectl -n fraud logs deploy/simulator --since=1h | grep payment_outcome | grep -v '"decision": "approved"' | tail -5
   ```

   Or from a known payment:

   ```
   kubectl -n fraud exec payments-postgres-0 -- psql -U payments -d payments -tAc \
     "select request_id from payments where payment_id = '<payment_id>' or transaction_id = '<transaction_id>'"
   ```

   Rows written before correlation IDs existed have `request_id` null.

2. **API side.** Every log line for that request carries it:

   ```
   kubectl -n fraud logs deploy/payments-api -c payments-api --since=24h | grep '<request_id>'
   ```

   Expect `payment_scored` (decision, score, scorer, model_version) and the
   `request` access line (status, duration). A `scorer_failed_using_fallback`
   line means rules decided it; its `error` names the reason.

3. **Predictor side.** The API sent the ID as the V2 request `id`, and
   MLServer echoes it in the response, but MLServer's access log does not
   print it. Match by time: the `POST /v2/models/fraud-detector/infer` line
   next to the `payment_scored` timestamp.

   ```
   kubectl -n fraud logs deploy/fraud-detector-predictor --since=1h | grep infer | tail
   ```

4. **The record.** Inputs, features and outcome:

   ```
   kubectl -n fraud exec payments-postgres-0 -- psql -U payments -d payments -c \
     "select payment_id, decision, score, scorer, model_version, features from payments where request_id = '<request_id>'"
   ```

## Verify

The decision, score and model version agree across the API log line, the
row, and (for the simulator) `payment_outcome`. If they don't, the payment
was replayed: a replay returns the stored decision and the original
`request_id`, while the replay's own access log line has the new ID.

## Rollback

Read-only; nothing to undo.
