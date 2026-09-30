#!/usr/bin/env bash
# Smoke test for a running payments API (Phase 1).
# Usage: scripts/smoke.sh [base-url]   (default http://localhost:8000)
# Checks /healthz, /readyz, posts one payment, reads it back, and checks
# that it shows up in /metrics. Exits non-zero on the first failure.
# EXPECT_SCORER=kserve (set by smoke-cluster.sh, Phase 4) also requires the
# payment to be scored by that scorer with a `fraud-detector/<n>` model version.
set -euo pipefail

API="${1:-http://localhost:8000}"
TXN_ID="smoke-$(date +%s)-$RANDOM"

fail() { echo "FAIL: $*" >&2; exit 1; }
json() { python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }

curl -fsS "$API/healthz" >/dev/null || fail "/healthz"
curl -fsS "$API/readyz" >/dev/null || fail "/readyz"
echo "ok   health and readiness"

body=$(cat <<JSON
{"transaction_id": "$TXN_ID", "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "card_token": "tok_00000000000000aa", "customer_id": "cus_smoke", "merchant_id": "mer_smoke",
 "merchant_category": "grocery", "amount": 12.34, "currency": "USD", "channel": "card_present",
 "device_id": "dev_smoke", "ip_country": "US", "billing_country": "US"}
JSON
)
resp=$(curl -fsS -X POST "$API/payments" -H 'content-type: application/json' -d "$body") \
    || fail "POST /payments"
decision=$(json '["decision"]' <<<"$resp")
payment_id=$(json '["payment_id"]' <<<"$resp")
[[ "$decision" =~ ^(approved|review|declined)$ ]] || fail "unexpected decision: $resp"
scorer=$(json '["scorer"]' <<<"$resp")
model_version=$(json '["model_version"]' <<<"$resp")
echo "ok   POST /payments -> $decision ($payment_id) by $scorer $model_version"
if [[ -n "${EXPECT_SCORER:-}" ]]; then
    [[ "$scorer" == "$EXPECT_SCORER" ]] || fail "scored by $scorer, expected $EXPECT_SCORER: $resp"
    [[ "$model_version" =~ ^fraud-detector/[0-9]+$ ]] || fail "unexpected model_version: $resp"
    echo "ok   scored by $EXPECT_SCORER, $model_version"
fi

curl -fsS "$API/payments/$payment_id" | json '["transaction"]["transaction_id"]' \
    | grep -qx "$TXN_ID" || fail "GET /payments/$payment_id"
echo "ok   GET /payments/{id}"

curl -fsS "$API/metrics" | grep -q '^fraud_payments_total' || fail "/metrics"
echo "ok   /metrics"
echo "smoke test passed"
