#!/usr/bin/env bash
# Smoke test for the in-cluster payments API (Phase 4).
# Usage: scripts/smoke-cluster.sh [namespace]   (default fraud)
# Port-forwards svc/payments-api to a local port, then runs smoke.sh against it
# requiring the payment to be scored by KServe. The port-forward is stopped on exit.
set -euo pipefail

NS="${1:-fraud}"
PORT="${SMOKE_PORT:-18000}"
here="$(cd "$(dirname "$0")" && pwd)"

kubectl -n "$NS" rollout status deploy/payments-api --timeout=120s >/dev/null
kubectl -n "$NS" port-forward svc/payments-api "$PORT:8000" >/dev/null 2>&1 &
pf=$!
trap 'kill $pf 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
    kill -0 "$pf" 2>/dev/null || { echo "FAIL: port-forward exited" >&2; exit 1; }
    sleep 0.5
done

EXPECT_SCORER="${EXPECT_SCORER:-kserve}" "$here/smoke.sh" "http://127.0.0.1:$PORT"
