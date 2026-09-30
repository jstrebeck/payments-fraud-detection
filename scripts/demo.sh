#!/usr/bin/env bash
# A guided tour of the whole loop (Phase 8).
#
#   scripts/demo.sh            cluster tour, read-only: what is serving, what it
#                              decided, labels, drift, the retrainer's last
#                              decision, then an end-to-end smoke test
#   scripts/demo.sh local      without the homelab: docker compose stack, a
#                              short simulated run and a smoke test (rule scorer
#                              unless MLflow is reachable)
#
# Nothing here changes the cluster. The drift drill that exercises automated
# retraining is a Git change, described at the end of the tour.
set -euo pipefail

MODE="${1:-cluster}"
NS="${NS:-fraud}"
MLFLOW="${MLFLOW_URL:-http://192.168.2.202}"
GRAFANA="${GRAFANA_URL:-http://192.168.2.204}"
ARGOCD="${ARGOCD_URL:-http://192.168.2.217}"
here="$(cd "$(dirname "$0")" && pwd)"

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
sql() { kubectl -n "$NS" exec payments-postgres-0 -- psql -U payments -d payments -P pager=off "$@"; }

if [[ "$MODE" == "local" ]]; then
    cd "$here/.."
    step "Local stack: Postgres + payments API (docker compose)"
    make dev
    step "Replay 30 seconds of synthetic traffic"
    make simulate SIM_ARGS="--rps 20 --duration 30s"
    step "Smoke test"
    make smoke
    echo; echo "API docs: http://localhost:8000/docs   Stop with: make down"
    exit 0
fi

step "Where to look"
printf '  %-10s %s\n' Grafana "$GRAFANA  (dashboards: Fraud / payments-api, model, drift, training)" \
    MLflow "$MLFLOW  (Model training > fraud-detector; Models > fraud-detector)" \
    "Argo CD" "$ARGOCD  (app payments-fraud-detection)"

step "What is serving"
curl -fsS "$MLFLOW/api/2.0/mlflow/registered-models/get?name=fraud-detector" \
    | python3 -c 'import json,sys; m=json.load(sys.stdin)["registered_model"]; print("  registry aliases:", {a["alias"]: a["version"] for a in m.get("aliases", [])})'
echo "  predictor rolled to: v$(kubectl -n "$NS" get isvc fraud-detector -o jsonpath='{.spec.predictor.annotations.fraud-detection/model-version}')"
kubectl -n "$NS" logs deploy/payments-api -c payments-api --since=24h 2>/dev/null \
    | grep model_version_verified | tail -1 \
    | python3 -c 'import json,sys
for line in sys.stdin:
    e = json.loads(line); print("  API decides with v%s, thresholds (review, decline) = %s" % (e["model_version"], e["thresholds"]))' || true

step "Last hour of decisions"
sql -c "select model_version, decision, count(*), round(avg(score)::numeric, 4) as avg_score
        from payments where created_at > now() - interval '1 hour' group by 1, 2 order by 1, 2"

step "Delayed labels (chargebacks and confirmations)"
sql -c "select label, coalesce(label_reason, '') as reason, count(*),
               round(avg(extract(epoch from labelled_at - created_at))) as delay_s
        from payments where labelled_at > now() - interval '1 hour' group by 1, 2 order by 1, 2"

step "Drift monitor (PSI of live features vs the champion's training profile)"
kubectl -n "$NS" logs deploy/drift-monitor --tail=2 2>/dev/null | sed 's/^/  /'

step "Retraining: the CronJob's last decision"
job=$(kubectl -n "$NS" get jobs -l app.kubernetes.io/name=retrain --sort-by=.metadata.creationTimestamp -o name | tail -1)
[[ -n "$job" ]] && kubectl -n "$NS" logs "$job" 2>/dev/null | grep '"event"' | cut -c1-220 | sed 's/^/  /'

step "End to end: one payment through the API and the model"
"$here/smoke-cluster.sh" "$NS"

step "Next: watch it adapt"
cat <<'TXT'
  The drift drill (docs/runbooks/drift-alert.md, ADR-0015):
   1. In deploy/base/simulator/deployment.yaml set --drift=fraud-shift, commit, push.
      Argo CD rolls the simulator; traffic now includes session_hijack fraud.
   2. ~1h: FraudFeatureDrift fires (Grafana: Fraud / drift).
   3. Next retrain tick: the CronJob retrains on labelled live payments, the
      gate compares on held-out cards, and on a win the predictor rolls to the
      new champion (Grafana: Fraud / model, served version).
   4. Revert the commit; put the old champion back with
      docs/runbooks/rollback-model.md.
TXT
