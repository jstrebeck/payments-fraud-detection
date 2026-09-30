# dashboards

Grafana dashboards as JSON, provisioned by GitOps: `kustomization.yaml` here
turns each file into a ConfigMap labelled `grafana_dashboard: "1"`,
`deploy/base` includes it, and Argo CD applies it to namespace `fraud`. The
homelab Grafana's sidecar (kube-prometheus-stack, watching all namespaces)
loads them. Grafana: `http://192.168.2.204`.

| File | uid | Panels |
|---|---|---|
| `payments-api.json` | `fraud-payments-api` | payments/min, 5xx ratio, POST /payments p50/p95/p99, scorer latency, decision mix, fallbacks by reason, replays, pods ready |
| `fraud-model.json` | `fraud-model` | served version vs `champion` alias, flagged share with the 24h drift band, score heatmap, model latency, precision/recall and confusion against simulator truth |
| `drift.json` | `fraud-drift` | features drifting, max PSI, window size, reference version, PSI per feature over time (0.1/0.25 lines), PSI now, PSI vs each feature's 24h baseline |
| `training.json` | `fraud-training` | champion, registered versions, time since the last cluster training run, PR-AUC / recall at 1% FPR / per-pattern recall across versions, training Jobs |

Every dashboard has a `datasource` variable (Prometheus), links to the other
fraud dashboards and to the runbooks, and an annotation layer showing
`Fraud*` alerts firing.

Data sources:

- API and simulator metrics: `docs/architecture.md`, "Key metrics".
- Recording rules `fraud:*` (flagged share, fallback ratio, POST p99):
  `deploy/base/prometheusrules.yaml`.
- Registry and training: `fraud_registry_*` and `fraud_training_*` from the
  registry exporter (`ml/evaluation/exporter.py`, `deploy/base/registry-exporter`),
  plus `kube_job_*` from kube-state-metrics.
- Drift: `fraud_feature_psi` and `fraud_drift_*` from the drift monitor
  (`ml/evaluation/drift_monitor.py`), and the `fraud:feature_psi:*` recording rules.

## Editing

Edit in Grafana, then "Export" > "Export for sharing externally" off, keep
the `datasource` variable, and paste over the file. Keep each `uid` stable so
links in runbooks do not break. Changes are live after merge + Argo sync;
Grafana-side edits to provisioned dashboards are overwritten.

