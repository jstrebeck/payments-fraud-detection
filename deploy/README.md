# deploy

Kustomize manifests for this project's workloads only. Platform components
are in the homelab repo (ADR-0002). Argo CD syncs `overlays/homelab`
(ADR-0008).

```
deploy/
  base/
    kustomization.yaml               namespace fraud, part-of label; the namespace itself is the homelab repo's
    payments-api/                    deployment (migrate init container, KServeScorer), service, servicemonitor
    simulator/                       deployment (--seed auto --loop, 1.5 rps), metrics service, servicemonitor
    serving-runtime.yaml             fraud-mlserver ServingRuntime, custom MLServer image (ADR-0014)
    inferenceservice.yaml            fraud-detector, modelFormat mlflow v3, runtime fraud-mlserver
    podmonitor-fraud-detector.yaml   scrapes MLServer metrics (:8082) on the predictor pods
    registry-exporter/               MLflow registry -> Prometheus (ml/evaluation/exporter.py), trainer image
    drift-monitor/                   PSI of live features vs the champion's training profile (ml/evaluation/drift_monitor.py)
    retrain/                         CronJob (every 30m: retrain on drift or age, gate, promote, roll) + RBAC, ADR-0015
    prometheusrules.yaml             recording rules and Fraud* alerts (runbooks in docs/runbooks)
    ../../dashboards                 Grafana dashboards as grafana_dashboard ConfigMaps
  overlays/
    homelab/
      kustomization.yaml             image tags (git SHAs; set by scripts/release.sh)
    local/
      README.md                      not implemented; compose is the local path
  jobs/
    train.yaml                       training Job template; created per run by scripts/train-cluster.sh,
                                     not in the base, so Argo CD never syncs it
```

Planned: an HPA for payments-api (if load needs it) and `jobs/` for
retraining and drift (Phase 7).

**Deploying:** Argo CD (homelab repo, app `payments-fraud-detection`) syncs
`overlays/homelab` from `main` with prune and self-heal, so a merged change to
this directory is live within minutes, and `kubectl apply` by hand gets
reverted. New code reaches the cluster when `make release` (or, once the
runner is online, `build-push.yml`) pins new image tags here. Check with
`make smoke-cluster`.

## Rules

- Every workload: resource requests and limits, liveness and readiness
  probes, `securityContext` with non-root and read-only root filesystem, a
  `ServiceMonitor` (or `PodMonitor`) labelled `release: kube-prometheus-stack`,
  the only label Prometheus selects.
- Secrets are referenced by name only (`payments-db`, `s3-credentials`);
  they are created from the homelab repo.
- `kustomize build deploy/overlays/homelab` must succeed in CI with no
  cluster access.
- Image tags in overlays are git SHAs, never `latest`.
