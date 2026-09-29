# deploy

Kustomize manifests for this project's workloads only. Platform components
are in the homelab repo (ADR-0002). Argo CD syncs `overlays/homelab`
(ADR-0008).

```
deploy/
  base/
    kustomization.yaml
    namespace.yaml                   fraud
    payments-api/                    deployment, service, servicemonitor, hpa
    simulator/                       deployment (low steady rate), servicemonitor
    serving-runtime.yaml             fraud-mlserver ServingRuntime, custom MLServer image (ADR-0014)
    inferenceservice.yaml            fraud-detector, modelFormat mlflow v3, runtime fraud-mlserver
    podmonitor-fraud-detector.yaml   scrapes MLServer metrics (:8082) on the predictor pods
    prometheusrules.yaml             alerts listed in ROADMAP Phase 6
    dashboards-configmap.yaml        generated from ../dashboards by kustomize configMapGenerator
  overlays/
    homelab/
      kustomization.yaml             image tags (bumped by CI), replicas, resource sizes, ingress/LB
    local/
      kustomization.yaml             kind/minikube sizing, for anyone without the homelab
  jobs/
    train.yaml                       Job template applied by train.yml workflow (not synced by Argo)
    retrain-cronjob.yaml             weekly retrain (synced by Argo from Phase 7)
    drift-cronjob.yaml               drift report (Phase 7)
```

## Rules

- Every workload: resource requests and limits, liveness and readiness
  probes, `securityContext` with non-root and read-only root filesystem, a
  `ServiceMonitor`.
- Secrets are referenced by name only (`payments-db`, `s3-credentials`);
  they are created from the homelab repo.
- `kustomize build deploy/overlays/homelab` must succeed in CI with no
  cluster access.
- Image tags in overlays are git SHAs, never `latest`.
