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
  overlays/
    homelab/
      kustomization.yaml             image tags (git SHAs; bumped by CI from Phase 5)
    local/
      README.md                      not implemented; compose is the local path
```

Planned: an HPA for payments-api (Phase 4+ if load needs it),
`prometheusrules.yaml` and a dashboards ConfigMap (Phase 6), and `jobs/`
for training and retraining (Phases 5 and 7).

Apply by hand until Argo CD arrives (Phase 5):
`kubectl apply -k deploy/overlays/homelab`, then `make smoke-cluster`.

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
