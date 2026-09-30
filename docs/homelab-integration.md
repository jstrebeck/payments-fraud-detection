# Homelab integration

Everything cluster-level is built in the `Homelab-Configuration` repo
(`../Homelab-Configuration`, `github.com/jstrebeck/Homelab-Configuration`).
This document is the contract between the two repos: what this project can
rely on today, and what it needs added. Keep it current.

## What exists today (as of 2026-09-29)

Observed from the homelab repo. Verify before relying on exact values.

| Component | Detail | Notes |
|---|---|---|
| Kubernetes | Talos Linux, control plane and workers on `192.168.2.0/24` (nodes seen: `.104` to `.107`) | Machine config changes via `talosctl patch mc -p @patch.yaml -n <ips>` |
| Storage | Rook-Ceph (`Kubernetes/ceph-rook`): `ceph-block` (default), `ceph-filesystem`, `ceph-bucket` | Longhorn manifests exist in the homelab repo but Longhorn is not running on the cluster. Use `ceph-block`. |
| Load balancer | MetalLB (`Kubernetes/metallb`) | `LoadBalancer` services get LAN IPs |
| Image registry | `registry:2` in namespace `registry`, `LoadBalancer` at `192.168.2.203:5000`, HTTP, no auth | Nodes trust it via `Kubernetes/registry/talos-patch.yaml`. The GH runner VM must also be able to push (Docker `insecure-registries`). |
| Monitoring | `kube-prometheus-stack` Helm release in namespace `monitoring` (`Kubernetes/grafana/values.yaml`) | Prometheus Operator CRDs present. Prometheus selects `ServiceMonitor`s labelled `release: kube-prometheus-stack` (any namespace), so every `ServiceMonitor` in `deploy/` needs that label. |
| MLflow | MLflow 3.16.1 (upgraded from 2.21.3 on 2026-09-29). Namespace `mlops` (`Kubernetes/mlflow`), `LoadBalancer` at `http://192.168.2.202` (port 80), in-cluster `http://mlflow.mlops.svc` | New hostnames must be added to `--allowed-hosts` or clients get 403. Shared with other homelab ML projects. Backend store is Postgres `mlflow-postgres.mlops.svc`. Artifacts in `s3://mlflow-artifacts` on SeaweedFS, proxied by `--serve-artifacts` (clients only need the MLflow URL). `mlflow-artifacts:/<path>` is stored at `s3://mlflow-artifacts/<path>`. |
| S3 store | SeaweedFS 4.48 in namespace `seaweedfs` (`Kubernetes/seaweedfs`), `http://seaweedfs.seaweedfs.svc:8333`, path-style, HTTP, region `us-east-1` | ClusterIP only. Identities `mlflow` (read/write `mlflow-artifacts`) and `kserve-fraud` (read-only). |
| KServe | `v0.20.0`, Standard (RawDeployment) mode, namespace `kserve` (`Kubernetes/kserve`); cert-manager `v1.21.2` for its webhook cert only | No ingress: predictors are reached in-cluster. The stock MLServer runtime cannot load our model; this repo ships its own `ServingRuntime` (ADR-0014). `ClusterStorageContainer` `mlflow` (`Kubernetes/kserve/mlflow-storage-initializer`) resolves `storageUri: models:/<name>@<alias>` at pod start through the MLflow artifact proxy and writes the registry version into MLServer's model settings (ADR-0006). |
| Namespace `fraud` | `Kubernetes/fraud`: ServiceAccount `kserve-sa`, Secret `s3-credentials` (read-only S3, KServe annotations; unused since `fraud-detector` loads via `models:/`, kept for raw `s3://` URIs), Postgres `payments-postgres` with Secrets `payments-postgres` and `payments-db` (`DATABASE_URL`, `postgresql+asyncpg://`) | Secrets created by hand, never committed. |
| External access | Cloudflare tunnel (`Kubernetes/cloudflare`) | Option for exposing MLflow/Grafana; not required |
| CI runner | Proxmox VM `ghactions` (`Terraform/Github-Actions/main.tf`), Ubuntu 22.04, 2 vCPU / 8 GB, `192.168.1.128` | On a different subnet from the cluster. Confirm routing to `192.168.2.203:5000` and to the API server. |
| Provisioning | Terraform (Proxmox), Ansible, cloud-init | Not used by this project directly |

## Requests to the homelab repo

Each component gets its own `Kubernetes/<component>/` folder in the homelab
repo, in the README-plus-manifests style the repo already uses. Postgres
instances live under `Kubernetes/databases/<name>/`, one per consumer, and run
in the consumer's namespace.

### Phase 3 (all applied 2026-09-29; kept as the record of what was asked for)

1. **S3 store: SeaweedFS** `4.48` (ADR-0011). `Kubernetes/seaweedfs/`. Single pod
   (`weed server -s3`) in namespace `seaweedfs`, 50Gi `ceph-block`, endpoint
   `http://seaweedfs.seaweedfs.svc:8333` (path-style, HTTP, `us-east-1`),
   ClusterIP only. Bucket `mlflow-artifacts`. Per-consumer identities:
   `mlflow` (read/write on the bucket) and `kserve-fraud` (read-only).
2. **Postgres for payments**. `Kubernetes/databases/payments/`. Plain StatefulSet
   (`postgres:17.11`, 5Gi `ceph-block`) in namespace `fraud`, Service
   `payments-postgres.fraud.svc:5432`, database `payments`. MLflow keeps its
   existing `mlflow-postgres` in `mlops`.
3. **MLflow tracking server**. Exists (see table above). Switch to
   `--artifacts-destination=s3://mlflow-artifacts` in
   `Kubernetes/mlflow/mlflow.yaml` plus a one-off
   `migrate-artifacts-job.yaml` that copies the old PVC into the bucket
   with the same paths. After the switch, a model whose MLflow source is
   `mlflow-artifacts:/<path>` lives at `s3://mlflow-artifacts/<path>`; that
   `s3://` URI is the `InferenceService` `storageUri`.
4. **cert-manager** `v1.21.2`. `Kubernetes/cert-manager/`. Used only for the self-signed cert on KServe's
   admission webhook (the KServe chart always renders a cert-manager
   `Certificate`). No ACME, no ingress TLS.
5. **KServe** `v0.20.0`. `Kubernetes/kserve/`. `deploymentMode: Standard` (KServe's new name for
   RawDeployment), no Knative, no Istio, ingress creation disabled, so
   predictors are reached in-cluster only. Stock runtimes come from the
   `kserve-runtime-configs` chart; the fraud model uses this repo's
   `fraud-mlserver` `ServingRuntime` instead (ADR-0014).
6. **Namespace `fraud`**. `Kubernetes/fraud/`:
   - ServiceAccount `kserve-sa` referencing Secret `s3-credentials`
   - Secret `s3-credentials` with the read-only `kserve-fraud` keys
     (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) and annotations
     `serving.kserve.io/s3-endpoint: seaweedfs.seaweedfs.svc:8333`,
     `serving.kserve.io/s3-usehttps: "0"`,
     `serving.kserve.io/s3-region: us-east-1`. Created by hand from
     `Kubernetes/fraud/README.md`.
   - Secrets `payments-postgres` (DB bootstrap) and `payments-db` (key
     `DATABASE_URL`, the connection string the API reads). Created by hand
     from `Kubernetes/databases/payments/README.md`, never committed.

### Phase 5

7. **Argo CD** (namespace `argocd`). One `AppProject` `fraud` and one
   `Application` with `source.repoURL` = this repo,
   `path: deploy/overlays/homelab`, `destination.namespace: fraud`,
   automated sync with prune and self-heal.
8. **Runner VM**: Docker daemon with `192.168.2.203:5000` in
   `insecure-registries`; `kubectl` context is *not* required if Argo CD
   does the deploying.

## Endpoints this repo's config will reference

Fill in as they come online.

| Purpose | In-cluster | From LAN |
|---|---|---|
| MLflow | `http://mlflow.mlops.svc` (port 80) | `http://192.168.2.202` |
| S3 (SeaweedFS) | `http://seaweedfs.seaweedfs.svc:8333`, bucket `mlflow-artifacts` | n/a (artifacts via MLflow proxy) |
| Postgres (payments) | `payments-postgres.fraud.svc:5432`, db `payments` | n/a |
| KServe predictor | `http://fraud-detector-predictor.fraud.svc/v2/models/fraud-detector/infer` | n/a |
| payments-api | `http://payments-api.fraud.svc:8000` | TBD (MetalLB) |
| Registry | `192.168.2.203:5000` | same |
| Grafana | `kube-prometheus-stack-grafana.monitoring.svc` | existing |

## Verification checklist after Phase 3

```
kubectl -n mlops get pods
kubectl -n fraud get pods,pvc
kubectl get clusterservingruntimes
curl -s http://192.168.2.202/api/2.0/mlflow/experiments/search -d '{}' | head
kubectl -n fraud get sa kserve-sa -o yaml
kubectl get crd inferenceservices.serving.kserve.io
```
