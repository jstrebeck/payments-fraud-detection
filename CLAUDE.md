# CLAUDE.md

Guidance for AI coding agents working in this repository. Read this first, then
`ROADMAP.md` to find the current phase, then the `README.md` of whichever
directory you are about to change.

## What this project is

A portfolio project demonstrating MLOps and DevOps engineering. A FastAPI
"payments API" scores every incoming transaction for fraud by calling a model
served with KServe. The model is trained on synthetic data, tracked and
registered in MLflow, promoted by an evaluation gate, deployed through GitOps
(Argo CD), and monitored with Prometheus and Grafana. Everything runs on the
author's Talos Kubernetes homelab.

The **audience is a hiring manager reading the GitHub repo**. Optimise for
clarity, reproducibility and visible engineering discipline over cleverness.
Every component should be explainable in one paragraph of its README.

## Two repositories, one hard boundary

| Belongs in THIS repo | Belongs in `Homelab-Configuration` (../Homelab-Configuration) |
|---|---|
| Application code (`services/`, `ml/`) | MLflow server, its Postgres and the S3 store (SeaweedFS) |
| Kustomize manifests for this project's workloads (`deploy/`) | KServe, cert-manager, Gateway/ingress |
| Grafana dashboard JSON for this project | kube-prometheus-stack itself |
| GitHub Actions workflows for this repo | Argo CD installation and the `Application` CRs that point at this repo |
| Docs, ADRs, runbooks for this project | Talos machine patches, MetalLB, storage, registry |

Never add a cluster platform component to `deploy/`. If a phase needs one,
write the requirement into `docs/homelab-integration.md` under "Pending" and
stop; the user builds it in the homelab repo. Do not edit the homelab repo
unless the user explicitly asks in the current conversation.

## Conventions

- **Python 3.12**, managed with `uv`. One `uv` workspace at the repo root;
  each of `services/payments-api`, `services/simulator`, `ml` is a workspace
  member with its own `pyproject.toml`.
- **Lint/format:** `ruff` (lint + format). **Types:** `mypy --strict` on
  library code, relaxed in tests. **Tests:** `pytest`, tests live next to
  the package they test (`<package>/tests/`).
- **Config** via `pydantic-settings`, read from environment variables. No
  secrets in the repo, ever. Local secrets go in `.env` (gitignored).
- **Logging** via `structlog`, JSON in containers, pretty in dev.
- **Metrics** via `prometheus-client`. Every service exposes `/metrics`.
- **Containers:** multi-stage Dockerfiles, non-root user, pinned base images,
  `uv` for dependency install. Images are tagged with the git SHA.
- **Kubernetes manifests:** Kustomize, `deploy/base` plus overlays. Every
  workload has resource requests/limits, liveness/readiness probes and a
  `ServiceMonitor`.
- **Makefile** at the root is the task runner. Common targets (add as they
  become real): `install`, `lint`, `test`, `dev` (local compose stack),
  `train`, `build`, `push`, `smoke`.
- **Commits:** conventional commits (`feat:`, `fix:`, `docs:`, `ci:`,
  `chore:`). Small, reviewable PRs. Do not commit or push unless asked.
- **Docs:** any decision with a non-obvious alternative gets an ADR in
  `docs/adr/` (copy `docs/adr/0000-template.md`). Update `ROADMAP.md` when a
  phase item is finished.

## How to work here

1. Find the current phase in `ROADMAP.md`. Work on the next unchecked item
   in that phase unless told otherwise. Do not start a later phase early.
2. Read the `README.md` of the directory you are changing. It states what
   belongs there and the interfaces other directories rely on.
3. Feature logic used at training time and at inference time lives in
   `ml/features/` only. Never duplicate feature code into the API.
4. Anything that touches the cluster must be expressible as a manifest in
   `deploy/` or a request in `docs/homelab-integration.md`. No `kubectl apply`
   of hand-written YAML from a terminal as the permanent solution.
5. When a phase item is done, tick it in `ROADMAP.md` and make sure the
   relevant README still describes reality.
6. Keep the synthetic data generator (`ml/data/`) deterministic for a given
   seed. Tests depend on it.

## Homelab facts agents can rely on

Details and IPs in `docs/homelab-integration.md`. Summary:

- Talos Linux cluster, nodes on `192.168.2.0/24`; machine config changes use
  `talosctl patch mc`.
- Storage classes from Rook-Ceph (`ceph-block` is the default; Longhorn is
  not running). MetalLB provides `LoadBalancer` IPs.
- Insecure in-cluster image registry at `192.168.2.203:5000` (HTTP), already
  trusted by the nodes.
- `kube-prometheus-stack` in namespace `monitoring` (Prometheus Operator CRDs
  are available, so `ServiceMonitor`/`PrometheusRule` work). Prometheus only
  selects `ServiceMonitor`s labelled `release: kube-prometheus-stack`.
- GitHub Actions self-hosted runner VM (`ghactions`) exists on Proxmox.
- MLflow exists in namespace `mlops` (`http://192.168.2.202`, in-cluster
  `http://mlflow.mlops.svc`).
- Postgres instances live in the homelab repo under `Kubernetes/databases/`,
  one per consumer, in the consumer's namespace.
- S3 store is SeaweedFS (ADR-0011), in-cluster at
  `http://seaweedfs.seaweedfs.svc:8333`.
- KServe `v0.20.0` (Standard mode, no ingress) with cert-manager for its
  webhook. Namespace `fraud` has `kserve-sa`, `s3-credentials`, and Postgres
  at `payments-postgres.fraud.svc:5432` (Secret `payments-db`).
- Argo CD (homelab repo, `Kubernetes/argocd`) deploys `deploy/overlays/homelab`
  from `main`. Image tags there are the deploy; change them with
  `make release`, not by hand.
- The self-hosted GitHub Actions runner is **not registered**:
  `build-push.yml` and `train.yml` are skipped (`SELF_HOSTED_RUNNER` variable),
  so deploys and cluster training are manual (`make release`,
  `make train-cluster`). See `docs/ci-cd.md`.

## Things not to do

- Do not pull a public fraud dataset into the repo. Synthetic data only
  (ADR-0003).
- Do not add Knative or Istio. KServe runs in RawDeployment mode (ADR-0005).
- Do not add Kubeflow. Orchestration is plain Kubernetes Jobs/CronJobs first,
  Argo Workflows later if needed (ADR-0007).
- Do not write feature engineering in SQL or inside the API handler.
- Do not put real card data shapes or PAN-like strings anywhere, even fake
  ones; use opaque token IDs for cards.
