# Payments Fraud Detection

[![ci](https://github.com/jstrebeck/payments-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/jstrebeck/payments-fraud-detection/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

An end-to-end MLOps showcase: a synthetic payments platform that scores every
transaction for fraud in real time, with the full lifecycle of the model
(training, tracking, registry, serving, monitoring, retraining) automated and
running on a self-hosted Kubernetes homelab.

**Author:** Josh Strebeck ([@jstrebeck](https://github.com/jstrebeck))
**Status:** Phases 0 to 7 in place. The API runs in the homelab and scores every payment with the `champion` model served by KServe, deployed by Argo CD from `deploy/overlays/homelab`, with dashboards, alerts, correlation IDs, delayed label feedback, drift detection, and alert-driven retraining that promotes only through the evaluation gate. Next: Phase 8, portfolio polish. See [ROADMAP.md](ROADMAP.md).
**Note:** the self-hosted GitHub Actions runner is offline for now, so the image build/push and cluster-training workflows are skipped and deploys are run by hand (`make release`, `make train-cluster`). Details in [docs/ci-cd.md](docs/ci-cd.md).
**License:** [MIT](LICENSE)

## What this demonstrates

| Skill area | How it shows up here |
|---|---|
| ML experiment tracking and model registry | MLflow tracking server, model registry with alias-based promotion |
| Model serving | KServe `InferenceService` (RawDeployment mode) pulling versioned artifacts from S3-compatible storage |
| Containerisation and CI | Multi-stage Docker images, GitHub Actions on a self-hosted runner, image push to an in-cluster registry |
| GitOps delivery | Argo CD syncs this repo's `deploy/` overlays into the cluster |
| Observability | Prometheus `ServiceMonitor`s, Grafana dashboards, model-quality and drift metrics |
| Automated retraining | Scheduled training job, evaluation gates, promotion only when the challenger beats the champion |
| Infrastructure as code | Cluster platform components live in the [Homelab-Configuration](https://github.com/jstrebeck/Homelab-Configuration) repo |

## Architecture at a glance

![Architecture: simulator, payments API, KServe predictor, Postgres, drift monitor and retrain CronJob in namespace fraud; MLflow, SeaweedFS, the MLflow storage initializer, Prometheus and Grafana on the homelab platform; GitHub, the image registry and Argo CD for delivery](docs/architecture.svg)

Full description: [docs/architecture.md](docs/architecture.md). The diagram is
`docs/architecture.svg`.

## Repository layout

```
services/payments-api/   FastAPI service: accepts payments, calls the fraud model, records decisions
services/simulator/      CLI that generates synthetic traffic against the API (uses ml/data)
ml/data/                 Seeded synthetic transaction generator (single source of truth for the data)
ml/features/             Feature engineering shared by training and online inference
ml/training/             Training entrypoint: trains, evaluates, logs to MLflow, registers the model
ml/evaluation/           Offline evaluation, champion/challenger comparison, promotion gate
ml/serving/              KServe transformer / custom predictor code (only if the MLflow runtime is not enough)
deploy/                  Kustomize manifests for THIS project's workloads (not cluster platform components)
dashboards/              Grafana dashboard JSON, provisioned via ConfigMap
scripts/                 One-off helper scripts (seed data, promote model, smoke tests)
docs/                    Architecture, ADRs, runbooks, homelab integration notes
.github/workflows/       CI (lint/test), image build and push, scheduled training trigger
```

Every directory has its own `README.md` describing what belongs there and what
is expected of it.

## Two-repo split

Anything that is a **cluster platform component** (MLflow server, S3 store,
Postgres, KServe, cert-manager, Argo CD, ingress) is deployed from the
`Homelab-Configuration` repo. This repo only contains **application workloads**
and their manifests. The exact boundary and what needs to be added to the
homelab repo is documented in [docs/homelab-integration.md](docs/homelab-integration.md)
and [ADR-0002](docs/adr/0002-two-repo-boundary.md).

## Getting started

Needs [uv](https://docs.astral.sh/uv/) and Docker Compose.

```
make install     # uv workspace + pre-commit hooks
make check       # ruff, mypy --strict, pytest (what CI runs)
make dev         # postgres + payments-api (MLflow is the homelab server)
make train       # train LightGBM, register it in MLflow
make promote     # promotion gate: move `champion` only if the new model wins
make simulate    # replay synthetic transactions against the API
make smoke       # end-to-end check of a running API
```

Decisions land in Postgres (`make psql`) and metrics at
`http://localhost:8000/metrics`. Details in
[docs/local-development.md](docs/local-development.md).

## Key documents

- [ROADMAP.md](ROADMAP.md): phased plan, current phase, definition of done per phase
- [CLAUDE.md](CLAUDE.md): conventions and instructions for AI coding agents working in this repo
- [docs/architecture.md](docs/architecture.md): system design, data flow, model lifecycle
- [docs/adr/](docs/adr/): decision records (why FastAPI, why synthetic data, why KServe RawDeployment, ...)
- [docs/homelab-integration.md](docs/homelab-integration.md): what already exists in the cluster and what must be added
- [docs/ci-cd.md](docs/ci-cd.md): pipeline design, image tagging, promotion flow
- [docs/model-card-template.md](docs/model-card-template.md): template filled in for every registered model version
