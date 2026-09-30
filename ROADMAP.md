# Roadmap

Phases are sequential. Each has a definition of done. Tick items as they land
and keep the "Current phase" pointer accurate. Items marked **(homelab)** are
built in the `Homelab-Configuration` repo, and are listed here only so the
dependency is visible.

**Current phase:** 6 (Phase 5 waits only on the self-hosted runner; deploys are manual until then)

## Phase 0: Scaffold and design

Goal: a stranger (or an agent) can read the repo and know exactly what is being
built, why, and where each piece goes.

- [x] Directory structure with a README in every directory
- [x] `README.md`, `CLAUDE.md`, `ROADMAP.md`
- [x] Architecture doc and ADRs 0001 to 0009
- [x] Homelab integration doc listing existing and pending cluster components
- [x] `LICENSE` (MIT)
- [x] Root `pyproject.toml` (uv workspace), `Makefile`, `.pre-commit-config.yaml`
- [x] CI workflow that runs lint and tests on every PR (green on an empty repo)

Done when: CI is green, and every directory README describes its interface.

## Phase 1: Synthetic data and the payments API (local only)

Goal: realistic transactions flow through a local API and get a fraud decision
from a placeholder rule, with everything observable.

- [x] `ml/data`: seeded synthetic generator. Customers, merchants, cards (opaque
      tokens), transactions with timestamps, amounts, MCC, geography, device,
      channel. Fraud injected via explicit patterns (card testing bursts,
      geo-impossible travel, high-value first-time merchant, account takeover).
      Deterministic per seed. Writes Parquet.
- [x] `ml/features`: pure functions from a transaction (plus a small rolling
      context) to a feature vector. Same code path for batch and online.
- [x] `services/payments-api`: `POST /payments` validates, builds features, asks
      a `FraudScorer` interface, persists the decision to Postgres, returns
      `approved | declined | review` with the score. `GET /payments/{id}`.
      `/healthz`, `/readyz`, `/metrics`. Placeholder `RuleScorer` implementation.
- [x] `services/simulator`: replays generator output against the API at a
      configurable rate, with a fraud-rate knob.
- [x] `docker-compose.yml`: api, postgres, mlflow, S3 (SeaweedFS, ADR-0011), simulator. `make dev`.
      (MLflow and S3 later moved behind the `local-mlflow` profile; dev uses the
      homelab MLflow, ADR-0012.)
- [x] Unit tests for generator determinism, feature functions, API contract.

Done when: `make dev` then `make simulate` produces decisions visible in
Postgres and metrics visible at `/metrics`.

## Phase 2: Training pipeline and MLflow (local)

Goal: a reproducible training run that produces a registered model version.

- [x] `ml/training`: loads Parquet, time-based split, trains a gradient boosted
      classifier (LightGBM) on the `ml.features` vector (ADR-0013: history-based
      features cannot live inside a stateless model), logs params/metrics/artifacts
      to MLflow, registers the model as `fraud-detector`.
- [x] `ml/evaluation`: PR-AUC, recall at fixed false-positive rate, calibration,
      per-fraud-pattern recall. Emits a JSON report logged as an artifact.
- [x] Promotion gate: compare challenger vs the `champion` alias on a held-out
      window; set alias only if it wins by a margin. `scripts/promote.py`.
- [x] Model card generated from the evaluation report (`docs/model-card-template.md`).
- [x] `MlflowScorer` in the API: loads the `champion` alias locally (used in
      compose, before KServe exists).

Done when: `make train` registers a version, `make promote` moves the alias,
and the API scores with the promoted model in compose.

## Phase 3: Homelab ML platform **(homelab)**

Goal: the cluster can host MLflow and serve models. Built in the homelab repo;
this repo only records the requirements.

- [x] **(homelab)** S3 store (SeaweedFS, replacing MinIO, ADR-0011) with a
      `mlflow-artifacts` bucket
- [x] **(homelab)** Postgres for MLflow backend store (existing
      `mlflow-postgres` in `mlops`)
- [x] **(homelab)** Postgres for the payments API (`Kubernetes/databases/payments`,
      runs in `fraud`)
- [x] **(homelab)** MLflow tracking server, reachable inside the cluster and
      via MetalLB IP (existing, `mlops` namespace, `192.168.2.202`)
- [x] **(homelab)** MLflow artifacts on S3 instead of a PVC, so KServe can
      pull models (old PVC copied 1:1; PVC kept until removed in a follow-up)
- [x] **(homelab)** cert-manager `v1.21.2`, self-signed only, for KServe's webhook
- [x] **(homelab)** KServe `v0.20.0` in Standard (formerly RawDeployment) mode
- [x] **(homelab)** Namespace `fraud` with S3 credential secret and service
      account for the KServe storage initializer
- [x] Serving runtime that can load the registered model (ADR-0014): KServe's stock
      MLServer (1.5.0, Python <= 3.11, MLflow 2.x) cannot load a MLflow 3.16
      skops-format model. Custom MLServer image with the model's pinned
      requirements, pushed to `192.168.2.203:5000`, plus a `ServingRuntime`
      in `fraud`
- [x] `docs/homelab-integration.md` updated with the resulting endpoints

Done when: a training run from a laptop logs to the cluster MLflow, and a
hand-applied `InferenceService` serves the `champion` model.
**Met 2026-09-29:** `fraud-detector` v2 (`champion`) served by
`deploy/base/inferenceservice.yaml`; the V2 request built from the model's
`serving_input_example.json` returns the same probabilities as
`mlflow.pyfunc` locally.

## Phase 4: Serve on KServe and wire the API

- [x] `deploy/base`: `InferenceService` (`modelFormat: mlflow`,
      `storageUri: models:/fraud-detector@champion`, resolved at pod start by
      the homelab's MLflow storage initializer, ADR-0006), payments-api
      `Deployment`/`Service`, Postgres connection secret reference,
      `ServiceMonitor`s
- [x] `KServeScorer` in the API using the V2 inference protocol; timeout and
      fallback to `RuleScorer` with a metric when the model is unavailable
- [x] `ml/serving`: no transformer. Features are computed by the API and the
      model takes the feature vector (ADR-0013); `ml/serving` holds only the
      custom runtime image (ADR-0014)
- [x] Simulator runs in-cluster as a `Deployment` with a low steady rate
- [x] Smoke test script hitting the in-cluster API

Done when: transactions scored in the cluster by the KServe model, decisions
in Postgres, latency and score histograms in Prometheus.
**Met 2026-09-30:** the in-cluster simulator's payments are scored by
`kserve` with `fraud-detector/2` and stored in `payments-postgres`; Prometheus
scrapes payments-api, simulator and the predictor, with scorer latency p50
17 ms / p99 25 ms and `fraud_score` populated. `make smoke-cluster` passes.

## Phase 5: CI/CD and GitOps

> **Runner offline, deploys are manual.** Everything below is built, and the
> manual path is verified on the cluster, but the self-hosted runner on the
> `ghactions` VM is not registered yet. `build-push.yml` and `train.yml` are
> skipped (gated on the `SELF_HOSTED_RUNNER` variable) until it is. Meanwhile
> `make release` and `make train-cluster` run the same scripts by hand. See
> [docs/ci-cd.md](docs/ci-cd.md#enabling-the-runner).

- [x] GitHub Actions on the self-hosted runner: lint, test, build images,
      push to `192.168.2.203:5000` tagged with the git SHA (`ci.yml` on
      GitHub-hosted runners for lint/test/render/builds; `build-push.yml` on the
      self-hosted runner for pushes, **skipped until the runner is registered**)
- [x] Workflow updates the image tag in `deploy/overlays/homelab` on merge to
      `main` (`scripts/release.sh`, committed by the bot with `[skip ci]`; by
      hand with `make release` for now)
- [x] **(homelab)** Argo CD installed, `Application` CR pointing at this
      repo's `deploy/overlays/homelab` (automated sync, prune, self-heal)
- [x] Training as a Kubernetes `Job` image (`ml/training/Dockerfile`,
      `deploy/jobs/train.yaml`); `train.yml` workflow triggers it
      (workflow_dispatch and weekly schedule; **skipped until the runner is
      registered**, `make train-cluster` meanwhile)
- [x] `docs/ci-cd.md` matches reality
- [ ] **(homelab)** Register the Actions runner on `ghactions` (label
      `homelab`), trust the registry in its Docker daemon, add the
      `KUBECONFIG_FRAUD` secret, set `SELF_HOSTED_RUNNER=true`. Deferred.

Done when: a merged PR results in a new API image running in the cluster with
no manual steps, and a training run can be triggered from GitHub.
**Status 2026-09-30:** met except for the runner. Verified by hand: `make
release` pushed `5a4d8c3` images and committed the tag bump, Argo CD rolled it
out without intervention, and `make smoke-cluster` passed. `make
train-cluster` ran a Job that registered `fraud-detector` v4. With the runner
registered, both steps run from GitHub unchanged.

## Phase 6: Observability

- [ ] Grafana dashboards in `dashboards/`: API (RED metrics), model (score
      distribution, decision mix, fallback rate), training (last run, metrics
      over versions via MLflow)
- [ ] `PrometheusRule`s: high fallback rate, p99 latency, fraud rate drift
      beyond band, model version changed
- [ ] Structured request logging with correlation IDs end to end

Done when: dashboards are provisioned by GitOps and at least one alert has
been seen firing and resolving.

## Phase 7: Feedback loop, drift and retraining

- [ ] Delayed label feedback: simulator posts chargeback/confirmation for a
      sample of past transactions; API stores labels
- [ ] Drift job (Evidently or hand-rolled PSI) comparing live feature
      distributions to the training reference; exports Prometheus metrics
- [ ] Scheduled retraining `CronJob` that pulls recent labelled decisions,
      retrains, evaluates, and promotes only through the gate
- [ ] Rollback runbook: repoint alias, Argo sync, verify

Done when: a deliberately drifted simulator run triggers the drift alert, a
retrain runs, and the new champion is serving without manual intervention.

## Phase 8: Polish for the portfolio

- [ ] Architecture diagram image in `docs/`
- [ ] README walkthrough with screenshots (MLflow, Grafana, Argo CD)
- [ ] Short demo script (`scripts/demo.sh`) that exercises the whole loop
- [ ] Repo badges: CI status, license, Python version

## Ideas parked (not committed)

- Argo Workflows for multi-step pipelines (ADR-0007 revisits)
- Feature store (Feast) if online/offline skew becomes a real problem
- Shadow deployment of challenger models via KServe traffic splitting
- Load testing with k6 and a documented capacity number
