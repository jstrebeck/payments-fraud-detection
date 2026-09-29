# CI/CD design

Target state for Phase 5. Update when the workflows exist.

## Workflows

| File | Trigger | Runs on | Does |
|---|---|---|---|
| `ci.yml` | every PR, push to `main` | self-hosted runner (fallback: `ubuntu-latest` for lint/test only) | `uv sync`, `ruff check`, `ruff format --check`, `mypy`, `pytest` with coverage, no-push image builds, `kustomize build` of every overlay (once overlays exist) |
| `build-push.yml` | push to `main`, tags `v*` | self-hosted runner (needs LAN access to registry) | Build `payments-api`, `simulator`, `trainer` images; tag `<git-sha>` and `latest`; push to `192.168.2.203:5000/fraud/*`; update image tags in `deploy/overlays/homelab/kustomization.yaml`; commit with `[skip ci]` |
| `train.yml` | `workflow_dispatch` (params: seed, data window), weekly `schedule` | self-hosted runner | Applies a Kubernetes `Job` from `deploy/jobs/train.yaml` with the current trainer tag, waits for completion, uploads the evaluation report as a workflow artifact, posts the MLflow run URL in the job summary |

## Image tagging

- Immutable tag: short git SHA. This is the tag written into the overlay.
- `latest` exists only for convenience in local scripts. Never referenced
  by a manifest.
- Trainer image also carries the `ml` package version label for MLflow
  `source` tracking.

## Deploy mechanism

Argo CD watches `deploy/overlays/homelab` on `main`. The build workflow's
tag bump commit is the deployment trigger. There is no `kubectl apply` in CI
for long-lived resources; the training `Job` is the one exception, and it is
short-lived and namespaced.

## Model promotion is not a code deploy

Changing the `champion` alias in MLflow does not go through Git. The
`InferenceService` resolves the alias at pod start (Phase 4 decides between
`storageUri: models:/fraud-detector@champion` via the MLflow storage
initializer, or a small controller step that writes the resolved S3 URI).
Either way `scripts/promote.py` triggers a rollout restart of the predictor
so the swap is observable in Argo CD's history and in the
`fraud_model_version_info` metric.

## Secrets in CI

None needed for `ci.yml`. `build-push.yml` needs a GitHub token with
`contents: write` for the tag-bump commit. `train.yml` needs a kubeconfig
scoped to namespace `fraud` (store as a repository secret, rotate when the
cluster is rebuilt). The registry has no auth.

## Branch protection

`main` requires `ci.yml` to pass. Squash merges. The tag-bump bot commit is
allowed to push directly because it is made by the workflow token and marked
`[skip ci]`.
