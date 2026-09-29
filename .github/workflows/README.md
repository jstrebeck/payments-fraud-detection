# GitHub Actions workflows

Design in `docs/ci-cd.md`.

| File | Status | Does |
|---|---|---|
| `ci.yml` | Phase 0 | On every PR and push to `main`: `uv sync --locked`, `make lint` (ruff, mypy strict), pytest with coverage, `docker compose config`, and a no-push build of both service images. `kustomize build` of each overlay is added when `deploy/` has one (Phase 4). |
| `build-push.yml` | Phase 5 | Build and push images on `main`, bump tags in `deploy/overlays/homelab`. |
| `train.yml` | Phase 5 | Manual and scheduled training runs via a Kubernetes Job. |

Runner: `ci.yml` runs on `ubuntu-latest` so PRs from forks still get
checked and nothing needs LAN access. Workflows that push to the registry or
reach the cluster use the self-hosted VM `ghactions` (see
`docs/homelab-integration.md`) with `runs-on: [self-hosted]`.

Run the same checks locally with `make check`.
