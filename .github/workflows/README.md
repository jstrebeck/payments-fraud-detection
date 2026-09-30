# GitHub Actions workflows

Design in `docs/ci-cd.md`.

| File | Status | Does |
|---|---|---|
| `ci.yml` | live | On every PR and push to `main`: `uv sync --locked`, `make lint` (ruff, mypy strict), pytest with coverage, `kubectl kustomize` of the homelab overlay, `docker compose config`, no-push builds of the payments-api, simulator and trainer images. |
| `build-push.yml` | written, **skipped: runner offline** | After a green `ci` on `main`: `scripts/release.sh` builds and pushes the images and pins them in `deploy/overlays/homelab`; the bot commits the bump and Argo CD deploys it. Manual stand-in: `make release`. |
| `train.yml` | written, **skipped: runner offline** | Manual (and weekly) training Job via `scripts/train-cluster.sh`; optional promotion gate. Manual stand-in: `make train-cluster`. |

Runner: `ci.yml` runs on `ubuntu-latest` so PRs from forks still get
checked and nothing needs LAN access. Workflows that push to the registry or
reach the cluster use the self-hosted VM `ghactions` with
`runs-on: [self-hosted, linux, homelab]`. That runner is not registered yet,
so those jobs are gated on the repository variable `SELF_HOSTED_RUNNER` and
are skipped; see `docs/ci-cd.md` for the manual path and how to enable it.
`actionlint.yaml` declares the custom `homelab` label.

Run the same checks locally with `make check`.
