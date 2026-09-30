# CI/CD

> **Status (2026-09-30): the self-hosted runner is offline, so deploys are
> manual.** `build-push.yml` and `train.yml` are written and validated, but no
> runner is registered on the `ghactions` VM yet. Their jobs are gated on the
> repository variable `SELF_HOSTED_RUNNER` and show as *skipped* until it is
> set to `true`. Until then, run the same scripts by hand:
>
> | Instead of | Run |
> |---|---|
> | `build-push.yml` (deploy a merged change) | `make release` |
> | `train.yml` (train in the cluster) | `make train-cluster`, then `make promote` if wanted |
>
> Everything downstream of Git is live: Argo CD deploys whatever
> `deploy/overlays/homelab` pins, exactly as it would after a CI bump. To turn
> CI delivery on, see [Enabling the runner](#enabling-the-runner).

## Pipeline

```
PR ──► ci.yml (GitHub-hosted) ──merge──► ci.yml on main ──success──► build-push.yml (self-hosted)
                                                                        │ scripts/release.sh
                                                                        │   build + push payments-api, simulator, trainer :<sha>
                                                                        │   pin <sha> in deploy/overlays/homelab
                                                                        ▼
                                                     commit "chore(deploy): images <sha> [skip ci]"
                                                                        │
                                                                        ▼
                                                     Argo CD (homelab repo, app payments-fraud-detection)
                                                     syncs the overlay: automated, prune, self-heal

train.yml (manual or weekly, self-hosted) ──► scripts/train-cluster.sh ──► Job in namespace fraud
      └─ optional: scripts/promote.py (gate; on a win moves champion and rolls the predictor)
```

## Workflows

| File | Trigger | Runs on | Does |
|---|---|---|---|
| `ci.yml` | every PR, push to `main` | `ubuntu-latest` | `uv sync --locked`, ruff, mypy strict, pytest with coverage, `kubectl kustomize deploy/overlays/homelab`, `docker compose config`, no-push builds of the payments-api, simulator and trainer images |
| `build-push.yml` | `ci` completed successfully on `main`; manual | self-hosted `[self-hosted, linux, homelab]` | `scripts/release.sh` on the exact commit CI tested, then commits the overlay tag bump as `github-actions[bot]` with `[skip ci]` (rebasing onto `main` if it moved) |
| `train.yml` | `workflow_dispatch` (seed, customers, days, promote), Mondays 06:00 UTC | self-hosted | `scripts/train-cluster.sh`: builds the trainer image for the commit if missing, creates a Job from `deploy/jobs/train.yaml`, follows its logs, reports its result. With `promote: true`, runs the gate afterwards. The schedule never promotes. |

CI stays on GitHub-hosted runners so fork PRs are checked and nothing in the
PR path needs LAN access. Only delivery needs the homelab.

## Image tagging

- Tag = short git SHA of the commit that was built, never `latest`.
- `payments-api` and `simulator` tags are pinned in
  `deploy/overlays/homelab/kustomization.yaml`; that commit is the deploy.
- `trainer:<sha>` is chosen per run; the image carries `GIT_SHA`, which the
  trainer records on the MLflow run and model version.
- The serving runtime image is versioned by its stack
  (`serving:mlserver-1.7.1-mlflow-3.16.1`, ADR-0014) and changes rarely:
  `make serving-image`.

## Deploy mechanism

Argo CD watches `deploy/overlays/homelab` on `main` (ADR-0008). Its
`Application` lives in the homelab repo (`Kubernetes/argocd/apps/fraud.yaml`,
project `fraud`, limited to this repo and namespace). CI never applies
long-lived resources. The training `Job` is the one exception: it is
short-lived, not in the Kustomize base, and created per run.

Rollback of a code deploy is `git revert` of the tag-bump commit.

## Model promotion is not a code deploy

Moving the `champion` alias does not go through Git (ADR-0006). The
`InferenceService` uses `storageUri: models:/fraud-detector@champion`, which
the homelab's MLflow storage initializer resolves when a predictor pod
starts. `scripts/promote.py` moves the alias on a gate win and then sets the
predictor pod annotation `fraud-detection/model-version`, which KServe turns
into a rolling update. Argo CD leaves that annotation alone because it is not
declared in Git. See `docs/runbooks/rollback-model.md`.

## Secrets and variables

| Name | Kind | Used by | Notes |
|---|---|---|---|
| `SELF_HOSTED_RUNNER` | repository variable | `build-push.yml`, `train.yml` | `true` once the runner is registered; anything else skips those jobs |
| `GITHUB_TOKEN` | built in | `build-push.yml` | `contents: write` for the tag-bump commit |
| `KUBECONFIG_FRAUD` | repository secret | `train.yml` | kubeconfig for a ServiceAccount allowed to create Jobs and read logs in `fraud`, and to patch the `InferenceService` when promoting |

The registry has no auth. Nothing else is needed.

## Enabling the runner

1. On the `ghactions` VM (`192.168.1.128`): install the Actions runner and
   register it to this repository with the extra label `homelab`.
2. Docker on the VM: add `192.168.2.203:5000` to `insecure-registries`.
   Confirm the VM can reach the registry, `http://192.168.2.202` (MLflow) and
   the Kubernetes API.
3. Create the `KUBECONFIG_FRAUD` secret (see the table above).
4. Set the repository variable `SELF_HOSTED_RUNNER` to `true`.
5. Push a trivial change to `main` and watch `build-push` pin a new tag and
   Argo CD roll it out. Then run `train` once by hand.

## Branch protection

`main` requires `ci` to pass; squash merges. The tag-bump bot commit pushes
directly because it is made by the workflow token and marked `[skip ci]`.
