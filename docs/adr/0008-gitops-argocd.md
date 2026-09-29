# ADR-0008: Argo CD for delivery

**Status:** Accepted
**Date:** 2026-09-28

## Context

The homelab currently deploys with `kubectl apply` and `helm install`
following per-directory READMEs. CI could keep doing that with a kubeconfig
secret, or a GitOps controller (Argo CD, Flux) could pull from the repo.

## Decision

Argo CD, installed from the homelab repo, with one `Application` per
environment overlay in this repo. CI never applies long-lived resources; it
changes image tags in Git. Argo CD does the rest.

## Consequences

- Cluster state is auditable in Git history, and rollback is a revert.
- Argo CD's UI is a good demo surface for the portfolio.
- The homelab repo gains a `Kubernetes/argocd/` directory. Over time other
  homelab apps can move to the same model, which is a separate decision.
- The training `Job` is applied by CI, outside Argo CD (see ADR-0007).
