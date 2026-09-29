# ADR-0002: Platform components live in Homelab-Configuration, workloads live here

**Status:** Accepted
**Date:** 2026-09-28

## Context

The author keeps all cluster configuration in a separate
`Homelab-Configuration` repo (Talos patches, storage, MetalLB, registry,
monitoring stack). This project needs MLflow, MinIO, Postgres, KServe,
cert-manager and Argo CD, none of which exist yet. They could be added here
for self-containment, or there for consistency with how the cluster is run.

## Decision

Anything that is shared infrastructure or would outlive this project goes in
the homelab repo: MLflow and its stores, KServe and cert-manager, Argo CD, the
`fraud` namespace's secrets and service accounts. This repo holds only the
manifests for its own workloads (`deploy/`), which version with the code that
they deploy. Argo CD `Application` CRs live in the homelab repo and point at
this repo. The contract between the two is `docs/homelab-integration.md`.

## Consequences

- A reader of this repo can see application delivery end to end; a reader of
  the homelab repo sees platform engineering. Both are showcased.
- This repo is not runnable in a cluster on its own. The local
  `docker-compose` stack fills that gap for development.
- Agents working here must not add platform components to `deploy/`; they
  write requirements into the integration doc instead (see `CLAUDE.md`).
