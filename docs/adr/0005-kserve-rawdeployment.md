# ADR-0005: KServe in RawDeployment mode, MLflow runtime, V2 protocol

**Status:** Accepted
**Date:** 2026-09-28

## Context

Options for serving: a plain FastAPI wrapper around the pickled model,
MLflow's built-in `mlflow models serve`, Seldon Core, BentoML, KServe. The
author wants KServe on the CV. KServe's default Serverless mode needs Knative
and Istio, which is a lot of moving parts on a small Talos cluster.

## Decision

KServe installed in **RawDeployment** mode (plain Deployments, Services and
ingress; no Knative, no Istio). The `InferenceService` uses
`modelFormat: mlflow`, so the MLServer runtime loads the registered model
straight from MinIO via the storage initializer. Clients speak the Open
Inference Protocol (V2) over HTTP.

## Consequences

- Scale-to-zero and request-based autoscaling are unavailable; HPA on CPU is
  the fallback. Acceptable for a homelab.
- Canary traffic splitting still works in RawDeployment mode for a future
  shadow/canary phase.
- Needs cert-manager for the webhook (homelab repo). It only issues a
  self-signed cert for KServe's admission webhook; nothing is exposed over TLS.
- KServe 0.20+ renamed RawDeployment to `Standard` (`RawDeployment` is still
  accepted but deprecated). The homelab install uses `Standard` with ingress
  creation disabled, so predictors are reachable in-cluster only, which is
  all the payments API needs.
- The API's `KServeScorer` must build V2 payloads; keep a fixture of a valid
  request in tests.
