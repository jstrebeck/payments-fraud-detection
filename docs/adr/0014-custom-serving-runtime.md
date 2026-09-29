# ADR-0014: Custom MLServer image as the KServe serving runtime

**Status:** Accepted
**Date:** 2026-09-29
**Supersedes:** the "stock MLServer MLflow runtime" assumption in ADR-0005 and ADR-0013

## Context

ADR-0005 and ADR-0013 assumed KServe's stock `kserve-mlserver` runtime could
load the registered model with no custom code. It cannot. KServe 0.20 (the
installed chart version) ships MLServer 1.5.0 (Python <= 3.11, MLflow 2.x)
and only advertises `mlflow` model format versions 1 and 2. The model (`fraud-detector`, MLflow sklearn
flavour) was logged with MLflow 3.16 in the skops format and needs
scikit-learn 1.9, skops 0.16, pandas 3, LightGBM 4.7 (plus `libgomp1`) and
Python 3.12.

Alternatives:

1. **Stock runtime, older training stack.** Pin training to what MLServer
   1.5 supports. Drags the whole project back to MLflow 2 and Python 3.11,
   and loses skops' safe loading.
2. **Custom predictor on the `kserve` Python SDK** (loads via
   `mlflow.pyfunc`). Needs our own server code, and `kserve` 0.20 and 0.21 pin
   `pandas<3`, which conflicts with the model.
3. **Custom MLServer image**: MLServer 1.7.1 and its MLflow runtime on
   Python 3.12 with the model's exact pins. No server code; same V2 protocol
   and the same MLServer behaviour KServe already knows how to drive.

## Decision

Option 3. `ml/serving/Dockerfile` builds MLServer 1.7.1 + `mlserver-mlflow`
1.7.1 on `python:3.12.14-slim`, with `mlflow-skinny` instead of the full
`mlflow` server package (`mlserver-mlflow` is installed with `--no-deps`; it
only needs the pyfunc loader). A namespaced `ServingRuntime` `fraud-mlserver`
in `deploy/base` uses it for `modelFormat: mlflow`, version `3`, with
`autoSelect: false`, so the `InferenceService` names it explicitly.

## Consequences

- The image must move in lockstep with the training environment.
  `ml/tests/test_serving_requirements.py` fails CI when a pin differs from
  `uv.lock`; a dependency bump means recompile, rebuild, retag, update the
  `ServingRuntime`.
- One more image to build and push (`make serving-image`). Its tag names the
  stack (`mlserver-1.7.1-mlflow-3.16.1`) rather than the git SHA, because
  its content only changes when the pins do.
- Verified locally with the champion model: ready in ~4s, ~210Mi RSS,
  V2 predictions identical to `mlflow.pyfunc` in-process scoring.
- MLServer 1.7.1 caps Python at < 3.13. Moving training to 3.13 needs a newer
  MLServer or option 2.
