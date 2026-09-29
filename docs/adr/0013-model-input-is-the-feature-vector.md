# ADR-0013: The model's input is the feature vector, not the raw transaction

**Status:** Accepted
**Date:** 2026-09-29
**Supersedes:** the "sklearn pipeline with feature transformer" part of ADR-0004

## Context

ADR-0004 planned to serialise `Pipeline([TransactionFeaturizer(), LGBMClassifier()])`
so that a raw transaction could be sent to KServe and the model would carry
its own feature logic. But most useful features (velocity, novelty, travel
speed) depend on the card's history (ADR-0010), and a V2 inference request is
one stateless row. A featurizer inside the model would only see the current
transaction, so it could not compute them. The choices were:

1. **Raw transaction + history in the request**, with a custom pyfunc that
   runs `ml.features`. Keeps feature code inside the model, but V2 payloads
   carrying variable-length history are awkward, and the serving image must
   install this repo's `ml` package, which defeats "no custom serving code".
2. **KServe transformer** that loads history and computes features
   (`ml/serving`). A second service to build, deploy and monitor, and a
   second database client.
3. **Feature vector in the request.** The API already loads history and
   calls `ml.features.build_features` (architecture, request path step 2).
   The model takes those named floats and nothing else.

## Decision

Option 3. The registered model is a plain `LGBMClassifier` logged with the
MLflow sklearn flavour:

- input: the `FEATURE_NAMES` columns (named, float64), enforced by the MLflow
  signature; output: `predict_proba` (`pyfunc_predict_fn`), fraud probability
  in column 1
- saved in the `skops` format with LightGBM's types explicitly trusted, so
  loading never unpickles arbitrary code
- `pip_requirements` pinned explicitly to the packages the model needs
  (mlflow, lightgbm, scikit-learn, skops, numpy, pandas) instead of letting
  MLflow export the whole uv workspace
- every version is tagged `feature_version`. Scorers refuse a model whose tag
  differs from the running code's `ml.features.FEATURE_VERSION` and fall back
  to rules; the gate never compares models across feature versions

## Consequences

- No custom serving code: MLServer's MLflow runtime can load the model as is.
  The stock KServe MLServer image turned out too old for it; ADR-0014 adds a
  custom image (still no server code) in `ml/serving`.
- Training/serving parity rests on the shared `ml.features` code (ADR-0010,
  with a parity test) plus the `feature_version` handshake, not on
  serialising feature code.
- The API must be deployed with the feature version the champion was trained
  on. A feature change means: bump `FEATURE_VERSION`, retrain, promote, deploy
  the API. Until the new model is promoted, the new API scores with rules.
- The serving runtime needs `libgomp1` (LightGBM's OpenMP runtime) and
  `skops`. The API image installs both; Phase 4 must check the MLServer image.
