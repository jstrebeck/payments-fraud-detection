# ml.serving

The KServe serving runtime image for `fraud-detector` (ADR-0014). No Python
code: the model is a plain LightGBM classifier in the MLflow sklearn flavour
(ADR-0013), and MLServer's MLflow runtime serves it as is. What this
directory adds is an image whose library versions match the ones the model
was trained with.

| File | Purpose |
|---|---|
| `requirements.in` | MLServer 1.7.1 plus the model's exact pins (`mlflow-skinny`, `lightgbm`, `scikit-learn`, `skops`, `numpy`, `pandas`) |
| `requirements.txt` | Lock compiled from `requirements.in` with `uv pip compile` |
| `Dockerfile` | uv build stage, then `python:3.12.14-slim-trixie` + `libgomp1`, non-root (uid 10001), `mlserver start /mnt/models` |

The `ServingRuntime` that uses the image and the `InferenceService` are in
`deploy/base/` (`serving-runtime.yaml`, `inferenceservice.yaml`).

## Why not KServe's stock MLServer runtime

`kserve-mlserver` ships MLServer 1.5.0 (Python <= 3.11, MLflow 2.x) and only
advertises `mlflow` model format versions 1 and 2. The registered model needs
MLflow 3.16 (skops serialisation), scikit-learn 1.9, pandas 3 and Python 3.12.

## Interface

- Open Inference Protocol (V2) over HTTP on `:8080`, gRPC on `:9000`,
  Prometheus metrics on `:8082/metrics`
- `POST /v2/models/fraud-detector/infer` with one `FP64` input per feature
  (name = `ml.features.FEATURE_NAMES` entry, shape `[N, 1]`). Optionally set
  request `parameters.content_type: "pd"`. Output `output-1`, `FP64`,
  shape `[N, 2]`: `predict_proba`, fraud probability in column 1.
- `GET /v2/health/ready`, `GET /v2/models/fraud-detector` (metadata incl. the
  signature's input names)
- Model is read from `/mnt/models` (KServe's storage initializer); the model
  name comes from `MLSERVER_MODEL_NAME`, which KServe sets to the
  `InferenceService` name

## Keeping it in step with training

`ml/tests/test_serving_requirements.py` fails when a pin in
`requirements.in` differs from the training environment (`uv.lock`). When it
does: edit the pin, recompile, rebuild and push with a new tag, and update
the image in `deploy/base/serving-runtime.yaml`.

```
uv pip compile --python-version 3.12 ml/serving/requirements.in -o ml/serving/requirements.txt
make serving-image          # builds and pushes $(REGISTRY)/serving:$(SERVING_TAG)
```

## Run it locally against a downloaded model

```
mlflow artifacts download -u "models:/fraud-detector@champion" -d /tmp/model
docker build -f ml/serving/Dockerfile -t fraud-serving ml/serving
docker run --rm -p 8080:8080 -v /tmp/model:/mnt/models:ro fraud-serving
curl -s localhost:8080/v2/health/ready -o /dev/null -w '%{http_code}\n'
```
