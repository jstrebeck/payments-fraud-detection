# syntax=docker/dockerfile:1
# KServe serving runtime for the fraud-detector model (ADR-0014): MLServer +
# its MLflow runtime on Python 3.12, with the exact library versions the model
# was trained with. The model itself is not baked in; KServe's storage
# initializer downloads it to /mnt/models at pod start.
# Build from the repo root: docker build -f ml/serving/Dockerfile ml/serving

FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv pip install --python /app/.venv -r requirements.txt \
    && uv pip install --python /app/.venv --no-deps mlserver-mlflow==1.7.1

FROM python:3.12.14-slim-trixie
# libgomp1: OpenMP runtime LightGBM needs to load the model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=build --chown=app:app /app/.venv /app/.venv
# Defaults for a single MLflow model at /mnt/models. KServe's ServingRuntime
# env overrides any of these (it sets MLSERVER_MODEL_NAME to the isvc name).
# Parallel workers are off: one small LightGBM model scores in well under a
# millisecond, and a worker pool would double the memory footprint.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLSERVER_MODEL_NAME=fraud-detector \
    MLSERVER_MODEL_URI=/mnt/models \
    MLSERVER_MODEL_IMPLEMENTATION=mlserver_mlflow.MLflowRuntime \
    MLSERVER_HTTP_PORT=8080 \
    MLSERVER_GRPC_PORT=9000 \
    MLSERVER_METRICS_PORT=8082 \
    MLSERVER_PARALLEL_WORKERS=0
WORKDIR /app
USER 10001
EXPOSE 8080 9000 8082
CMD ["mlserver", "start", "/mnt/models"]
