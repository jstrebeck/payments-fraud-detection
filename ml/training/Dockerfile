# syntax=docker/dockerfile:1
# Trainer image: generates the synthetic dataset, then trains, evaluates, logs
# to MLflow and registers a version (`python -m ml.training`). Run as the
# Kubernetes Job in deploy/jobs/train.yaml.
# Build from the repo root: docker build -f ml/training/Dockerfile --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
# Same workspace pattern as services/payments-api/Dockerfile.

FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/pyproject.toml
COPY services/payments-api/pyproject.toml services/payments-api/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package ml --extra train
COPY ml ml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package ml --extra train

FROM python:3.12.14-slim-trixie
# libgomp1: OpenMP runtime LightGBM needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=build --chown=app:app /app/.venv /app/.venv
ARG GIT_SHA=unknown
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA
WORKDIR /app
USER 10001
CMD ["python", "-m", "ml.training"]
