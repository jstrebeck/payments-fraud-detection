# syntax=docker/dockerfile:1
# Build from the repo root: docker build -f services/payments-api/Dockerfile .
# The workspace pattern (ADR-0009): copy the lockfile and every member's
# pyproject.toml, install third-party deps (cached layer), then copy only the
# sources this image needs and install them non-editable.

FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/pyproject.toml
COPY services/payments-api/pyproject.toml services/payments-api/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package payments-api
COPY ml ml
COPY services/payments-api services/payments-api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package payments-api

FROM python:3.12.14-slim-trixie
# libgomp1: OpenMP runtime LightGBM needs to load the model (MlflowScorer).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=build --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    HOME=/tmp
WORKDIR /app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "payments_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
