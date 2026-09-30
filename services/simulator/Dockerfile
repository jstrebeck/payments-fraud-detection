# syntax=docker/dockerfile:1
# Build from the repo root: docker build -f services/simulator/Dockerfile .
# Same workspace pattern as services/payments-api/Dockerfile.

FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/pyproject.toml
COPY services/payments-api/pyproject.toml services/payments-api/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package simulator
COPY ml ml
COPY services/simulator services/simulator
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package simulator

FROM python:3.12.14-slim-trixie
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=build --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json
WORKDIR /app
USER 10001
EXPOSE 9100
ENTRYPOINT ["simulator"]
CMD ["run", "--help"]
