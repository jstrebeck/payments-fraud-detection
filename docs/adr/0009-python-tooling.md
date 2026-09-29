# ADR-0009: uv workspace, ruff, mypy, pytest, Python 3.12

**Status:** Accepted
**Date:** 2026-09-28

## Context

Three Python packages (`payments-api`, `simulator`, `ml`) share code
(`ml.data`, `ml.features`). They can be one package, three unrelated ones,
or a workspace. The author already uses `uv` and FastAPI elsewhere.

## Decision

One `uv` workspace rooted at the repo, three members. `ml` is a library
depended on by both services. Python 3.12 (LightGBM, MLflow and MLServer
wheels are reliable there; 3.14 is not yet safe for this stack). `ruff` for
lint and format, `mypy --strict` for library code, `pytest` with coverage,
`pre-commit` to run them locally. Each member has its own Dockerfile that
installs only that member and its dependencies.

## Consequences

- One lockfile, consistent versions across services and trainer.
- Images must copy the workspace root `pyproject.toml` and `uv.lock` plus
  the members they need; document the pattern once in
  `services/payments-api/README.md` and copy it.
- The `ml` package's serving dependencies (MLServer) must not leak into
  the API image; use optional dependency groups.
