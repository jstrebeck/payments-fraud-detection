"""App factory. Run with `uvicorn payments_api.main:create_app --factory`."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version

import structlog
from fastapi import FastAPI, Request, Response

from payments_api import metrics
from payments_api.config import Settings
from payments_api.db import create_engine, session_factory
from payments_api.logs import REQUEST_ID_HEADER, configure_logging, request_id_from
from payments_api.policy import DecisionPolicy
from payments_api.routes import router
from payments_api.scoring import build_scorer
from payments_api.service import PaymentService

log = structlog.get_logger(__name__)

QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url)
        scorer = build_scorer(settings)
        policy = DecisionPolicy(settings.review_threshold, settings.decline_threshold)
        app.state.service = PaymentService(session_factory(engine), scorer, policy)
        app.state.require_scorer = settings.require_scorer
        await scorer.start()
        metrics.set_model_version(scorer.name, scorer.model_version)
        log.info(
            "startup",
            scorer=scorer.name,
            model_version=scorer.model_version,
            review_threshold=policy.review_threshold,
            decline_threshold=policy.decline_threshold,
        )
        try:
            yield
        finally:
            await scorer.close()
            await engine.dispose()

    app = FastAPI(title="payments-api", version=version("payments-api"), lifespan=lifespan)
    app.middleware("http")(observe_request)
    app.include_router(router)
    return app


async def observe_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Request ID, structured access log and RED metrics for every request."""
    request_id = request_id_from(request.headers.get(REQUEST_ID_HEADER))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.HTTP_REQUESTS.labels(request.method, path, str(status_code)).inc()
        metrics.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        if request.url.path not in QUIET_PATHS:
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(elapsed * 1000, 2),
            )
