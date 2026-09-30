"""structlog setup: JSON in containers, human-readable in dev."""

from __future__ import annotations

import logging
import re
import uuid

import structlog

REQUEST_ID_HEADER = "x-request-id"
# What a caller may supply as a correlation ID. Anything else (too long, or
# characters that could forge log fields) is replaced with a fresh ID.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


def request_id_from(header: str | None) -> str:
    """The caller's correlation ID if it is well-formed, otherwise a new one."""
    if header and _VALID_REQUEST_ID.fullmatch(header):
        return header
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """The correlation ID bound for the request being handled, if any."""
    value = structlog.contextvars.get_contextvars().get("request_id")
    return value if isinstance(value, str) else None


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
