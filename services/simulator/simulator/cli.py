"""`simulator run ...`: generate a synthetic stream and replay it against the API."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime

import httpx2 as httpx
import structlog
from prometheus_client import start_http_server

from ml.data.drift import DRIFT_PROFILES
from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions
from ml.data.schema import LabelledTransaction
from simulator.feedback import FeedbackConfig, FeedbackScheduler
from simulator.runner import RunConfig, run

log = structlog.get_logger(__name__)

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smh]?)$")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600}


def parse_duration(value: str) -> float:
    """'90', '90s', '10m', '1.5h' -> seconds."""
    match = _DURATION.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration {value!r}; use e.g. 30s, 10m, 1h")
    return float(match.group(1)) * _UNITS[match.group(2)]


def _parse_start(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def parse_seed(value: str) -> int:
    """An integer, or `auto` for one derived from the clock (fresh IDs on every start)."""
    if value == "auto":
        return int(time.time())
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid seed {value!r}; use an integer or 'auto'"
        ) from None


def stream(config: GeneratorConfig, *, loop: bool) -> Iterator[LabelledTransaction]:
    """Transactions for `config.seed`; with `loop`, then seed+1, seed+2, ... forever.

    Each pass is generated only when the previous one is exhausted, and each is
    deterministic for its seed. A new seed means new transaction IDs and cards,
    so the API scores fresh traffic instead of replaying stored decisions.
    """
    while True:
        table = generate(config)
        log.info("generated", transactions=table.num_rows, **config.metadata())
        yield from iter_transactions(table)
        if not loop:
            return
        config = replace(config, seed=config.seed + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="replay generated transactions against the API")
    r.add_argument("--api", default=os.environ.get("SIMULATOR_API_URL", "http://localhost:8000"))
    r.add_argument(
        "--seed", type=parse_seed, default=42, help="integer, or 'auto' (from the clock)"
    )
    r.add_argument(
        "--loop", action="store_true", help="after the stream ends, continue with seed+1 (forever)"
    )
    r.add_argument("--customers", type=int, default=500)
    r.add_argument("--days", type=int, default=14)
    r.add_argument("--fraud-rate", type=float, default=0.02)
    r.add_argument("--start", type=_parse_start, default=GeneratorConfig().start)
    r.add_argument("--rps", type=float, default=20.0, help="target requests/s; 0 = unlimited")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--duration", type=parse_duration, default=None, help="e.g. 30s, 10m")
    r.add_argument("--limit", type=int, default=None, help="stop after N transactions")
    r.add_argument("--timeout", type=float, default=5.0, help="per-request timeout, seconds")
    r.add_argument("--metrics-port", type=int, default=None, help="serve /metrics on this port")
    r.add_argument(
        "--drift",
        choices=sorted(DRIFT_PROFILES),
        default="none",
        help="traffic drift profile (ml/data/drift.py); fraud-shift exercises drift detection",
    )
    fb = r.add_argument_group("delayed label feedback (POST /payments/{id}/feedback)")
    fb.add_argument("--no-feedback", action="store_true", help="send no labels")
    fb.add_argument(
        "--feedback-delay", type=parse_duration, default=300.0, help="e.g. 5m (+/-20%% jitter)"
    )
    fb.add_argument(
        "--chargeback-rate", type=float, default=1.0, help="share of fraud that is charged back"
    )
    fb.add_argument(
        "--legit-label-rate", type=float, default=0.2, help="share of legit payments confirmed"
    )
    fb.add_argument("--feedback-max-pending", type=int, default=50_000)
    fb.add_argument(
        "--wait-for-feedback",
        action="store_true",
        help="when the stream ends, wait until pending labels are sent",
    )
    return parser


async def _main(args: argparse.Namespace) -> int:
    gen = GeneratorConfig(
        seed=args.seed,
        customers=args.customers,
        days=args.days,
        fraud_rate=args.fraud_rate,
        start=args.start,
        drift=args.drift,
    )
    if args.metrics_port:
        start_http_server(args.metrics_port)

    config = RunConfig(
        rps=args.rps,
        concurrency=args.concurrency,
        duration_s=args.duration,
        limit=args.limit,
        wait_for_feedback=args.wait_for_feedback,
    )
    feedback = (
        None
        if args.no_feedback
        else FeedbackScheduler(
            FeedbackConfig(
                delay_s=args.feedback_delay,
                chargeback_rate=args.chargeback_rate,
                legit_label_rate=args.legit_label_rate,
                max_pending=args.feedback_max_pending,
            ),
            seed=args.seed,
        )
    )
    async with httpx.AsyncClient(base_url=args.api, timeout=args.timeout) as client:
        stats = await run(stream(gen, loop=args.loop), client, config, feedback)
    print(stats.summary())
    if feedback is not None:
        print(feedback.summary())
    return 1 if stats.sent == 0 else 0


def configure_logging(level: str, fmt: str) -> None:
    """Same shape as the API's logs: JSON in containers, console in dev.

    INFO by default, so per-payment `payment_outcome` lines only appear for
    flagged or fraudulent payments; LOG_LEVEL=DEBUG shows every payment.
    """
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"), os.environ.get("LOG_FORMAT", "console"))
    return asyncio.run(_main(args))
