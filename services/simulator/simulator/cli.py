"""`simulator run ...`: generate a synthetic stream and replay it against the API."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx2 as httpx
import structlog
from prometheus_client import start_http_server

from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="replay generated transactions against the API")
    r.add_argument("--api", default=os.environ.get("SIMULATOR_API_URL", "http://localhost:8000"))
    r.add_argument("--seed", type=int, default=42)
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
    return parser


async def _main(args: argparse.Namespace) -> int:
    gen = GeneratorConfig(
        seed=args.seed,
        customers=args.customers,
        days=args.days,
        fraud_rate=args.fraud_rate,
        start=args.start,
    )
    table = generate(gen)
    log.info("generated", transactions=table.num_rows, **gen.metadata())
    if args.metrics_port:
        start_http_server(args.metrics_port)

    config = RunConfig(
        rps=args.rps, concurrency=args.concurrency, duration_s=args.duration, limit=args.limit
    )
    async with httpx.AsyncClient(base_url=args.api, timeout=args.timeout) as client:
        stats = await run(iter_transactions(table), client, config)
    print(stats.summary())
    return 1 if stats.sent == 0 else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_main(args))
