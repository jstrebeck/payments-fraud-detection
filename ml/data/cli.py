"""Command line interface: `python -m ml.data generate|stream ...`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.data.drift import DRIFT_PROFILES
from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions, write_parquet
from ml.data.schema import LABEL_FIELDS


def _parse_start(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _add_generator_args(p: argparse.ArgumentParser) -> None:
    d = GeneratorConfig()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--customers", type=int, default=d.customers)
    p.add_argument("--days", type=int, default=d.days)
    p.add_argument("--fraud-rate", type=float, default=d.fraud_rate)
    p.add_argument("--start", type=_parse_start, default=d.start, help="ISO date, UTC if naive")
    p.add_argument(
        "--drift", choices=sorted(DRIFT_PROFILES), default=d.drift, help="ml/data/drift.py"
    )


def _config(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        seed=args.seed,
        customers=args.customers,
        days=args.days,
        fraud_rate=args.fraud_rate,
        start=args.start,
        drift=args.drift,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a labelled Parquet file")
    _add_generator_args(gen)
    gen.add_argument("--out", type=Path, default=Path("data/transactions.parquet"))

    stream = sub.add_parser("stream", help="print transactions as JSON lines, in time order")
    _add_generator_args(stream)
    stream.add_argument("--with-labels", action="store_true", help="include is_fraud/fraud_pattern")
    stream.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    table = generate(_config(args))

    if args.command == "generate":
        write_parquet(table, args.out)
        n_fraud = sum(table.column("is_fraud").to_pylist())
        print(
            f"wrote {table.num_rows} transactions ({n_fraud} fraud, "
            f"{n_fraud / max(table.num_rows, 1):.2%}) to {args.out}",
            file=sys.stderr,
        )
        return 0

    exclude = None if args.with_labels else set(LABEL_FIELDS)
    for i, txn in enumerate(iter_transactions(table)):
        if args.limit is not None and i >= args.limit:
            break
        sys.stdout.write(txn.model_dump_json(exclude=exclude) + "\n")
    return 0
