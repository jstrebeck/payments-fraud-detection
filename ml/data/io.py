"""Parquet layout of generated data and helpers to read and write it.

Requires the `data` extra (pyarrow).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.data.schema import LabelledTransaction

ARROW_SCHEMA = pa.schema(
    [
        pa.field("transaction_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("card_token", pa.string(), nullable=False),
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("merchant_id", pa.string(), nullable=False),
        pa.field("merchant_category", pa.string(), nullable=False),
        pa.field("amount", pa.float64(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("channel", pa.string(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("ip_country", pa.string(), nullable=False),
        pa.field("billing_country", pa.string(), nullable=False),
        pa.field("is_fraud", pa.bool_(), nullable=False),
        pa.field("fraud_pattern", pa.string(), nullable=True),
    ]
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def write_parquet(table: pa.Table, path: Path) -> None:
    """Write a table so that identical input gives byte-identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        # Fixed row group size so the file layout does not depend on defaults.
        row_group_size=128 * 1024,
    )


def read_parquet(path: Path) -> pa.Table:
    """Read with the canonical schema, keeping the generator metadata from the footer."""
    table = pq.read_table(path, schema=ARROW_SCHEMA)
    return table.replace_schema_metadata(pq.read_schema(path).metadata)


def iter_transactions(table: pa.Table) -> Iterator[LabelledTransaction]:
    """Yield validated rows in table order, converting timestamps to aware datetimes."""
    ts_us = table.column("timestamp").cast(pa.int64()).to_pylist()
    for i, row in enumerate(table.drop_columns(["timestamp"]).to_pylist()):
        row["timestamp"] = _EPOCH + timedelta(microseconds=ts_us[i])
        yield LabelledTransaction.model_validate(row)
