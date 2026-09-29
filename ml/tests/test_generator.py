from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ml.data.cli import main as cli_main
from ml.data.generator import GENERATOR_VERSION, GeneratorConfig, generate
from ml.data.io import ARROW_SCHEMA, iter_transactions, read_parquet, write_parquet
from ml.data.schema import FRAUD_PATTERNS
from ml.tests.conftest import SMALL


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_gives_byte_identical_parquet(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet(generate(SMALL), a)
    write_parquet(generate(SMALL), b)
    assert _sha(a) == _sha(b)


def test_different_seed_gives_different_data(small_table: pa.Table) -> None:
    other = generate(replace(SMALL, seed=SMALL.seed + 1))
    assert other.column("transaction_id") != small_table.column("transaction_id")


def test_golden_summary_for_seed(small_table: pa.Table) -> None:
    """Guards against accidental generator changes. If you change the generator on
    purpose, bump GENERATOR_VERSION and update these numbers."""
    patterns = Counter(small_table.column("fraud_pattern").to_pylist())
    assert GENERATOR_VERSION == "2"
    assert small_table.num_rows == 10_864
    assert patterns == {
        None: 10_643,
        "account_takeover": 71,
        "impossible_travel": 51,
        "high_value_new_merchant": 50,
        "card_testing": 49,
    }


def test_schema_and_ordering(small_table: pa.Table) -> None:
    assert small_table.schema.remove_metadata() == ARROW_SCHEMA
    ts = small_table.column("timestamp").cast(pa.int64()).to_pylist()
    assert ts == sorted(ts)
    assert pc.count_distinct(small_table.column("transaction_id")).as_py() == small_table.num_rows
    meta = small_table.schema.metadata
    assert meta[b"seed"] == b"123"
    assert meta[b"generator_version"] == GENERATOR_VERSION.encode()


def test_every_row_validates_against_the_pydantic_schema(small_table: pa.Table) -> None:
    rows = list(iter_transactions(small_table))
    assert len(rows) == small_table.num_rows
    assert all(r.is_fraud == (r.fraud_pattern is not None) for r in rows)


@pytest.mark.parametrize("rate", [0.005, 0.02, 0.05])
def test_fraud_rate_within_tolerance(rate: float) -> None:
    table = generate(replace(SMALL, fraud_rate=rate))
    observed = pc.sum(table.column("is_fraud")).as_py() / table.num_rows
    # Fraud is injected as whole incidents, so the last one can overshoot by up
    # to one card-testing burst (12 rows).
    assert abs(observed - rate) <= max(0.1 * rate, 12 / table.num_rows)


def test_zero_fraud_rate_has_no_fraud() -> None:
    table = generate(replace(SMALL, fraud_rate=0.0))
    assert pc.sum(table.column("is_fraud")).as_py() in (0, None)


def test_fraud_rate_does_not_change_legit_traffic(small_table: pa.Table) -> None:
    other = generate(replace(SMALL, fraud_rate=0.05))
    legit = small_table.filter(pc.invert(small_table.column("is_fraud")))
    other_legit = other.filter(pc.invert(other.column("is_fraud")))
    assert legit.column("transaction_id") == other_legit.column("transaction_id")


def test_all_patterns_present(small_table: pa.Table) -> None:
    assert set(small_table.column("fraud_pattern").drop_null().to_pylist()) == set(FRAUD_PATTERNS)


def test_card_tokens_are_opaque(small_table: pa.Table) -> None:
    tokens = pc.unique(small_table.column("card_token")).to_pylist()
    assert all(t.startswith("tok_") and len(t) == 20 for t in tokens)


@pytest.mark.parametrize(
    "kwargs",
    [{"customers": 0}, {"days": 0}, {"fraud_rate": 0.6}, {"fraud_rate": -0.1}],
)
def test_invalid_config_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        GeneratorConfig(**kwargs)  # type: ignore[arg-type]


def test_cli_generate_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "t.parquet"
    assert (
        cli_main(["generate", "--seed", "5", "--customers", "20", "--days", "3", "--out", str(out)])
        == 0
    )
    table = read_parquet(out)
    assert table.num_rows > 0
    assert table.schema.metadata[b"seed"] == b"5"
    assert table.equals(generate(GeneratorConfig(seed=5, customers=20, days=3)))


def test_cli_stream_strips_labels(capsys: pytest.CaptureFixture[str]) -> None:
    cli_main(["stream", "--customers", "10", "--days", "2", "--limit", "3"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert all('"is_fraud"' not in line for line in lines)
