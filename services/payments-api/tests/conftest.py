from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi.testclient import TestClient

from payments_api.config import Settings
from payments_api.main import create_app
from payments_api.migrate import alembic_config


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    """A fresh SQLite database with migrations applied (Postgres is covered by `make smoke`)."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'payments.db'}"
    command.upgrade(alembic_config(url), "head")
    return url


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(database_url=database_url, _env_file=None)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


TxnFactory = Callable[..., dict[str, Any]]


def _make_txn(**overrides: Any) -> dict[str, Any]:
    txn: dict[str, Any] = {
        "transaction_id": "t-0001",
        "timestamp": datetime(2026, 1, 5, 12, 0, tzinfo=UTC).isoformat(),
        "card_token": "tok_0123456789abcdef",
        "customer_id": "cus_0000001",
        "merchant_id": "mer_000001",
        "merchant_category": "grocery",
        "amount": 42.5,
        "currency": "USD",
        "channel": "card_present",
        "device_id": "dev_aaaaaaaaaaaa",
        "ip_country": "US",
        "billing_country": "US",
    }
    return txn | overrides


@pytest.fixture
def make_txn() -> TxnFactory:
    """Build a valid POST /payments body, with field overrides."""
    return _make_txn
