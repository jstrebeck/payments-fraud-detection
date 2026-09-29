from __future__ import annotations

import pyarrow as pa
import pytest

from ml.data.generator import GeneratorConfig, generate

SMALL = GeneratorConfig(seed=123, customers=300, days=21, fraud_rate=0.02)


@pytest.fixture(scope="session")
def small_table() -> pa.Table:
    return generate(SMALL)
