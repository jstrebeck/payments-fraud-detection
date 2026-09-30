from __future__ import annotations

import argparse
import asyncio
import json
from itertools import islice

import httpx2 as httpx
import pytest

from ml.data.generator import GeneratorConfig, generate
from ml.data.io import iter_transactions
from ml.data.schema import LabelledTransaction
from simulator.cli import parse_duration, parse_seed, stream
from simulator.runner import RunConfig, RunStats, run, shard


def _stream(n: int | None = None) -> list[LabelledTransaction]:
    table = generate(GeneratorConfig(seed=3, customers=40, days=3, fraud_rate=0.05))
    return list(islice(iter_transactions(table), n))


class FakeApi:
    """Records requests; declines anything over $500."""

    def __init__(self, fail_every: int = 0) -> None:
        self.bodies: list[dict[str, object]] = []
        self.fail_every = fail_every

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail_every and len(self.bodies) % self.fail_every == 0:
            return httpx.Response(503)
        decision = "declined" if body["amount"] > 500 else "approved"
        return httpx.Response(201, json={"decision": decision})


def _run(api: FakeApi, txns: list[LabelledTransaction], config: RunConfig) -> RunStats:
    async def go() -> RunStats:
        transport = httpx.MockTransport(api)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            return await run(txns, client, config)

    return asyncio.run(go())


def test_sends_every_transaction_without_labels() -> None:
    txns = _stream()
    api = FakeApi()
    stats = _run(api, txns, RunConfig(rps=0, concurrency=4))
    assert stats.sent == len(txns) == len(api.bodies)
    assert stats.errors == 0
    assert all("is_fraud" not in b and "fraud_pattern" not in b for b in api.bodies)


def test_per_card_order_is_preserved() -> None:
    txns = _stream()
    api = FakeApi()
    _run(api, txns, RunConfig(rps=0, concurrency=8))
    sent: dict[str, list[str]] = {}
    for b in api.bodies:
        sent.setdefault(str(b["card_token"]), []).append(str(b["timestamp"]))
    assert all(ts == sorted(ts) for ts in sent.values())


def test_limit_and_errors_are_counted() -> None:
    api = FakeApi(fail_every=5)
    stats = _run(api, _stream(), RunConfig(rps=0, concurrency=2, limit=20))
    assert len(api.bodies) == 20
    assert (stats.sent, stats.errors) == (16, 4)


def test_rate_limit_paces_requests() -> None:
    api = FakeApi()
    loop_time: list[float] = []

    async def go() -> None:
        transport = httpx.MockTransport(api)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            t0 = asyncio.get_running_loop().time()
            await run(_stream(11), client, RunConfig(rps=50, concurrency=2))
            loop_time.append(asyncio.get_running_loop().time() - t0)

    asyncio.run(go())
    assert loop_time[0] >= 10 / 50 * 0.9


def test_confusion_matrix() -> None:
    stats = RunStats()
    for decision, fraud in [("declined", True), ("review", False), ("approved", True),
                            ("approved", False), ("approved", False)]:  # fmt: skip
        stats.record(decision, fraud)
    assert stats.confusion() == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert "precision=0.500 recall=0.500" in stats.summary()


def test_shard_is_stable() -> None:
    assert shard("tok_0123456789abcdef", 8) == shard("tok_0123456789abcdef", 8)
    assert {shard(f"tok_{i:016x}", 4) for i in range(100)} == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("text", "seconds"), [("90", 90), ("30s", 30), ("10m", 600), ("1.5h", 5400)]
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_garbage() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_duration("ten minutes")


SMALL = GeneratorConfig(seed=11, customers=20, days=2, fraud_rate=0.05)


def test_stream_without_loop_is_one_deterministic_pass() -> None:
    once = [t.transaction_id for t in stream(SMALL, loop=False)]
    again = [t.transaction_id for t in stream(SMALL, loop=False)]
    assert once == again == [t.transaction_id for t in iter_transactions(generate(SMALL))]


def test_loop_continues_with_fresh_seeds() -> None:
    n = len(list(stream(SMALL, loop=False)))
    looped = list(islice(stream(SMALL, loop=True), 3 * n))
    ids = [t.transaction_id for t in looped]
    assert len(ids) == 3 * n
    assert len(set(ids)) == len(ids)  # no replays: every pass has new transaction IDs
    next_pass = GeneratorConfig(**{**SMALL.__dict__, "seed": SMALL.seed + 1})
    assert ids[n : 2 * n] == [t.transaction_id for t in iter_transactions(generate(next_pass))]


def test_parse_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_seed("7") == 7
    monkeypatch.setattr("simulator.cli.time.time", lambda: 1_790_000_000.9)
    assert parse_seed("auto") == 1_790_000_000
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seed("lucky")
