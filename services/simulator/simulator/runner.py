"""Replay transactions against the payments API at a target rate.

Transactions for the same card always go through the same worker, in time
order, so the API sees each card's history in the order it happened (the
card context depends on it; see ADR-0010). Different cards run concurrently.
"""

from __future__ import annotations

import asyncio
import time
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx2 as httpx
import structlog

from ml.data.schema import LabelledTransaction
from simulator import metrics

log = structlog.get_logger(__name__)

FLAGGED = frozenset({"review", "declined"})


@dataclass(frozen=True, slots=True)
class RunConfig:
    rps: float = 20.0  # 0 means as fast as the API allows
    concurrency: int = 8
    duration_s: float | None = None
    limit: int | None = None


@dataclass
class RunStats:
    sent: int = 0
    errors: int = 0
    # (decision, is_fraud) -> count
    outcomes: Counter[tuple[str, bool]] = field(default_factory=Counter)

    def record(self, decision: str, is_fraud: bool) -> None:
        self.outcomes[(decision, is_fraud)] += 1

    def confusion(self) -> dict[str, int]:
        """Treat `review` and `declined` as flagged."""
        c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for (decision, is_fraud), n in self.outcomes.items():
            flagged = decision in FLAGGED
            key = ("tp" if is_fraud else "fp") if flagged else ("fn" if is_fraud else "tn")
            c[key] += n
        return c

    def summary(self) -> str:
        c = self.confusion()
        precision = c["tp"] / max(c["tp"] + c["fp"], 1)
        recall = c["tp"] / max(c["tp"] + c["fn"], 1)
        decisions = Counter[str]()
        for (decision, _), n in self.outcomes.items():
            decisions[decision] += n
        mix = ", ".join(f"{d}={decisions[d]}" for d in ("approved", "review", "declined"))
        return (
            f"sent={self.sent} errors={self.errors} | {mix} | "
            f"tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']} "
            f"precision={precision:.3f} recall={recall:.3f}"
        )


def shard(card_token: str, n: int) -> int:
    """Stable worker index for a card (not Python's randomised hash())."""
    return zlib.crc32(card_token.encode()) % n


async def run(
    transactions: Iterable[LabelledTransaction],
    client: httpx.AsyncClient,
    config: RunConfig,
) -> RunStats:
    stats = RunStats()
    queues: list[asyncio.Queue[LabelledTransaction | None]] = [
        asyncio.Queue(maxsize=64) for _ in range(config.concurrency)
    ]

    async def worker(queue: asyncio.Queue[LabelledTransaction | None]) -> None:
        while (txn := await queue.get()) is not None:
            await _send(client, txn, stats)

    workers = [asyncio.create_task(worker(q)) for q in queues]
    started = time.monotonic()
    interval = 1.0 / config.rps if config.rps > 0 else 0.0
    next_at = started
    try:
        for i, txn in enumerate(transactions):
            if config.limit is not None and i >= config.limit:
                break
            now = time.monotonic()
            if config.duration_s is not None and now - started >= config.duration_s:
                break
            if interval:
                if next_at > now:
                    await asyncio.sleep(next_at - now)
                next_at += interval
            await queues[shard(txn.card_token, config.concurrency)].put(txn)
    finally:
        for q in queues:
            await q.put(None)
        await asyncio.gather(*workers)
    return stats


async def _send(client: httpx.AsyncClient, txn: LabelledTransaction, stats: RunStats) -> None:
    body = txn.unlabelled().model_dump(mode="json")
    start = time.perf_counter()
    try:
        resp = await client.post("/payments", json=body)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        stats.errors += 1
        metrics.REQUESTS.labels("error").inc()
        log.warning("payment_failed", transaction_id=txn.transaction_id, error=repr(exc))
        return
    finally:
        metrics.LATENCY.observe(time.perf_counter() - start)
    stats.sent += 1
    decision = resp.json()["decision"]
    stats.record(decision, txn.is_fraud)
    metrics.REQUESTS.labels("ok").inc()
    metrics.DECISIONS.labels(decision, "fraud" if txn.is_fraud else "legit").inc()
