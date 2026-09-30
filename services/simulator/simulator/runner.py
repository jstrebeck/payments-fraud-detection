"""Replay transactions against the payments API at a target rate.

Transactions for the same card always go through the same worker, in time
order, so the API sees each card's history in the order it happened (the
card context depends on it; see ADR-0010). Different cards run concurrently.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx2 as httpx
import structlog

from ml.data.schema import LabelledTransaction
from simulator import metrics
from simulator.feedback import FeedbackScheduler

log = structlog.get_logger(__name__)

FLAGGED = frozenset({"review", "declined"})
REQUEST_ID_HEADER = "x-request-id"


@dataclass(frozen=True, slots=True)
class RunConfig:
    rps: float = 20.0  # 0 means as fast as the API allows
    concurrency: int = 8
    duration_s: float | None = None
    limit: int | None = None
    # With feedback: when the stream ends, wait for pending labels instead of
    # dropping them (they can be minutes out).
    wait_for_feedback: bool = False


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
    feedback: FeedbackScheduler | None = None,
) -> RunStats:
    stats = RunStats()
    queues: list[asyncio.Queue[LabelledTransaction | None]] = [
        asyncio.Queue(maxsize=64) for _ in range(config.concurrency)
    ]

    async def worker(queue: asyncio.Queue[LabelledTransaction | None]) -> None:
        while (txn := await queue.get()) is not None:
            await _send(client, txn, stats, feedback)

    stop_feedback = asyncio.Event()
    feedback_task = (
        asyncio.create_task(feedback.run(client, stop_feedback)) if feedback is not None else None
    )
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
        if feedback is not None and feedback_task is not None:
            stop_feedback.set()
            await feedback_task
            if config.wait_for_feedback:
                await feedback.drain(client)
            elif len(feedback):
                log.info("feedback_pending_lost", pending=len(feedback))
    return stats


async def _send(
    client: httpx.AsyncClient,
    txn: LabelledTransaction,
    stats: RunStats,
    feedback: FeedbackScheduler | None = None,
) -> None:
    body = txn.unlabelled().model_dump(mode="json")
    # Correlation ID for this payment: the API logs it, forwards it to the
    # predictor and stores it on the payment row (docs/runbooks/trace-a-payment.md).
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    try:
        resp = await client.post("/payments", json=body, headers={REQUEST_ID_HEADER: request_id})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        stats.errors += 1
        metrics.REQUESTS.labels("error").inc()
        log.warning(
            "payment_failed",
            request_id=request_id,
            transaction_id=txn.transaction_id,
            error=repr(exc),
        )
        return
    finally:
        metrics.LATENCY.observe(time.perf_counter() - start)
    stats.sent += 1
    payload = resp.json()
    decision = payload["decision"]
    if feedback is not None:
        feedback.offer(payload["payment_id"], is_fraud=txn.is_fraud, pattern=txn.fraud_pattern)
    stats.record(decision, txn.is_fraud)
    metrics.REQUESTS.labels("ok").inc()
    truth = "fraud" if txn.is_fraud else "legit"
    metrics.DECISIONS.labels(decision, truth).inc()
    # Every payment at debug; at info only the ones worth tracing (flagged, or
    # fraud by ground truth, which includes misses). Keeps a steady 1.5 rps
    # simulator from writing a log line per legit approval.
    interesting = decision in FLAGGED or txn.is_fraud
    (log.info if interesting else log.debug)(
        "payment_outcome",
        request_id=request_id,
        transaction_id=txn.transaction_id,
        decision=decision,
        truth=truth,
        pattern=txn.fraud_pattern,
    )
