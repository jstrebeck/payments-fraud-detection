"""Delayed label feedback: chargebacks and confirmations for past payments.

In production a chargeback arrives days or weeks after the payment, and most
legitimate payments are never confirmed at all. The simulator imitates that
shape at homelab speed: after a payment is scored it may schedule one
`POST /payments/{id}/feedback` for later.

- fraud (by ground truth): a chargeback with probability `chargeback_rate`,
  label `fraud`, reason = the generator's fraud pattern
- legit: a confirmation with probability `legit_label_rate`, label `legit`

Feedback is sent `delay_s` (+/- `jitter`) after scoring by one background
task, so the send loop never waits on it. Pending feedback lives in memory,
capped at `max_pending` (the earliest-due item is dropped and counted when
full), and is lost if the process stops before it is due.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx2 as httpx
import structlog

from simulator import metrics

log = structlog.get_logger(__name__)

SOURCE = "simulator"


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    delay_s: float = 300.0
    jitter: float = 0.2  # fraction of delay_s, uniform +/-
    chargeback_rate: float = 1.0
    legit_label_rate: float = 0.2
    max_pending: int = 50_000

    def __post_init__(self) -> None:
        if self.delay_s < 0:
            raise ValueError("delay_s must be >= 0")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        for name in ("chargeback_rate", "legit_label_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_pending < 1:
            raise ValueError("max_pending must be >= 1")


@dataclass(order=True, slots=True)
class _Pending:
    due: float
    seq: int
    payment_id: str = field(compare=False)
    label: str = field(compare=False)
    reason: str | None = field(compare=False)


class FeedbackScheduler:
    """Samples which payments get a label and posts it when due."""

    def __init__(self, config: FeedbackConfig, *, seed: int) -> None:
        self.config = config
        self._rng = random.Random(seed)  # sampling and jitter: reproducible per seed
        self._heap: list[_Pending] = []
        self._seq = itertools.count()
        self._wake = asyncio.Event()
        self.counts: Counter[str] = Counter()  # outcome -> n, for the run summary
        self._now = time.monotonic

    def __len__(self) -> int:
        return len(self._heap)

    def offer(self, payment_id: str, *, is_fraud: bool, pattern: str | None) -> bool:
        """Maybe schedule feedback for a scored payment. True if it was scheduled."""
        rate = self.config.chargeback_rate if is_fraud else self.config.legit_label_rate
        if self._rng.random() >= rate:
            return False
        label = "fraud" if is_fraud else "legit"
        spread = self.config.delay_s * self.config.jitter
        delay = self.config.delay_s + self._rng.uniform(-spread, spread)
        if len(self._heap) >= self.config.max_pending:
            dropped = heapq.heappop(self._heap)  # earliest due = oldest scheduled
            self._count(dropped.label, "dropped")
        item = _Pending(
            self._now() + delay,
            next(self._seq),
            payment_id,
            label,
            pattern if is_fraud else None,
        )
        heapq.heappush(self._heap, item)
        self._count(label, "scheduled")
        self._wake.set()
        return True

    async def run(self, client: httpx.AsyncClient, stop: asyncio.Event) -> None:
        """Post feedback as it falls due until `stop` is set."""
        while not stop.is_set():
            now = self._now()
            if self._heap and self._heap[0].due <= now:
                await self._post(client, heapq.heappop(self._heap))
                continue
            timeout = min(self._heap[0].due - now, 1.0) if self._heap else 1.0
            self._wake.clear()
            stopped = asyncio.ensure_future(stop.wait())
            woken = asyncio.ensure_future(self._wake.wait())
            await asyncio.wait({stopped, woken}, timeout=timeout, return_when="FIRST_COMPLETED")
            for task in (stopped, woken):
                task.cancel()

    async def drain(self, client: httpx.AsyncClient) -> None:
        """Post everything still pending, each when it falls due."""
        while self._heap:
            wait = self._heap[0].due - self._now()
            if wait > 0:
                await asyncio.sleep(wait)
            await self._post(client, heapq.heappop(self._heap))

    async def _post(self, client: httpx.AsyncClient, item: _Pending) -> None:
        body = {"label": item.label, "reason": item.reason, "source": SOURCE}
        try:
            resp = await client.post(f"/payments/{item.payment_id}/feedback", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._count(item.label, "error")
            log.warning(
                "feedback_failed", payment_id=item.payment_id, label=item.label, error=repr(exc)
            )
            return
        self._count(item.label, "sent")
        log.debug("feedback_sent", payment_id=item.payment_id, label=item.label, reason=item.reason)

    def _count(self, label: str, outcome: str) -> None:
        metrics.FEEDBACK.labels(label, outcome).inc()
        self.counts[outcome] += 1

    def summary(self) -> str:
        c = self.counts
        return (
            f"feedback scheduled={c['scheduled']} sent={c['sent']} errors={c['error']} "
            f"dropped={c['dropped']} pending={len(self._heap)}"
        )
