"""Payment scoring workflow: context -> features -> score -> decision -> persist."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ml.data.schema import Transaction
from ml.features import CardContext, build_features
from payments_api import metrics
from payments_api.logs import current_request_id
from payments_api.models import Payment
from payments_api.policy import DecisionPolicy
from payments_api.repository import PaymentRepository
from payments_api.scoring import FraudScorer, RuleScorer, ScoreResult

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PaymentOutcome:
    payment: Payment
    created: bool  # False when an earlier submission of the same transaction was replayed


class PaymentService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        scorer: FraudScorer,
        policy: DecisionPolicy,
    ) -> None:
        self.sessions = sessions
        self.scorer = scorer
        self.fallback = RuleScorer()
        self.policy = policy

    async def submit(self, txn: Transaction) -> PaymentOutcome:
        """Score and record a transaction. Idempotent on `transaction_id`."""
        async with self.sessions() as session:
            repo = PaymentRepository(session)
            existing = await repo.get_by_transaction_id(txn.transaction_id)
            if existing is not None:
                metrics.PAYMENTS_REPLAYED.inc()
                return PaymentOutcome(existing, created=False)

            history = await repo.card_history(txn.card_token, txn.timestamp)
            features = build_features(txn, CardContext.from_history(history, as_of=txn.timestamp))
            result = await self._score(features)
            decision = self.policy.decide(result.score)

            payment = Payment(
                payment_id=uuid.uuid4(),
                **txn.model_dump(),
                features=features,
                score=result.score,
                decision=decision,
                scorer=result.scorer,
                model_version=result.model_version,
                # A replay returns the stored row, so the original ID is kept.
                request_id=current_request_id(),
            )
            session.add(payment)
            try:
                await session.commit()
            except IntegrityError:
                # Lost a race with a concurrent submission of the same transaction.
                await session.rollback()
                existing = await repo.get_by_transaction_id(txn.transaction_id)
                if existing is None:
                    raise
                metrics.PAYMENTS_REPLAYED.inc()
                return PaymentOutcome(existing, created=False)

        metrics.PAYMENTS.labels(decision).inc()
        metrics.SCORE.observe(result.score)
        log.info(
            "payment_scored",
            payment_id=str(payment.payment_id),
            transaction_id=txn.transaction_id,
            decision=decision,
            score=round(result.score, 4),
            scorer=result.scorer,
            model_version=result.model_version,
        )
        return PaymentOutcome(payment, created=True)

    async def get(self, payment_id: uuid.UUID) -> Payment | None:
        async with self.sessions() as session:
            return await PaymentRepository(session).get(payment_id)

    async def ping_db(self) -> None:
        async with self.sessions() as session:
            await PaymentRepository(session).ping()

    async def _score(self, features: Mapping[str, float]) -> ScoreResult:
        """Primary scorer, falling back to rules on any error. Never raises for scoring."""
        start = time.perf_counter()
        try:
            return await self.scorer.score(features)
        except Exception as exc:
            metrics.SCORER_FALLBACK.labels(type(exc).__name__).inc()
            log.warning("scorer_failed_using_fallback", scorer=self.scorer.name, error=repr(exc))
            return await self.fallback.score(features)
        finally:
            metrics.SCORER_LATENCY.labels(self.scorer.name).observe(time.perf_counter() - start)
