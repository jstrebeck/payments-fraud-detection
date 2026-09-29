"""Data access for payments. The only module that builds SQL."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ml.features import LOOKBACK, MAX_HISTORY, PastTransaction
from payments_api.models import Payment, as_utc


class PaymentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, payment_id: uuid.UUID) -> Payment | None:
        return await self.session.get(Payment, payment_id)

    async def get_by_transaction_id(self, transaction_id: str) -> Payment | None:
        stmt = select(Payment).where(Payment.transaction_id == transaction_id)
        return (await self.session.scalars(stmt)).one_or_none()

    async def card_history(self, card_token: str, as_of: datetime) -> list[PastTransaction]:
        """Prior transactions for the card inside the feature window (ADR-0010)."""
        stmt = (
            select(Payment)
            .where(
                Payment.card_token == card_token,
                Payment.timestamp < as_of,
                Payment.timestamp >= as_of - LOOKBACK,
            )
            .order_by(Payment.timestamp.desc(), Payment.transaction_id.desc())
            .limit(MAX_HISTORY)
        )
        return [
            PastTransaction(
                transaction_id=p.transaction_id,
                timestamp=as_utc(p.timestamp),
                amount=p.amount,
                merchant_id=p.merchant_id,
                merchant_category=p.merchant_category,
                channel=p.channel,
                device_id=p.device_id,
                ip_country=p.ip_country,
            )
            for p in await self.session.scalars(stmt)
        ]

    async def ping(self) -> None:
        await self.session.execute(text("SELECT 1"))
