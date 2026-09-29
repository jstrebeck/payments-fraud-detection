"""HTTP response models. The request body is `ml.data.schema.Transaction`."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from ml.data.schema import Transaction
from payments_api.models import Payment, as_utc
from payments_api.policy import Decision


class PaymentDecision(BaseModel):
    payment_id: uuid.UUID
    transaction_id: str
    decision: Decision
    score: float
    scorer: str
    model_version: str

    @classmethod
    def from_row(cls, p: Payment) -> PaymentDecision:
        return cls(
            payment_id=p.payment_id,
            transaction_id=p.transaction_id,
            decision=p.decision,  # constrained on write
            score=p.score,
            scorer=p.scorer,
            model_version=p.model_version,
        )


class PaymentRecord(PaymentDecision):
    transaction: Transaction
    features: dict[str, float]
    created_at: datetime

    @classmethod
    def from_row(cls, p: Payment) -> PaymentRecord:
        txn = Transaction.model_validate(
            {name: getattr(p, name) for name in Transaction.model_fields}
            | {"timestamp": as_utc(p.timestamp)}
        )
        return cls(
            **PaymentDecision.from_row(p).model_dump(),
            transaction=txn,
            features=p.features,
            created_at=as_utc(p.created_at),
        )
