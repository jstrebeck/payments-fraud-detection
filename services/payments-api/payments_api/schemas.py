"""HTTP response models. The request body is `ml.data.schema.Transaction`."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    request_id: str | None

    @classmethod
    def from_row(cls, p: Payment) -> PaymentDecision:
        return cls(
            payment_id=p.payment_id,
            transaction_id=p.transaction_id,
            decision=p.decision,  # constrained on write
            score=p.score,
            scorer=p.scorer,
            model_version=p.model_version,
            request_id=p.request_id,
        )


Label = Literal["fraud", "legit"]


class Feedback(BaseModel):
    """Delayed ground truth for a payment: a chargeback or a confirmation."""

    model_config = ConfigDict(extra="forbid")

    label: Label
    # Chargeback reason (for fraud). The simulator sends the fraud pattern.
    reason: str | None = Field(default=None, min_length=1, max_length=32)
    source: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9._:-]+$")

    @model_validator(mode="after")
    def _reason_only_for_fraud(self) -> Self:
        if self.label == "legit" and self.reason is not None:
            raise ValueError("reason is only meaningful for label 'fraud'")
        return self


class PaymentRecord(PaymentDecision):
    transaction: Transaction
    features: dict[str, float]
    created_at: datetime
    label: Label | None
    label_reason: str | None
    label_source: str | None
    labelled_at: datetime | None

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
            label=p.label,  # constrained on write
            label_reason=p.label_reason,
            label_source=p.label_source,
            labelled_at=as_utc(p.labelled_at) if p.labelled_at is not None else None,
        )
