"""Database tables. Schema changes go through Alembic (`migrations/`)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, MetaData, Numeric, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Payment(Base):
    """One scored transaction: the input, the features, and the decision.

    Doubles as the card history the API reads to build `CardContext`, and
    (Phase 7) as the labelled dataset for retraining.
    """

    __tablename__ = "payments"
    __table_args__ = (Index("ix_payments_card_token_timestamp", "card_token", "timestamp"),)

    payment_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True)

    # Transaction fields (ml.data.schema.Transaction)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    card_token: Mapped[str] = mapped_column(String(32))
    customer_id: Mapped[str] = mapped_column(String(32))
    merchant_id: Mapped[str] = mapped_column(String(32))
    merchant_category: Mapped[str] = mapped_column(String(32))
    amount: Mapped[float] = mapped_column(Numeric(12, 2, asdecimal=False))
    currency: Mapped[str] = mapped_column(String(3))
    channel: Mapped[str] = mapped_column(String(16))
    device_id: Mapped[str] = mapped_column(String(32))
    ip_country: Mapped[str] = mapped_column(String(2))
    billing_country: Mapped[str] = mapped_column(String(2))

    # Scoring outcome
    features: Mapped[dict[str, float]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16))
    scorer: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    # Correlation ID of the request that created the row (x-request-id); lets one
    # payment be followed through simulator, API and predictor logs.
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def as_utc(ts: datetime) -> datetime:
    """SQLite drops tzinfo; Postgres returns aware values. Normalise both to UTC."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
