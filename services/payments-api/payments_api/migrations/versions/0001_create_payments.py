"""create payments

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "payments",
        sa.Column("payment_id", sa.Uuid(), nullable=False),
        sa.Column("transaction_id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("card_token", sa.String(length=32), nullable=False),
        sa.Column("customer_id", sa.String(length=32), nullable=False),
        sa.Column("merchant_id", sa.String(length=32), nullable=False),
        sa.Column("merchant_category", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2, asdecimal=False), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column("device_id", sa.String(length=32), nullable=False),
        sa.Column("ip_country", sa.String(length=2), nullable=False),
        sa.Column("billing_country", sa.String(length=2), nullable=False),
        sa.Column(
            "features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("scorer", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("payment_id", name=op.f("pk_payments")),
        sa.UniqueConstraint("transaction_id", name=op.f("uq_payments_transaction_id")),
    )
    op.create_index(
        "ix_payments_card_token_timestamp", "payments", ["card_token", "timestamp"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_payments_card_token_timestamp", table_name="payments")
    op.drop_table("payments")
