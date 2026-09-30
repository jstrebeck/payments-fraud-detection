"""add delayed labels to payments

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # All nullable: a payment has no label until feedback arrives, maybe never.
    with op.batch_alter_table("payments") as batch:
        batch.add_column(sa.Column("label", sa.String(length=8), nullable=True))
        batch.add_column(sa.Column("label_reason", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("label_source", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("labelled_at", sa.DateTime(timezone=True), nullable=True))
    # Partial index: retraining reads only labelled rows, newest first.
    op.create_index(
        "ix_payments_labelled_created_at",
        "payments",
        ["created_at"],
        unique=False,
        postgresql_where=sa.text("label IS NOT NULL"),
        sqlite_where=sa.text("label IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_payments_labelled_created_at", table_name="payments")
    with op.batch_alter_table("payments") as batch:
        batch.drop_column("labelled_at")
        batch.drop_column("label_source")
        batch.drop_column("label_reason")
        batch.drop_column("label")
