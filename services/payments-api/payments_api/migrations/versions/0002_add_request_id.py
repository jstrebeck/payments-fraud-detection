"""add payments.request_id

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: rows scored before correlation IDs existed have none.
    with op.batch_alter_table("payments") as batch:
        batch.add_column(sa.Column("request_id", sa.String(length=64), nullable=True))
        batch.create_index("ix_payments_request_id", ["request_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("payments") as batch:
        batch.drop_index("ix_payments_request_id")
        batch.drop_column("request_id")
