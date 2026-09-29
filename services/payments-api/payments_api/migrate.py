"""Apply database migrations. Installed as the `payments-api-migrate` command.

Run by the compose `migrate` service, and later by an init container.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from payments_api.config import Settings


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", "payments_api:migrations")
    url = database_url or Settings().database_url
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def main() -> None:
    command.upgrade(alembic_config(), "head")


if __name__ == "__main__":
    main()
