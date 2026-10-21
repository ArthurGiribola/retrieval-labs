"""Alembic environment: migrates the database named by ``DATABASE_URL``.

Tests pass a URL through ``config.attributes["url"]`` instead, so they never touch the
application database by accident.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from retrieval_labs.db.models import Base
from retrieval_labs.db.session import database_url

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.attributes.get("url") or database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(config.attributes.get("url") or database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
