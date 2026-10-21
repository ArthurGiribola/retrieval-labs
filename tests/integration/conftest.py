"""Fixtures for tests against a real PostgreSQL + pgvector server.

They run only when ``TEST_DATABASE_URL`` is set, and refuse any database whose name does not
end in ``_test``, because every run drops and recreates the schema.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.db.models import Base
from retrieval_labs.db.session import make_session_factory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail(f"refusing to wipe database {database!r}: its name must end in '_test'")
    return url


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["url"] = url
    config.attributes["configure_logger"] = False
    return config


def drop_everything(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))


@pytest.fixture(scope="session")
def migrations(test_database_url: str) -> Config:
    return alembic_config(test_database_url)


@pytest.fixture(scope="session")
def engine(test_database_url: str, migrations: Config) -> Iterator[Engine]:
    engine = create_engine(test_database_url)
    drop_everything(engine)
    command.upgrade(migrations, "head")
    yield engine
    engine.dispose()


def truncate_all(engine: Engine) -> None:
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} CASCADE"))


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session
    truncate_all(engine)


@pytest.fixture
def sessions(engine: Engine) -> Iterator[sessionmaker[Session]]:
    """Session factory, as the worker uses it; tables are emptied afterwards."""
    yield make_session_factory(engine)
    truncate_all(engine)
