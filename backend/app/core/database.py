"""
Database connection setup (Phase 10).

Uses SQLAlchemy. The database is chosen by the DATABASE_URL setting:

  - PostgreSQL, e.g. postgresql+psycopg2://juris_user:PASSWORD@localhost:5432/juris_db
  - If DATABASE_URL is empty, a local SQLite file (data/metadata/juris.db)
    is used instead, so the project runs without a database server.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger("juris.database")


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class DatabaseError(Exception):
    """Raised when the database cannot be reached or initialised."""


_engine: Engine | None = None
_session_factory = None
_initialized = False


def get_database_url() -> str:
    """Return the configured database URL, or a SQLite fallback if none is set."""
    settings = get_settings()
    configured = settings.database_url.strip()
    if configured:
        return configured

    settings.data_metadata_path.mkdir(parents=True, exist_ok=True)
    sqlite_file = settings.data_metadata_path / "juris.db"
    return f"sqlite:///{sqlite_file.as_posix()}"


def get_engine() -> Engine:
    """Create (once) and return the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_database_url()
        options: dict = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            # FastAPI runs sync endpoints in a thread pool.
            options["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **options)
        logger.info(
            "Database engine created for %s",
            make_url(url).render_as_string(hide_password=True),
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session; commit on success, roll back on error, always close."""
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create any missing tables. Safe to call repeatedly."""
    global _initialized
    if _initialized:
        return

    from app.models import document  # noqa: F401  (registers the documents table)

    try:
        Base.metadata.create_all(get_engine())
    except SQLAlchemyError as exc:
        logger.error(
            "Could not initialise the database: %s. "
            "Check DATABASE_URL and that PostgreSQL is running.",
            exc,
        )
        raise DatabaseError(
            "Database is unavailable. Check DATABASE_URL and that PostgreSQL is running."
        ) from exc

    _initialized = True
    logger.info("Database tables are ready")