"""Database connection management.

SQLite configuration
--------------------
SQLite is the default because a local NAS run needs durable, queryable storage with zero
operational overhead. Four pragmas are set on every connection, and each earns its place:

``journal_mode=WAL``
    Write-ahead logging lets readers proceed while a writer holds the write lock. Without
    it, the CLI cannot inspect a search while that search is running.
``foreign_keys=ON``
    SQLite ignores foreign keys unless explicitly enabled, per connection. Without this,
    the cascade deletes declared in the ORM silently do nothing and deleting a search
    leaves orphan candidates behind.
``busy_timeout=<ms>``
    With multiprocessing workers, write-lock contention is normal. Without a timeout,
    SQLite raises ``database is locked`` immediately; with one it waits and usually
    succeeds.
``synchronous=NORMAL``
    Under WAL this is durable against application crashes (the failure mode that actually
    happens) while avoiding an fsync per transaction. ``FULL`` protects against OS-level
    power loss mid-write, at a large throughput cost that is not worth paying for
    reproducible experiment metadata.

Sessions
--------
:meth:`Database.session` is a context manager that commits on success and rolls back on
any exception. Every repository method runs inside one, so a failed write can never leave
a half-updated candidate — the transaction is the unit of consistency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nas_engine.exceptions import PersistenceError
from nas_engine.observability.logging import get_logger
from nas_engine.persistence.models import Base

_LOGGER = get_logger(__name__)

#: Default time a connection waits for a write lock before failing, in milliseconds.
DEFAULT_BUSY_TIMEOUT_MS: int = 30_000

#: URL used for a purely in-memory database.
IN_MEMORY_URL: str = "sqlite+pysqlite:///:memory:"


def _configure_sqlite(dbapi_connection: Any, _record: Any, *, busy_timeout_ms: int) -> None:
    """Apply per-connection SQLite pragmas.

    Args:
        dbapi_connection: The raw DBAPI connection.
        _record: SQLAlchemy connection record (unused).
        busy_timeout_ms: Write-lock wait in milliseconds.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    finally:
        cursor.close()


class Database:
    """Owns the SQLAlchemy engine and session factory for one database.

    Args:
        url: SQLAlchemy URL. Use :data:`IN_MEMORY_URL` for an ephemeral database.
        echo: Whether to log every statement; useful when debugging queries.
        busy_timeout_ms: SQLite write-lock wait.

    Raises:
        PersistenceError: If the engine cannot be created.
    """

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self._url = url
        self._busy_timeout_ms = busy_timeout_ms
        connect_args: dict[str, Any] = {}
        engine_kwargs: dict[str, Any] = {"echo": echo, "future": True}

        if url.startswith("sqlite"):
            # `check_same_thread=False` is required because a session may be created on one
            # thread and used on another during report generation. SQLAlchemy's pooling
            # still serialises access, so this does not introduce a data race.
            connect_args["check_same_thread"] = False
            if ":memory:" in url:
                # An in-memory database lives inside one connection. StaticPool reuses the
                # single connection so that tables created by one session are visible to
                # the next — without it every session gets an empty database.
                engine_kwargs["poolclass"] = StaticPool

        try:
            self._engine: Engine = create_engine(url, connect_args=connect_args, **engine_kwargs)
        except (SQLAlchemyError, ValueError) as exc:
            msg = f"could not create a database engine for {url!r}: {exc}"
            raise PersistenceError(msg, details={"url": url, "error": str(exc)}) from exc

        if url.startswith("sqlite"):
            event.listen(
                self._engine,
                "connect",
                lambda connection, record: _configure_sqlite(
                    connection, record, busy_timeout_ms=busy_timeout_ms
                ),
            )

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    # -- properties ----------------------------------------------------------------
    @property
    def url(self) -> str:
        """The database URL."""
        return self._url

    @property
    def engine(self) -> Engine:
        """The underlying SQLAlchemy engine."""
        return self._engine

    # -- lifecycle -----------------------------------------------------------------
    @classmethod
    def from_path(
        cls, path: Path, *, echo: bool = False, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
    ) -> Database:
        """Build a SQLite database at ``path``, creating parent directories.

        Args:
            path: Database file path.
            echo: Whether to log statements.
            busy_timeout_ms: SQLite write-lock wait.

        Returns:
            The database handle.

        Raises:
            PersistenceError: If the parent directory cannot be created.
        """
        resolved = Path(path).expanduser()
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"cannot create database directory {resolved.parent}: {exc}"
            raise PersistenceError(msg, details={"path": str(resolved), "error": str(exc)}) from exc
        return cls(f"sqlite+pysqlite:///{resolved}", echo=echo, busy_timeout_ms=busy_timeout_ms)

    @classmethod
    def in_memory(cls, *, echo: bool = False) -> Database:
        """Build an ephemeral in-memory database, used by tests.

        Args:
            echo: Whether to log statements.

        Returns:
            The database handle.
        """
        return cls(IN_MEMORY_URL, echo=echo)

    def create_all(self) -> None:
        """Create every table declared on :class:`~nas_engine.persistence.models.Base`.

        Raises:
            PersistenceError: If schema creation fails.
        """
        try:
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as exc:
            msg = f"failed to create database schema: {exc}"
            raise PersistenceError(msg, details={"error": str(exc)}) from exc

    def dispose(self) -> None:
        """Close every pooled connection.

        Called when a process finishes with the database. Failing to dispose leaves SQLite
        file handles open, which on Windows prevents the file from being deleted and in
        tests causes confusing teardown errors.
        """
        self._engine.dispose()

    # -- sessions ------------------------------------------------------------------
    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session inside a transaction.

        The transaction commits when the block exits normally and rolls back on any
        exception, so a partially applied write is impossible.

        Yields:
            An open :class:`sqlalchemy.orm.Session`.

        Raises:
            PersistenceError: If the commit fails.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            msg = f"database transaction failed and was rolled back: {exc}"
            raise PersistenceError(msg, details={"error": str(exc)}) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Yield a raw connection inside a transaction, for migrations.

        Yields:
            An open :class:`sqlalchemy.engine.Connection`.

        Raises:
            PersistenceError: If the transaction fails.
        """
        try:
            with self._engine.begin() as connection:
                yield connection
        except SQLAlchemyError as exc:
            msg = f"database connection failed: {exc}"
            raise PersistenceError(msg, details={"error": str(exc)}) from exc

    def __enter__(self) -> Database:
        """Enter a context that disposes the engine on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Dispose the engine."""
        self.dispose()


__all__ = ["DEFAULT_BUSY_TIMEOUT_MS", "IN_MEMORY_URL", "Database"]
