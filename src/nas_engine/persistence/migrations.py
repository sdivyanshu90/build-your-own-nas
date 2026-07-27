"""Versioned schema management.

Why not just ``create_all``?
----------------------------
``Base.metadata.create_all`` creates tables that do not exist. It does **not** alter
tables that do exist. A user who upgrades the package and opens last month's database gets
a schema missing the new column and a stack trace from deep inside SQLAlchemy, hundreds of
lines from the actual cause.

A migration list fixes that. Each migration has a version number, a description, and an
upgrade function. On connect, the applied version is compared to the target:

* **Equal** — nothing to do.
* **Lower** — apply the missing migrations in order and record the new version.
* **Higher** — the database was written by a newer build. Refuse to touch it, with an
  error saying so. Silently downgrading would corrupt data.

Why not Alembic?
----------------
Alembic is the right answer for a service with a long-lived production database and a
team. Here it adds a dependency, a config file, a versions directory, and an autogenerate
workflow, in exchange for handling a migration cadence that this project does not have.
The trade-off is recorded in ``docs/adr/0002-persistence-layer.md``; if the schema starts
changing often, Alembic is the documented next step and this module's version table maps
onto its ``alembic_version`` table directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select

from nas_engine.exceptions import PersistenceError, SchemaVersionError
from nas_engine.observability.logging import get_logger
from nas_engine.persistence.database import Database
from nas_engine.persistence.models import Base, SchemaVersionRecord
from nas_engine.utilities.timing import utc_now

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Migration:
    """One schema migration.

    Attributes:
        version: Version this migration brings the database to.
        description: What the migration does, recorded in the version table.
        upgrade: Callable applying the change.
    """

    version: int
    description: str
    upgrade: Callable[[Database], None]


def _create_initial_schema(database: Database) -> None:
    """Create the version-1 schema.

    Args:
        database: Database to create tables in.
    """
    Base.metadata.create_all(database.engine)


#: Ordered migrations. Append only; never edit a released migration, because databases in
#: the field have already applied it and would silently diverge.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="initial schema: searches, candidates, trials, metrics, artifacts, "
        "checkpoints, events",
        upgrade=_create_initial_schema,
    ),
)

#: The schema version this build expects.
TARGET_SCHEMA_VERSION: int = max(migration.version for migration in MIGRATIONS)


def current_version(database: Database) -> int:
    """Return the schema version recorded in the database.

    Args:
        database: Database to inspect.

    Returns:
        The applied version, or ``0`` when the database is empty or unversioned.
    """
    from sqlalchemy import inspect

    inspector = inspect(database.engine)
    if not inspector.has_table(SchemaVersionRecord.__tablename__):
        return 0
    with database.session() as session:
        record = session.scalars(
            select(SchemaVersionRecord).where(SchemaVersionRecord.id == 1)
        ).one_or_none()
        return int(record.version) if record is not None else 0


def _record_version(database: Database, version: int, description: str) -> None:
    """Write the applied schema version.

    Args:
        database: Database to update.
        version: Version just applied.
        description: Migration description.
    """
    with database.session() as session:
        record = session.get(SchemaVersionRecord, 1)
        if record is None:
            session.add(
                SchemaVersionRecord(
                    id=1, version=version, applied_at=utc_now(), description=description
                )
            )
        else:
            record.version = version
            record.applied_at = utc_now()
            record.description = description


def apply_migrations(database: Database, *, target: int | None = None) -> int:
    """Bring the database up to the target schema version.

    Args:
        database: Database to migrate.
        target: Version to migrate to; defaults to :data:`TARGET_SCHEMA_VERSION`.

    Returns:
        The version the database is at after migrating.

    Raises:
        SchemaVersionError: If the database is newer than this build supports.
        PersistenceError: If a migration fails.
    """
    goal = target if target is not None else TARGET_SCHEMA_VERSION
    applied = current_version(database)

    if applied > goal:
        msg = (
            f"the database is at schema version {applied} but this build of nas-engine "
            f"supports at most version {goal}. Upgrade nas-engine, or point at a different "
            "database file. Downgrading a schema is not supported and would lose data."
        )
        raise SchemaVersionError(msg, details={"database": applied, "supported": goal})

    if applied == goal:
        return applied

    for migration in MIGRATIONS:
        if migration.version <= applied or migration.version > goal:
            continue
        _LOGGER.info(
            "database.migrating",
            from_version=applied,
            to_version=migration.version,
            description=migration.description,
        )
        try:
            migration.upgrade(database)
        except PersistenceError:
            raise
        except Exception as exc:
            msg = (
                f"migration to schema version {migration.version} failed: {exc}. The "
                "database is unchanged; restore from a backup if it is inconsistent."
            )
            raise PersistenceError(
                msg, details={"version": migration.version, "error": str(exc)}
            ) from exc
        _record_version(database, migration.version, migration.description)
        applied = migration.version

    return applied


def ensure_schema(database: Database) -> int:
    """Create or migrate the schema, returning the resulting version.

    The single entry point every caller uses: it is safe on a fresh file, on an
    up-to-date database, and on an out-of-date one.

    Args:
        database: Database to prepare.

    Returns:
        The schema version now in effect.

    Raises:
        SchemaVersionError: If the database is newer than this build supports.
        PersistenceError: If a migration fails.
    """
    return apply_migrations(database)


__all__ = [
    "MIGRATIONS",
    "TARGET_SCHEMA_VERSION",
    "Migration",
    "apply_migrations",
    "current_version",
    "ensure_schema",
]
