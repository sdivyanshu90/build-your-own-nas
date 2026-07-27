"""SQLAlchemy ORM models: the persisted shape of a search.

Design notes
------------
**String primary keys.** Searches, candidates, and trials use opaque string ids (UUID
hex). Auto-increment integers would be assigned by the database, which means a worker
cannot know a record's id until it has committed — awkward for logging and impossible for
artifact paths chosen before the write. String ids are generated in the process, so an id
exists before the row does.

**Timezone-aware timestamps everywhere.** SQLite has no native timestamp type and
discards timezone information, so :class:`UTCDateTime` re-attaches UTC on read. Naive
datetimes compare incorrectly across processes and are ambiguous once exported.

**JSON columns for structured payloads.** Architecture specifications, budgets, and error
details are nested documents with a schema that belongs to the application, not the
database. Storing them as JSON keeps the relational schema stable while the domain evolves.
They are always written through :func:`~nas_engine.utilities.json_io.canonical_json_dumps`
and read back with validation, so a corrupt or tampered payload fails loudly.

**Metrics in their own table, not JSON.** Metrics *are* queried, aggregated, and ranked by
the database, which a JSON blob makes awkward and slow. One row per ``(trial, name)`` keeps
``SELECT ... ORDER BY value`` a plain index scan.

**Cascade deletes.** Deleting a search removes its candidates, trials, metrics, and
artifact records. Without cascades a deleted search leaves orphan rows that corrupt every
aggregate query.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nas_engine.utilities.timing import utc_now


def new_id() -> str:
    """Return a fresh 32-character hexadecimal identifier.

    UUID4 rather than a sequence: identifiers must be generatable in a worker process
    without coordinating with the database or with other workers.

    Returns:
        A random hex identifier.
    """
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator[datetime]):
    """A ``DateTime`` column that always round-trips timezone-aware UTC values.

    SQLite stores datetimes as strings and returns them naive. Without this decorator a
    timestamp written as ``2026-01-01T00:00:00+00:00`` would come back as
    ``2026-01-01 00:00:00`` with no timezone, and comparing it against an aware datetime
    raises ``TypeError`` at some distant call site.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Normalise a Python value to aware UTC before writing.

        Args:
            value: Value being written.
            dialect: Active SQLAlchemy dialect.

        Returns:
            An aware UTC datetime, or ``None``.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Re-attach UTC to a naive value read from the database.

        Args:
            value: Value read from the database.
            dialect: Active SQLAlchemy dialect.

        Returns:
            An aware UTC datetime, or ``None``.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


class SearchStatus(str, Enum):
    """Lifecycle status of a whole search run.

    Members:
        CREATED: The run exists but no candidate has been proposed.
        RUNNING: Evaluations are in progress.
        PAUSED: The run was interrupted cleanly and can be resumed.
        COMPLETED: The stopping condition was reached.
        FAILED: The run stopped because of an engine-level error.
        CANCELLED: The operator cancelled the run.
    """

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchRecord(Base):
    """One search run.

    Attributes:
        id: Opaque search identifier.
        name: Human-readable name from configuration.
        strategy: Strategy name.
        status: Lifecycle status.
        config_json: The full validated configuration as JSON.
        config_hash: Hash of the configuration, used to detect edits across a resume.
        config_version: Configuration schema version.
        search_space_json: The search space as JSON.
        seed: Master random seed.
        seeds_json: Derived component seeds.
        environment_json: Environment snapshot.
        planned_evaluations: Evaluation budget from configuration.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        started_at: When the first candidate was proposed.
        completed_at: When the run finished.
        notes: Free-text operator notes.
        candidates: Candidates belonging to this run.
        checkpoints: Checkpoints written for this run.
        events: Audit events for this run.
    """

    __tablename__ = "searches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SearchStatus.CREATED.value, index=True
    )
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    config_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    search_space_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    seeds_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    environment_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    planned_evaluations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    candidates: Mapped[list[CandidateRecord]] = relationship(
        back_populates="search", cascade="all, delete-orphan", lazy="selectin"
    )
    checkpoints: Mapped[list[CheckpointRecord]] = relationship(
        back_populates="search", cascade="all, delete-orphan", lazy="selectin"
    )
    events: Mapped[list[SearchEventRecord]] = relationship(
        back_populates="search", cascade="all, delete-orphan", lazy="selectin"
    )


class CandidateRecord(Base):
    """One architecture proposed within a search.

    A candidate is identified within its search by ``(architecture_hash, rung)``. The rung
    is part of the key because multi-fidelity search deliberately re-evaluates the same
    architecture at a larger budget, and those are genuinely different measurements.

    Attributes:
        id: Opaque candidate identifier.
        search_id: Owning search.
        architecture_hash: Canonical architecture hash.
        rung: Fidelity rung; ``0`` for single-fidelity search.
        spec_json: Canonical architecture JSON.
        status: Candidate lifecycle state.
        parent_id: Parent candidate for evolutionary lineage.
        mutation: Description of the mutation that produced this candidate.
        origin: How the candidate was produced.
        generation: Strategy-assigned generation index.
        objective_value: Cached online scalar fitness.
        retry_count: Number of retries consumed.
        error_json: Last failure record.
        metadata_json: Strategy-specific extra data.
        created_at: Proposal timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("search_id", "architecture_hash", "rung", name="uq_candidate_identity"),
        Index("ix_candidates_search_status", "search_id", "status"),
        Index("ix_candidates_objective", "search_id", "objective_value"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    search_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("searches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    architecture_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rung: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True
    )
    mutation: Mapped[str | None] = mapped_column(String(200), nullable=True)
    origin: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    objective_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now, onupdate=utc_now
    )

    search: Mapped[SearchRecord] = relationship(back_populates="candidates")
    trials: Mapped[list[TrialRecord]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="selectin"
    )
    artifacts: Mapped[list[ArtifactRecord]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", lazy="selectin"
    )


class TrialRecord(Base):
    """One evaluation attempt for a candidate.

    A candidate can have several trials: the first attempt plus any retries. Keeping them
    as separate rows preserves the history — how often a candidate failed and why is
    exactly the information needed to debug a flaky search.

    Attributes:
        id: Opaque trial identifier.
        candidate_id: Owning candidate.
        attempt: Zero-based attempt number.
        budget_json: The budget the trial ran at.
        status: Trial outcome status.
        worker_id: Worker that ran the trial.
        device: Device string.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        duration_seconds: Wall-clock duration.
        error_json: Failure record when the trial failed.
        training_json: Serialised training outcome.
    """

    __tablename__ = "trials"
    __table_args__ = (
        UniqueConstraint("candidate_id", "attempt", name="uq_trial_attempt"),
        Index("ix_trials_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    candidate_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    budget_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    device: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    training_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    candidate: Mapped[CandidateRecord] = relationship(back_populates="trials")
    metrics: Mapped[list[MetricRecord]] = relationship(
        back_populates="trial", cascade="all, delete-orphan", lazy="selectin"
    )


class MetricRecord(Base):
    """One measured scalar for one trial.

    Attributes:
        id: Surrogate key.
        trial_id: Owning trial.
        name: Metric name, matching the keys objectives refer to.
        value: Measured value.
    """

    __tablename__ = "metrics"
    __table_args__ = (
        UniqueConstraint("trial_id", "name", name="uq_metric_name"),
        Index("ix_metrics_name_value", "name", "value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("trials.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    trial: Mapped[TrialRecord] = relationship(back_populates="metrics")


class ArtifactRecord(Base):
    """A file produced by an evaluation.

    Only the *path* is stored, never the bytes. A database holding 50 MB weight blobs is
    slow to query, slow to back up, and awkward to inspect; the filesystem is the right
    store for large binaries. Paths are relative to the configured artifact root so a run
    directory can be moved or archived without rewriting the database.

    Attributes:
        id: Surrogate key.
        candidate_id: Owning candidate.
        trial_id: Trial that produced the artifact, when applicable.
        kind: Artifact kind, e.g. ``"weights"`` or ``"training_checkpoint"``.
        path: Path relative to the artifact root.
        size_bytes: File size at creation time.
        created_at: Creation timestamp.
    """

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("candidate_id", "kind", "path", name="uq_artifact_identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trial_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("trials.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)

    candidate: Mapped[CandidateRecord] = relationship(back_populates="artifacts")


class CheckpointRecord(Base):
    """A snapshot of search-strategy state.

    Attributes:
        id: Surrogate key.
        search_id: Owning search.
        sequence: Monotonically increasing checkpoint number.
        format_version: Checkpoint payload format version.
        payload_json: The strategy's ``state_dict`` plus engine bookkeeping.
        created_at: Creation timestamp.
    """

    __tablename__ = "checkpoints"
    __table_args__ = (UniqueConstraint("search_id", "sequence", name="uq_checkpoint_sequence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    search_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("searches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    format_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)

    search: Mapped[SearchRecord] = relationship(back_populates="checkpoints")


class SearchEventRecord(Base):
    """An audit event in a search's history.

    Structured logs go to stderr and are ephemeral. This table keeps the events that
    matter for *explaining a result later*: when the run started, when it was resumed,
    what was pruned and why. It is small — one short row per lifecycle event, not one per
    training step.

    Attributes:
        id: Surrogate key.
        search_id: Owning search.
        event: Event name from :class:`~nas_engine.observability.events.Event`.
        candidate_id: Related candidate, when applicable.
        payload_json: Event-specific structured context.
        created_at: Event timestamp.
    """

    __tablename__ = "search_events"
    __table_args__ = (Index("ix_events_search_created", "search_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    search_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("searches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    candidate_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)

    search: Mapped[SearchRecord] = relationship(back_populates="events")


class SchemaVersionRecord(Base):
    """The applied database schema version.

    A single-row table (enforced by the primary key) recording which migration the
    database is at. Checking it on connect turns "mysterious missing column" errors into
    "your database is at version 1, this build needs version 2, run the migration".

    Attributes:
        id: Always ``1``.
        version: Applied schema version.
        applied_at: When the version was applied.
        description: What the migration did.
    """

    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)
    description: Mapped[str] = mapped_column(String(200), nullable=False, default="")


__all__ = [
    "ArtifactRecord",
    "Base",
    "CandidateRecord",
    "CheckpointRecord",
    "MetricRecord",
    "SchemaVersionRecord",
    "SearchEventRecord",
    "SearchRecord",
    "SearchStatus",
    "TrialRecord",
    "UTCDateTime",
    "new_id",
]
