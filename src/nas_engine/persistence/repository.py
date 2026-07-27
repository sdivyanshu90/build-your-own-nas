"""The repository: the only place that talks SQL.

Why a repository
----------------
Without one, ``session.query(...)`` calls spread across the engine, the CLI, and the
report generator. Three consequences follow, all bad: the schema can no longer change
without touching unrelated modules; testing any of them requires a database; and
transaction boundaries end up implicit and inconsistent.

The repository confines persistence to a single seam. Everything above it speaks in domain
objects and never sees a :class:`~sqlalchemy.orm.Session`.

Detached read models
--------------------
Every query returns a frozen dataclass, never an ORM instance. ORM objects are bound to the
session that loaded them; touching a lazily-loaded attribute after the session closes
raises ``DetachedInstanceError`` at a call site far from the cause. Returning plain data
makes that failure impossible and keeps the domain free of SQLAlchemy types.

Transactions
------------
Each public method runs inside exactly one transaction, via
:meth:`~nas_engine.persistence.database.Database.session`. A method either fully applies or
fully rolls back. Multi-step operations that must be atomic — claiming a queued candidate,
recording a completed trial with its metrics and artifacts — are single methods for exactly
this reason.

Safety
------
No SQL is ever built by string interpolation. Every query goes through SQLAlchemy's
expression language, which parameterises values, so a hostile architecture hash or search
name cannot alter a statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from nas_engine.architectures.lineage import LineageNode
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import (
    DuplicateRecordError,
    PersistenceError,
    RecordNotFoundError,
)
from nas_engine.observability.logging import get_logger
from nas_engine.orchestration.lifecycle import CandidateState, TrialState, validate_transition
from nas_engine.persistence.database import Database
from nas_engine.persistence.models import (
    ArtifactRecord,
    CandidateRecord,
    CheckpointRecord,
    MetricRecord,
    SearchEventRecord,
    SearchRecord,
    SearchStatus,
    TrialRecord,
    new_id,
)
from nas_engine.utilities.timing import utc_now

_LOGGER = get_logger(__name__)

#: Version of the search-checkpoint payload envelope written by the engine.
SEARCH_CHECKPOINT_VERSION: int = 1


@dataclass(frozen=True)
class SearchSummary:
    """A detached view of a search run.

    Attributes:
        id: Search identifier.
        name: Human-readable name.
        strategy: Strategy name.
        status: Lifecycle status.
        seed: Master seed.
        config_hash: Configuration hash.
        config_version: Configuration schema version.
        planned_evaluations: Evaluation budget.
        created_at: Creation timestamp.
        started_at: First-proposal timestamp.
        completed_at: Completion timestamp.
        notes: Operator notes.
    """

    id: str
    name: str
    strategy: str
    status: str
    seed: int
    config_hash: str
    config_version: int
    planned_evaluations: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    notes: str = ""

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration, or ``None`` when the run has not finished."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "name": self.name,
            "strategy": self.strategy,
            "status": self.status,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "config_version": self.config_version,
            "planned_evaluations": self.planned_evaluations,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CandidateSummary:
    """A detached view of a candidate, including its best trial's metrics.

    Attributes:
        id: Candidate identifier.
        search_id: Owning search.
        architecture_hash: Canonical hash.
        rung: Fidelity rung.
        status: Lifecycle state.
        origin: How the candidate was produced.
        parent_id: Parent candidate id.
        mutation: Mutation description.
        generation: Strategy generation index.
        objective_value: Cached online scalar fitness.
        retry_count: Retries consumed.
        metrics: Metrics from the most recent completed trial.
        error: Last failure record.
        created_at: Proposal timestamp.
        updated_at: Last update timestamp.
        trial_count: Number of attempts recorded.
        artifacts: Artifact kind to relative path.
    """

    id: str
    search_id: str
    architecture_hash: str
    rung: int
    status: str
    origin: str
    parent_id: str | None
    mutation: str | None
    generation: int | None
    objective_value: float | None
    retry_count: int
    metrics: dict[str, float]
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    trial_count: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "search_id": self.search_id,
            "architecture_hash": self.architecture_hash,
            "rung": self.rung,
            "status": self.status,
            "origin": self.origin,
            "parent_id": self.parent_id,
            "mutation": self.mutation,
            "generation": self.generation,
            "objective_value": self.objective_value,
            "retry_count": self.retry_count,
            "metrics": dict(self.metrics),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trial_count": self.trial_count,
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class RecoveryReport:
    """What a recovery sweep found and repaired after an interrupted run.

    Attributes:
        interrupted_running: Candidates found in ``RUNNING`` and returned to the queue.
        interrupted_trials: Trials marked ``INTERRUPTED``.
        requeued: Candidate ids returned to ``QUEUED``.
        abandoned: Candidate ids moved to ``FAILED`` because retries were exhausted.
        completed: Candidates already finished, left untouched.
    """

    interrupted_running: int = 0
    interrupted_trials: int = 0
    requeued: tuple[str, ...] = ()
    abandoned: tuple[str, ...] = ()
    completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "interrupted_running": self.interrupted_running,
            "interrupted_trials": self.interrupted_trials,
            "requeued": list(self.requeued),
            "abandoned": list(self.abandoned),
            "completed": self.completed,
        }


def _metrics_of(trial: TrialRecord) -> dict[str, float]:
    """Extract a metric mapping from a trial record."""
    return {metric.name: float(metric.value) for metric in trial.metrics}


def _latest_completed_trial(candidate: CandidateRecord) -> TrialRecord | None:
    """Return the most recent successful trial for a candidate, if any."""
    completed = [trial for trial in candidate.trials if trial.status == TrialState.COMPLETED.value]
    if not completed:
        return None
    return max(completed, key=lambda trial: (trial.attempt, trial.started_at))


def _to_candidate_summary(candidate: CandidateRecord) -> CandidateSummary:
    """Convert an ORM candidate into a detached summary."""
    trial = _latest_completed_trial(candidate)
    return CandidateSummary(
        id=candidate.id,
        search_id=candidate.search_id,
        architecture_hash=candidate.architecture_hash,
        rung=candidate.rung,
        status=candidate.status,
        origin=candidate.origin,
        parent_id=candidate.parent_id,
        mutation=candidate.mutation,
        generation=candidate.generation,
        objective_value=candidate.objective_value,
        retry_count=candidate.retry_count,
        metrics=_metrics_of(trial) if trial is not None else {},
        error=candidate.error_json,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
        trial_count=len(candidate.trials),
        artifacts={artifact.kind: artifact.path for artifact in candidate.artifacts},
    )


def _to_search_summary(record: SearchRecord) -> SearchSummary:
    """Convert an ORM search into a detached summary."""
    return SearchSummary(
        id=record.id,
        name=record.name,
        strategy=record.strategy,
        status=record.status,
        seed=record.seed,
        config_hash=record.config_hash,
        config_version=record.config_version,
        planned_evaluations=record.planned_evaluations,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        notes=record.notes,
    )


class SearchRepository:
    """Persistence operations for searches, candidates, trials, and artifacts.

    Args:
        database: The database to operate on. The schema must already exist; call
            :func:`~nas_engine.persistence.migrations.ensure_schema` first.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        """The underlying database handle."""
        return self._database

    # ---------------------------------------------------------------- searches ----
    def create_search(
        self,
        *,
        name: str,
        strategy: str,
        config: dict[str, Any],
        config_hash: str,
        config_version: int,
        search_space: dict[str, Any],
        seed: int,
        seeds: dict[str, int],
        environment: dict[str, Any],
        planned_evaluations: int,
        search_id: str | None = None,
    ) -> str:
        """Create a new search run.

        Args:
            name: Human-readable name.
            strategy: Strategy name.
            config: Validated configuration as plain data.
            config_hash: Hash of the configuration.
            config_version: Configuration schema version.
            search_space: The search space as plain data.
            seed: Master seed.
            seeds: Derived component seeds.
            environment: Environment snapshot.
            planned_evaluations: Evaluation budget.
            search_id: Explicit identifier; generated when omitted.

        Returns:
            The search identifier.

        Raises:
            DuplicateRecordError: If ``search_id`` already exists.
        """
        identifier = search_id or new_id()
        with self._database.session() as session:
            if session.get(SearchRecord, identifier) is not None:
                msg = f"a search with id {identifier} already exists"
                raise DuplicateRecordError(msg, details={"search_id": identifier})
            session.add(
                SearchRecord(
                    id=identifier,
                    name=name,
                    strategy=strategy,
                    status=SearchStatus.CREATED.value,
                    config_json=config,
                    config_hash=config_hash,
                    config_version=config_version,
                    search_space_json=search_space,
                    seed=seed,
                    seeds_json=dict(seeds),
                    environment_json=environment,
                    planned_evaluations=planned_evaluations,
                )
            )
        _LOGGER.info(
            "repository.search_created",
            search_id=identifier,
            name=name,
            strategy=strategy,
            config_hash=config_hash,
        )
        return identifier

    def get_search(self, search_id: str) -> SearchSummary:
        """Return one search.

        Args:
            search_id: Identifier to look up.

        Returns:
            The summary.

        Raises:
            RecordNotFoundError: If the search does not exist.
        """
        with self._database.session() as session:
            record = session.get(SearchRecord, search_id)
            if record is None:
                msg = f"no search found with id {search_id!r}"
                raise RecordNotFoundError(msg, details={"search_id": search_id})
            return _to_search_summary(record)

    def get_search_config(self, search_id: str) -> dict[str, Any]:
        """Return the stored configuration for a search.

        Args:
            search_id: Identifier to look up.

        Returns:
            The configuration as plain data.

        Raises:
            RecordNotFoundError: If the search does not exist.
        """
        with self._database.session() as session:
            record = session.get(SearchRecord, search_id)
            if record is None:
                msg = f"no search found with id {search_id!r}"
                raise RecordNotFoundError(msg, details={"search_id": search_id})
            return dict(record.config_json)

    def get_search_environment(self, search_id: str) -> dict[str, Any]:
        """Return the environment snapshot captured when a search was created.

        Args:
            search_id: Identifier to look up.

        Returns:
            The environment snapshot.

        Raises:
            RecordNotFoundError: If the search does not exist.
        """
        with self._database.session() as session:
            record = session.get(SearchRecord, search_id)
            if record is None:
                msg = f"no search found with id {search_id!r}"
                raise RecordNotFoundError(msg, details={"search_id": search_id})
            return dict(record.environment_json)

    def list_searches(self, *, limit: int = 50) -> list[SearchSummary]:
        """Return recent searches, newest first.

        Args:
            limit: Maximum number of rows.

        Returns:
            The summaries.
        """
        with self._database.session() as session:
            records = session.scalars(
                select(SearchRecord).order_by(SearchRecord.created_at.desc()).limit(limit)
            ).all()
            return [_to_search_summary(record) for record in records]

    def find_latest_search(self, *, name: str | None = None) -> SearchSummary | None:
        """Return the most recently created search, optionally filtered by name.

        Args:
            name: Restrict to searches with this name.

        Returns:
            The summary, or ``None`` when nothing matches.
        """
        with self._database.session() as session:
            statement = select(SearchRecord).order_by(SearchRecord.created_at.desc()).limit(1)
            if name is not None:
                statement = (
                    select(SearchRecord)
                    .where(SearchRecord.name == name)
                    .order_by(SearchRecord.created_at.desc())
                    .limit(1)
                )
            record = session.scalars(statement).one_or_none()
            return _to_search_summary(record) if record else None

    def update_search_status(
        self,
        search_id: str,
        status: SearchStatus,
        *,
        started: bool = False,
        completed: bool = False,
        notes: str | None = None,
    ) -> None:
        """Update a search's lifecycle status and timestamps.

        Args:
            search_id: Search to update.
            status: New status.
            started: Whether to stamp ``started_at`` if not already set.
            completed: Whether to stamp ``completed_at``.
            notes: Replacement operator notes.

        Raises:
            RecordNotFoundError: If the search does not exist.
        """
        with self._database.session() as session:
            record = session.get(SearchRecord, search_id)
            if record is None:
                msg = f"no search found with id {search_id!r}"
                raise RecordNotFoundError(msg, details={"search_id": search_id})
            record.status = status.value
            if started and record.started_at is None:
                record.started_at = utc_now()
            if completed:
                record.completed_at = utc_now()
            if notes is not None:
                record.notes = notes

    # -------------------------------------------------------------- candidates ----
    def add_candidate(
        self,
        *,
        search_id: str,
        architecture_hash: str,
        spec: ArchitectureSpec,
        rung: int = 0,
        status: CandidateState = CandidateState.PROPOSED,
        parent_id: str | None = None,
        mutation: str | None = None,
        origin: str = "unknown",
        generation: int | None = None,
        metadata: dict[str, Any] | None = None,
        candidate_id: str | None = None,
    ) -> str:
        """Insert a candidate.

        The ``(search_id, architecture_hash, rung)`` uniqueness constraint is enforced by
        the database, not by a prior ``SELECT``. Checking first and inserting second is a
        race: two workers can both see "not present" and both insert. Letting the
        constraint fail and translating the error is the only correct approach under
        concurrency.

        Args:
            search_id: Owning search.
            architecture_hash: Canonical hash.
            spec: The architecture.
            rung: Fidelity rung.
            status: Initial lifecycle state.
            parent_id: Parent candidate for lineage.
            mutation: Mutation description.
            origin: How the candidate was produced.
            generation: Strategy generation index.
            metadata: Strategy-specific extra data.
            candidate_id: Explicit identifier; generated when omitted.

        Returns:
            The candidate identifier.

        Raises:
            DuplicateRecordError: If the identity already exists in this search.
        """
        from nas_engine.architectures.canonical import to_canonical_dict

        identifier = candidate_id or new_id()
        try:
            with self._database.session() as session:
                session.add(
                    CandidateRecord(
                        id=identifier,
                        search_id=search_id,
                        architecture_hash=architecture_hash,
                        rung=rung,
                        spec_json=to_canonical_dict(spec),
                        status=status.value,
                        parent_id=parent_id,
                        mutation=mutation,
                        origin=origin,
                        generation=generation,
                        metadata_json=dict(metadata or {}),
                    )
                )
        except PersistenceError as exc:
            cause = exc.__cause__
            if isinstance(cause, IntegrityError):
                msg = (
                    f"candidate {architecture_hash} at rung {rung} already exists in search "
                    f"{search_id}"
                )
                raise DuplicateRecordError(
                    msg,
                    details={
                        "search_id": search_id,
                        "architecture_hash": architecture_hash,
                        "rung": rung,
                    },
                ) from exc
            raise
        return identifier

    def get_candidate(self, candidate_id: str) -> CandidateSummary:
        """Return one candidate.

        Args:
            candidate_id: Identifier to look up.

        Returns:
            The summary.

        Raises:
            RecordNotFoundError: If the candidate does not exist.
        """
        with self._database.session() as session:
            record = session.get(CandidateRecord, candidate_id)
            if record is None:
                msg = f"no candidate found with id {candidate_id!r}"
                raise RecordNotFoundError(msg, details={"candidate_id": candidate_id})
            return _to_candidate_summary(record)

    def get_candidate_spec(self, candidate_id: str) -> ArchitectureSpec:
        """Return a candidate's architecture, revalidated on read.

        Stored JSON is treated as untrusted: a hand-edited or corrupted row produces a
        clear validation error instead of a partially constructed object.

        Args:
            candidate_id: Identifier to look up.

        Returns:
            The architecture.

        Raises:
            RecordNotFoundError: If the candidate does not exist.
            ArchitectureValidationError: If the stored JSON is not a valid architecture.
        """
        from nas_engine.architectures.canonical import from_canonical_dict

        with self._database.session() as session:
            record = session.get(CandidateRecord, candidate_id)
            if record is None:
                msg = f"no candidate found with id {candidate_id!r}"
                raise RecordNotFoundError(msg, details={"candidate_id": candidate_id})
            return from_canonical_dict(record.spec_json)

    def find_candidate(
        self, search_id: str, architecture_hash: str, *, rung: int = 0
    ) -> CandidateSummary | None:
        """Look up a candidate by its identity within a search.

        Args:
            search_id: Owning search.
            architecture_hash: Canonical hash.
            rung: Fidelity rung.

        Returns:
            The summary, or ``None`` when absent.
        """
        with self._database.session() as session:
            record = session.scalars(
                select(CandidateRecord).where(
                    CandidateRecord.search_id == search_id,
                    CandidateRecord.architecture_hash == architecture_hash,
                    CandidateRecord.rung == rung,
                )
            ).one_or_none()
            return _to_candidate_summary(record) if record else None

    def list_candidates(
        self,
        search_id: str,
        *,
        statuses: Sequence[CandidateState] | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_by_objective: bool = False,
    ) -> list[CandidateSummary]:
        """List a search's candidates.

        Args:
            search_id: Owning search.
            statuses: Restrict to these states.
            limit: Maximum rows.
            offset: Rows to skip.
            order_by_objective: Order by cached objective value, best first, instead of by
                creation time.

        Returns:
            The summaries.
        """
        with self._database.session() as session:
            statement = select(CandidateRecord).where(CandidateRecord.search_id == search_id)
            if statuses:
                statement = statement.where(
                    CandidateRecord.status.in_([state.value for state in statuses])
                )
            if order_by_objective:
                statement = statement.order_by(
                    CandidateRecord.objective_value.desc().nullslast(),
                    CandidateRecord.created_at.asc(),
                )
            else:
                statement = statement.order_by(CandidateRecord.created_at.asc())
            if offset:
                statement = statement.offset(offset)
            if limit is not None:
                statement = statement.limit(limit)
            return [_to_candidate_summary(record) for record in session.scalars(statement).all()]

    def count_candidates_by_status(self, search_id: str) -> dict[str, int]:
        """Return a count of candidates per lifecycle state.

        Args:
            search_id: Owning search.

        Returns:
            State name to count, including states with no rows.
        """
        with self._database.session() as session:
            rows = session.execute(
                select(CandidateRecord.status, func.count())
                .where(CandidateRecord.search_id == search_id)
                .group_by(CandidateRecord.status)
            ).all()
        counts = {state.value: 0 for state in CandidateState}
        for status, count in rows:
            counts[str(status)] = int(count)
        return counts

    def update_candidate_state(
        self,
        candidate_id: str,
        target: CandidateState,
        *,
        reason: str | None = None,
        error: dict[str, Any] | None = None,
        objective_value: float | None = None,
    ) -> CandidateState:
        """Transition a candidate to a new state, validating the edge.

        Args:
            candidate_id: Candidate to update.
            target: New state.
            reason: Human-readable explanation recorded as an event.
            error: Failure record stored on the candidate.
            objective_value: Cached fitness to store.

        Returns:
            The new state.

        Raises:
            RecordNotFoundError: If the candidate does not exist.
            InvalidStateTransitionError: If the transition is not permitted.
        """
        with self._database.session() as session:
            record = session.get(CandidateRecord, candidate_id)
            if record is None:
                msg = f"no candidate found with id {candidate_id!r}"
                raise RecordNotFoundError(msg, details={"candidate_id": candidate_id})
            source = CandidateState(record.status)
            validate_transition(source, target)
            record.status = target.value
            if error is not None:
                record.error_json = error
            if objective_value is not None:
                record.objective_value = objective_value
            if reason:
                session.add(
                    SearchEventRecord(
                        search_id=record.search_id,
                        event=f"candidate.{target.value}",
                        candidate_id=candidate_id,
                        payload_json={"from": source.value, "reason": reason},
                    )
                )
        return target

    def increment_retry(self, candidate_id: str) -> int:
        """Increase a candidate's retry counter and return the new value.

        Args:
            candidate_id: Candidate to update.

        Returns:
            The updated retry count.

        Raises:
            RecordNotFoundError: If the candidate does not exist.
        """
        with self._database.session() as session:
            record = session.get(CandidateRecord, candidate_id)
            if record is None:
                msg = f"no candidate found with id {candidate_id!r}"
                raise RecordNotFoundError(msg, details={"candidate_id": candidate_id})
            record.retry_count += 1
            return int(record.retry_count)

    def claim_next_queued(self, search_id: str, *, worker_id: str) -> CandidateSummary | None:
        """Atomically claim the oldest queued candidate for a worker.

        The select and the update happen inside one transaction. SQLite serialises writers,
        so exactly one worker can win the claim; the loser sees the row already in
        ``RUNNING`` and moves on. This is what prevents two workers from training the same
        architecture.

        Args:
            search_id: Owning search.
            worker_id: Worker making the claim, recorded as an event.

        Returns:
            The claimed candidate, or ``None`` when the queue is empty.
        """
        with self._database.session() as session:
            record = session.scalars(
                select(CandidateRecord)
                .where(
                    CandidateRecord.search_id == search_id,
                    CandidateRecord.status == CandidateState.QUEUED.value,
                )
                .order_by(CandidateRecord.created_at.asc())
                .limit(1)
                .with_for_update(nowait=False)
            ).one_or_none()
            if record is None:
                return None
            validate_transition(CandidateState.QUEUED, CandidateState.RUNNING)
            record.status = CandidateState.RUNNING.value
            session.add(
                SearchEventRecord(
                    search_id=search_id,
                    event="candidate.claimed",
                    candidate_id=record.id,
                    payload_json={"worker_id": worker_id},
                )
            )
            return _to_candidate_summary(record)

    # ------------------------------------------------------------------ trials ----
    def start_trial(
        self,
        *,
        candidate_id: str,
        attempt: int,
        budget: TrainingBudget,
        worker_id: str | None = None,
        device: str = "cpu",
        trial_id: str | None = None,
    ) -> str:
        """Record the start of an evaluation attempt.

        Args:
            candidate_id: Candidate being evaluated.
            attempt: Zero-based attempt number.
            budget: Budget the attempt runs at.
            worker_id: Worker running the attempt.
            device: Device string.
            trial_id: Explicit identifier; generated when omitted.

        Returns:
            The trial identifier.

        Raises:
            DuplicateRecordError: If this attempt already exists for the candidate.
        """
        identifier = trial_id or new_id()
        try:
            with self._database.session() as session:
                session.add(
                    TrialRecord(
                        id=identifier,
                        candidate_id=candidate_id,
                        attempt=attempt,
                        budget_json=budget.to_dict(),
                        status=TrialState.RUNNING.value,
                        worker_id=worker_id,
                        device=device,
                    )
                )
        except PersistenceError as exc:
            if isinstance(exc.__cause__, IntegrityError):
                msg = f"attempt {attempt} already exists for candidate {candidate_id}"
                raise DuplicateRecordError(
                    msg, details={"candidate_id": candidate_id, "attempt": attempt}
                ) from exc
            raise
        return identifier

    def complete_trial(
        self,
        trial_id: str,
        *,
        metrics: dict[str, float],
        duration_seconds: float,
        training: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        artifact_sizes: dict[str, int] | None = None,
    ) -> None:
        """Record a successful evaluation, its metrics, and its artifacts.

        All three writes happen in one transaction: a trial marked completed but missing
        its metrics would be indistinguishable from a trial that measured nothing.

        Args:
            trial_id: Trial to complete.
            metrics: Measured metrics.
            duration_seconds: Wall-clock duration.
            training: Serialised training outcome.
            artifacts: Artifact kind to relative path.
            artifact_sizes: Artifact kind to byte size.

        Raises:
            RecordNotFoundError: If the trial does not exist.
        """
        with self._database.session() as session:
            trial = session.get(TrialRecord, trial_id)
            if trial is None:
                msg = f"no trial found with id {trial_id!r}"
                raise RecordNotFoundError(msg, details={"trial_id": trial_id})
            trial.status = TrialState.COMPLETED.value
            trial.completed_at = utc_now()
            trial.duration_seconds = duration_seconds
            trial.training_json = training
            for name, value in metrics.items():
                session.add(MetricRecord(trial_id=trial_id, name=name, value=float(value)))
            for kind, path in (artifacts or {}).items():
                # A retried or resumed evaluation rewrites the same artifact file. The row
                # is updated rather than inserted, because a second insert would violate the
                # uniqueness constraint and abort the whole transaction — losing the metrics
                # alongside it.
                existing = session.scalars(
                    select(ArtifactRecord).where(
                        ArtifactRecord.candidate_id == trial.candidate_id,
                        ArtifactRecord.kind == kind,
                        ArtifactRecord.path == path,
                    )
                ).one_or_none()
                size = int((artifact_sizes or {}).get(kind, 0))
                if existing is not None:
                    existing.trial_id = trial_id
                    existing.size_bytes = size or existing.size_bytes
                    existing.created_at = utc_now()
                    continue
                session.add(
                    ArtifactRecord(
                        candidate_id=trial.candidate_id,
                        trial_id=trial_id,
                        kind=kind,
                        path=path,
                        size_bytes=size,
                    )
                )

    def fail_trial(
        self,
        trial_id: str,
        *,
        error: dict[str, Any],
        duration_seconds: float,
        timeout: bool = False,
    ) -> None:
        """Record a failed evaluation attempt.

        Args:
            trial_id: Trial to fail.
            error: Failure record.
            duration_seconds: Wall-clock duration before failing.
            timeout: Whether the failure was a wall-clock timeout.

        Raises:
            RecordNotFoundError: If the trial does not exist.
        """
        with self._database.session() as session:
            trial = session.get(TrialRecord, trial_id)
            if trial is None:
                msg = f"no trial found with id {trial_id!r}"
                raise RecordNotFoundError(msg, details={"trial_id": trial_id})
            trial.status = (TrialState.TIMEOUT if timeout else TrialState.FAILED).value
            trial.completed_at = utc_now()
            trial.duration_seconds = duration_seconds
            trial.error_json = error

    def list_trials(self, candidate_id: str) -> list[dict[str, Any]]:
        """Return every attempt for a candidate, oldest first.

        Args:
            candidate_id: Candidate to inspect.

        Returns:
            Attempt records as plain data.
        """
        with self._database.session() as session:
            trials = session.scalars(
                select(TrialRecord)
                .where(TrialRecord.candidate_id == candidate_id)
                .order_by(TrialRecord.attempt.asc())
            ).all()
            return [
                {
                    "id": trial.id,
                    "attempt": trial.attempt,
                    "status": trial.status,
                    "budget": dict(trial.budget_json),
                    "worker_id": trial.worker_id,
                    "device": trial.device,
                    "started_at": trial.started_at.isoformat(),
                    "completed_at": (
                        trial.completed_at.isoformat() if trial.completed_at else None
                    ),
                    "duration_seconds": trial.duration_seconds,
                    "error": trial.error_json,
                    "metrics": _metrics_of(trial),
                }
                for trial in trials
            ]

    # --------------------------------------------------------------- artifacts ----
    def record_artifact(
        self,
        *,
        candidate_id: str,
        kind: str,
        path: str,
        size_bytes: int = 0,
        trial_id: str | None = None,
    ) -> None:
        """Record an artifact produced for a candidate.

        Args:
            candidate_id: Owning candidate.
            kind: Artifact kind.
            path: Path relative to the artifact root.
            size_bytes: File size.
            trial_id: Producing trial, when applicable.
        """
        try:
            with self._database.session() as session:
                session.add(
                    ArtifactRecord(
                        candidate_id=candidate_id,
                        trial_id=trial_id,
                        kind=kind,
                        path=path,
                        size_bytes=size_bytes,
                    )
                )
        except PersistenceError as exc:
            if isinstance(exc.__cause__, IntegrityError):
                # The same artifact recorded twice is harmless; a resumed evaluation
                # legitimately rewrites the same file.
                _LOGGER.debug(
                    "repository.artifact_exists",
                    candidate_id=candidate_id,
                    kind=kind,
                    path=path,
                )
                return
            raise

    # ------------------------------------------------------------- checkpoints ----
    def save_checkpoint(self, search_id: str, payload: dict[str, Any]) -> int:
        """Append a strategy-state checkpoint.

        Checkpoints are append-only. Overwriting a single row would mean a crash during the
        write leaves no usable checkpoint at all; keeping the history means the previous one
        is always available.

        Args:
            search_id: Owning search.
            payload: Strategy state plus engine bookkeeping.

        Returns:
            The checkpoint sequence number.
        """
        with self._database.session() as session:
            highest = session.scalar(
                select(func.max(CheckpointRecord.sequence)).where(
                    CheckpointRecord.search_id == search_id
                )
            )
            sequence = int(highest or 0) + 1
            session.add(
                CheckpointRecord(
                    search_id=search_id,
                    sequence=sequence,
                    format_version=SEARCH_CHECKPOINT_VERSION,
                    payload_json=payload,
                )
            )
            return sequence

    def latest_checkpoint(self, search_id: str) -> dict[str, Any] | None:
        """Return the most recent checkpoint payload for a search.

        Args:
            search_id: Owning search.

        Returns:
            The payload, or ``None`` when no checkpoint exists.
        """
        with self._database.session() as session:
            record = session.scalars(
                select(CheckpointRecord)
                .where(CheckpointRecord.search_id == search_id)
                .order_by(CheckpointRecord.sequence.desc())
                .limit(1)
            ).one_or_none()
            return dict(record.payload_json) if record else None

    def count_checkpoints(self, search_id: str) -> int:
        """Return how many checkpoints exist for a search.

        Args:
            search_id: Owning search.

        Returns:
            The checkpoint count.
        """
        with self._database.session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(CheckpointRecord)
                    .where(CheckpointRecord.search_id == search_id)
                )
                or 0
            )

    def prune_checkpoints(self, search_id: str, *, keep: int = 5) -> int:
        """Delete all but the most recent ``keep`` checkpoints.

        Strategy state includes the population and every seen hash, so checkpoints grow
        with the search. Keeping the last handful preserves the ability to roll back a
        corrupt write without letting the table grow without bound.

        Args:
            search_id: Owning search.
            keep: Number of recent checkpoints to retain.

        Returns:
            The number of checkpoints deleted.

        Raises:
            ValueError: If ``keep`` is negative.
        """
        if keep < 0:
            msg = f"keep must be non-negative, received {keep}"
            raise ValueError(msg)
        with self._database.session() as session:
            records = session.scalars(
                select(CheckpointRecord)
                .where(CheckpointRecord.search_id == search_id)
                .order_by(CheckpointRecord.sequence.desc())
            ).all()
            doomed = records[keep:]
            for record in doomed:
                session.delete(record)
            return len(doomed)

    # ------------------------------------------------------------------ events ----
    def record_event(
        self,
        *,
        search_id: str,
        event: str,
        candidate_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an audit event.

        Args:
            search_id: Owning search.
            event: Event name.
            candidate_id: Related candidate, when applicable.
            payload: Structured context.
        """
        with self._database.session() as session:
            session.add(
                SearchEventRecord(
                    search_id=search_id,
                    event=event,
                    candidate_id=candidate_id,
                    payload_json=dict(payload or {}),
                )
            )

    def list_events(self, search_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """Return a search's audit events, oldest first.

        Args:
            search_id: Owning search.
            limit: Maximum rows.

        Returns:
            Events as plain data.
        """
        with self._database.session() as session:
            records = session.scalars(
                select(SearchEventRecord)
                .where(SearchEventRecord.search_id == search_id)
                .order_by(SearchEventRecord.created_at.asc())
                .limit(limit)
            ).all()
            return [
                {
                    "event": record.event,
                    "candidate_id": record.candidate_id,
                    "payload": dict(record.payload_json),
                    "created_at": record.created_at.isoformat(),
                }
                for record in records
            ]

    # ------------------------------------------------------------- aggregation ----
    def completed_metrics(self, search_id: str) -> list[tuple[str, str, dict[str, float]]]:
        """Return ``(candidate_id, architecture_hash, metrics)`` for completed candidates.

        This is the input to ranking and Pareto-front computation, which are always
        recomputed from persisted metrics rather than cached — a cached front silently goes
        stale the moment another candidate completes.

        Args:
            search_id: Owning search.

        Returns:
            One entry per completed candidate that produced metrics.
        """
        results: list[tuple[str, str, dict[str, float]]] = []
        with self._database.session() as session:
            candidates = session.scalars(
                select(CandidateRecord).where(
                    CandidateRecord.search_id == search_id,
                    CandidateRecord.status == CandidateState.COMPLETED.value,
                )
            ).all()
            for candidate in candidates:
                trial = _latest_completed_trial(candidate)
                if trial is None:
                    continue
                metrics = _metrics_of(trial)
                if metrics:
                    results.append((candidate.id, candidate.architecture_hash, metrics))
        return results

    def best_candidate(
        self, search_id: str, *, metric: str = "validation_accuracy", maximize: bool = True
    ) -> CandidateSummary | None:
        """Return the completed candidate with the best value of one metric.

        This is a *single-metric* query, deliberately simple and fast. Multi-objective
        selection goes through :func:`nas_engine.objectives.ranking.rank_candidates`, which
        needs the whole population and cannot be expressed as one SQL ``ORDER BY``.

        Args:
            search_id: Owning search.
            metric: Metric to rank by.
            maximize: Whether larger is better.

        Returns:
            The best candidate, or ``None`` when nothing has completed.
        """
        entries = self.completed_metrics(search_id)
        scored = [(cid, metrics[metric]) for cid, _, metrics in entries if metric in metrics]
        if not scored:
            return None
        # Ties break by candidate id so repeated calls give the same answer.
        best_id = (max if maximize else min)(scored, key=lambda item: (item[1], item[0]))[0]
        return self.get_candidate(best_id)

    def lineage_nodes(self, search_id: str) -> list[LineageNode]:
        """Return lineage nodes for every candidate in a search.

        Args:
            search_id: Owning search.

        Returns:
            Nodes suitable for :class:`~nas_engine.architectures.lineage.LineageGraph`.
        """
        with self._database.session() as session:
            candidates = session.scalars(
                select(CandidateRecord).where(CandidateRecord.search_id == search_id)
            ).all()
            return [
                LineageNode(
                    candidate_id=candidate.id,
                    architecture_hash=candidate.architecture_hash,
                    parent_id=candidate.parent_id,
                    mutation=candidate.mutation,
                    generation=candidate.generation,
                    objective_value=candidate.objective_value,
                )
                for candidate in candidates
            ]

    # ---------------------------------------------------------------- recovery ----
    def recover_interrupted(self, search_id: str, *, max_retries: int) -> RecoveryReport:
        """Repair state left behind by an interrupted run.

        Candidates in ``RUNNING`` had a process die under them. Each is returned to
        ``QUEUED`` if it still has retries left, or moved to ``FAILED`` if not. Their
        in-flight trials are marked ``INTERRUPTED`` so the history records what happened
        rather than leaving a trial that never ends.

        Args:
            search_id: Owning search.
            max_retries: Retry allowance per candidate.

        Returns:
            A :class:`RecoveryReport`.
        """
        requeued: list[str] = []
        abandoned: list[str] = []
        interrupted_trials = 0

        with self._database.session() as session:
            running = session.scalars(
                select(CandidateRecord).where(
                    CandidateRecord.search_id == search_id,
                    CandidateRecord.status == CandidateState.RUNNING.value,
                )
            ).all()
            for candidate in running:
                for trial in candidate.trials:
                    if trial.status == TrialState.RUNNING.value:
                        trial.status = TrialState.INTERRUPTED.value
                        trial.completed_at = utc_now()
                        trial.error_json = {
                            "code": "interrupted",
                            "message": "the evaluating process exited before reporting",
                        }
                        interrupted_trials += 1
                if candidate.retry_count < max_retries:
                    candidate.status = CandidateState.QUEUED.value
                    candidate.retry_count += 1
                    requeued.append(candidate.id)
                else:
                    candidate.status = CandidateState.FAILED.value
                    candidate.error_json = {
                        "code": "retry_exhausted_error",
                        "message": (
                            f"interrupted {candidate.retry_count} times, which exhausts the "
                            f"retry allowance of {max_retries}"
                        ),
                    }
                    abandoned.append(candidate.id)
                session.add(
                    SearchEventRecord(
                        search_id=search_id,
                        event="candidate.recovered",
                        candidate_id=candidate.id,
                        payload_json={"new_status": candidate.status},
                    )
                )

            completed = int(
                session.scalar(
                    select(func.count())
                    .select_from(CandidateRecord)
                    .where(
                        CandidateRecord.search_id == search_id,
                        CandidateRecord.status == CandidateState.COMPLETED.value,
                    )
                )
                or 0
            )

        report = RecoveryReport(
            interrupted_running=len(requeued) + len(abandoned),
            interrupted_trials=interrupted_trials,
            requeued=tuple(requeued),
            abandoned=tuple(abandoned),
            completed=completed,
        )
        if report.interrupted_running:
            _LOGGER.warning(
                "repository.recovered_interrupted",
                search_id=search_id,
                requeued=len(requeued),
                abandoned=len(abandoned),
                interrupted_trials=interrupted_trials,
            )
        return report

    def seen_hashes(self, search_id: str) -> set[str]:
        """Return every architecture hash already proposed in a search.

        Args:
            search_id: Owning search.

        Returns:
            The set of hashes.
        """
        with self._database.session() as session:
            rows = session.scalars(
                select(CandidateRecord.architecture_hash).where(
                    CandidateRecord.search_id == search_id
                )
            ).all()
            return set(rows)

    def delete_search(self, search_id: str) -> bool:
        """Delete a search and every record that belongs to it.

        Artifact *files* are not removed; only the records referencing them. Deleting user
        data from disk on a metadata operation would be surprising and irreversible.

        Args:
            search_id: Search to delete.

        Returns:
            ``True`` when a search was deleted.
        """
        with self._database.session() as session:
            record = session.get(SearchRecord, search_id)
            if record is None:
                return False
            session.delete(record)
            return True


__all__ = [
    "SEARCH_CHECKPOINT_VERSION",
    "CandidateSummary",
    "RecoveryReport",
    "SearchRepository",
    "SearchSummary",
]
