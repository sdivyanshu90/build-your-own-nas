"""Evaluation results and the failure taxonomy that classifies what went wrong.

Every evaluation produces exactly one :class:`EvaluationResult`, successful or not. There
is no "returns ``None`` on failure" path, because a failed evaluation still carries
information the search needs: which architecture failed, at which budget, how long it ran
before failing, and — crucially — whether retrying could plausibly help.

Retriable versus permanent
--------------------------
The distinction drives the retry policy in
:mod:`nas_engine.orchestration.retry`:

============================  ===========  ================================================
Failure                       Retriable?   Why
============================  ===========  ================================================
Invalid architecture          No           Deterministic; the same input fails identically
Divergent loss (NaN)          No           Same seed, same architecture, same divergence
Model build error             No           Structural
Constraint violation          No           The architecture is simply too expensive
Out of memory                 Yes          Depends on what else was resident
Timeout                       Yes          Depends on machine load
Worker crash                  Yes          Infrastructure, not the candidate
Database write failure        Yes          Transient lock contention
============================  ===========  ================================================

Getting this wrong is expensive in both directions: retrying a permanent failure burns the
budget three times over, and giving up on a transient one silently loses a good candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import (
    ArchitectureValidationError,
    ConstraintViolationError,
    EvaluationTimeoutError,
    ModelBuildError,
    NasEngineError,
    NonFiniteLossError,
    PersistenceError,
    ResourceLimitError,
    TrainingError,
    WorkerError,
)
from nas_engine.utilities.timing import utc_now


class FailureKind(str, Enum):
    """Coarse classification of an evaluation failure.

    Members:
        VALIDATION: The architecture was invalid.
        CONSTRAINT: The architecture exceeded a resource constraint.
        BUILD: The model could not be constructed.
        TRAINING: Training failed for a recoverable reason.
        DIVERGENCE: The loss became non-finite.
        TIMEOUT: The evaluation exceeded its wall-clock budget.
        RESOURCE: A memory or resource limit was hit.
        PERSISTENCE: The result could not be stored.
        WORKER: The worker process died or misbehaved.
        UNKNOWN: Anything not otherwise classified.
    """

    VALIDATION = "validation"
    CONSTRAINT = "constraint"
    BUILD = "build"
    TRAINING = "training"
    DIVERGENCE = "divergence"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    PERSISTENCE = "persistence"
    WORKER = "worker"
    UNKNOWN = "unknown"


#: Exception type to failure kind. Ordered most-specific first, because several of these
#: are subclasses of one another (``NonFiniteLossError`` is a ``TrainingError``).
_FAILURE_MAPPING: tuple[tuple[type[BaseException], FailureKind], ...] = (
    (NonFiniteLossError, FailureKind.DIVERGENCE),
    (ConstraintViolationError, FailureKind.CONSTRAINT),
    (ArchitectureValidationError, FailureKind.VALIDATION),
    (ModelBuildError, FailureKind.BUILD),
    (EvaluationTimeoutError, FailureKind.TIMEOUT),
    (ResourceLimitError, FailureKind.RESOURCE),
    (PersistenceError, FailureKind.PERSISTENCE),
    (WorkerError, FailureKind.WORKER),
    (TrainingError, FailureKind.TRAINING),
    (MemoryError, FailureKind.RESOURCE),
)


def classify_failure(error: BaseException) -> tuple[FailureKind, bool]:
    """Classify an exception into a failure kind and a retry decision.

    Args:
        error: The exception raised during evaluation.

    Returns:
        A ``(kind, retriable)`` tuple.
    """
    for error_type, kind in _FAILURE_MAPPING:
        if isinstance(error, error_type):
            if isinstance(error, NasEngineError):
                return kind, error.retriable
            # MemoryError and similar built-ins: a resource shortage is worth one retry
            # once other work has finished and freed memory.
            return kind, True
    if isinstance(error, NasEngineError):
        return FailureKind.UNKNOWN, error.retriable
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return FailureKind.RESOURCE, True
    # Unknown exceptions are treated as permanent. Retrying something we do not understand
    # risks an infinite loop of identical crashes.
    return FailureKind.UNKNOWN, False


@dataclass(frozen=True)
class EvaluationFailure:
    """A structured description of why an evaluation failed.

    Attributes:
        kind: Coarse classification.
        code: Stable machine-readable error code from the exception taxonomy.
        message: Human-readable description.
        retriable: Whether the orchestrator may retry.
        exception_type: Name of the exception class.
        details: Structured context from the exception.
        traceback_text: Formatted traceback, when captured.
    """

    kind: FailureKind
    code: str
    message: str
    retriable: bool
    exception_type: str
    details: dict[str, Any] = field(default_factory=dict)
    traceback_text: str | None = None

    @classmethod
    def from_exception(
        cls, error: BaseException, *, traceback_text: str | None = None
    ) -> EvaluationFailure:
        """Build a failure record from an exception.

        Args:
            error: The exception.
            traceback_text: Optional formatted traceback.

        Returns:
            The failure record.
        """
        kind, retriable = classify_failure(error)
        code = error.code if isinstance(error, NasEngineError) else type(error).__name__.lower()
        details = dict(error.details) if isinstance(error, NasEngineError) else {}
        return cls(
            kind=kind,
            code=code,
            message=str(error),
            retriable=retriable,
            exception_type=type(error).__name__,
            details=details,
            traceback_text=traceback_text,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "kind": self.kind.value,
            "code": self.code,
            "message": self.message,
            "retriable": self.retriable,
            "exception_type": self.exception_type,
            "details": dict(self.details),
            "traceback": self.traceback_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationFailure:
        """Rebuild a failure record from :meth:`to_dict` output.

        Args:
            payload: Serialised failure.

        Returns:
            The reconstructed record.
        """
        return cls(
            kind=FailureKind(payload.get("kind", FailureKind.UNKNOWN.value)),
            code=str(payload.get("code", "unknown")),
            message=str(payload.get("message", "")),
            retriable=bool(payload.get("retriable", False)),
            exception_type=str(payload.get("exception_type", "Exception")),
            details=dict(payload.get("details", {})),
            traceback_text=payload.get("traceback"),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """The complete outcome of evaluating one candidate at one budget.

    Attributes:
        candidate_id: Identifier of the candidate.
        architecture_hash: Canonical architecture hash.
        budget: The budget the evaluation ran at.
        metrics: Measured metrics, keyed by the names objectives refer to.
        succeeded: Whether the evaluation completed.
        failure: Failure record when ``succeeded`` is ``False``.
        artifacts: Artifact names mapped to paths, relative to the artifact root.
        artifact_bytes: Artifact names mapped to file sizes in bytes.
        started_at: UTC start time.
        completed_at: UTC completion time.
        duration_seconds: Wall-clock duration.
        device: Device string the evaluation ran on.
        worker_id: Worker identifier, when run under multiprocessing.
        training: Serialised training outcome, when training ran.
        notes: Free-text warnings that are not failures, e.g. latency caveats.
    """

    candidate_id: str
    architecture_hash: str
    budget: TrainingBudget
    metrics: dict[str, float] = field(default_factory=dict)
    succeeded: bool = True
    failure: EvaluationFailure | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_bytes: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)
    duration_seconds: float = 0.0
    device: str = "cpu"
    worker_id: str | None = None
    training: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def primary_metric(self) -> float | None:
        """Validation accuracy, the conventional primary objective."""
        return self.metrics.get("validation_accuracy")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "candidate_id": self.candidate_id,
            "architecture_hash": self.architecture_hash,
            "budget": self.budget.to_dict(),
            "metrics": dict(self.metrics),
            "succeeded": self.succeeded,
            "failure": self.failure.to_dict() if self.failure else None,
            "artifacts": dict(self.artifacts),
            "artifact_bytes": dict(self.artifact_bytes),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "device": self.device,
            "worker_id": self.worker_id,
            "training": dict(self.training),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            payload: Serialised result.

        Returns:
            The reconstructed result.
        """
        failure = payload.get("failure")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            architecture_hash=str(payload["architecture_hash"]),
            budget=TrainingBudget.from_dict(payload["budget"]),
            metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
            succeeded=bool(payload.get("succeeded", True)),
            failure=EvaluationFailure.from_dict(failure) if failure else None,
            artifacts=dict(payload.get("artifacts", {})),
            artifact_bytes={
                key: int(value) for key, value in payload.get("artifact_bytes", {}).items()
            },
            started_at=datetime.fromisoformat(payload["started_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            device=str(payload.get("device", "cpu")),
            worker_id=payload.get("worker_id"),
            training=dict(payload.get("training", {})),
            notes=tuple(payload.get("notes", ())),
        )


__all__ = [
    "EvaluationFailure",
    "EvaluationResult",
    "FailureKind",
    "classify_failure",
]
