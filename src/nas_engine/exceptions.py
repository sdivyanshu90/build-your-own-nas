"""Error taxonomy for ``nas_engine``.

Every error raised deliberately by this package derives from :class:`NasEngineError`.
The taxonomy exists so that callers can react to *categories* of problem without
string-matching messages, and so that the orchestration engine can classify a
failure as **retriable** or **permanent** without knowing which component raised it.

Design rules
------------
* Every exception carries a machine-readable ``code`` (stable across releases) and
  a human-readable message that states *what was received*, *what was expected*, and
  *how to fix it* whenever that information is available.
* ``details`` holds structured context (field names, offending values, limits). It is
  emitted into structured logs and persisted with failed candidates.
* ``retriable`` is a class-level property. Transient infrastructure problems (a locked
  database, a timeout) are retriable; a structurally invalid architecture never is.

See ``docs/architecture/component-design.md`` for how the taxonomy maps onto the
candidate state machine.
"""

from __future__ import annotations

from typing import Any


class NasEngineError(Exception):
    """Base class for all deliberate ``nas_engine`` failures.

    Args:
        message: Human-readable description of the failure.
        details: Structured context describing the failure (field names, values,
            limits). Values must be JSON-serialisable so the failure can be persisted.
    """

    #: Stable machine-readable identifier for this error category.
    code: str = "nas_engine_error"
    #: Whether the orchestration engine may retry the operation that raised this.
    retriable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation used for logging and persistence."""
        return {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.message,
            "retriable": self.retriable,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Render the message followed by sorted structured details, when present."""
        if not self.details:
            return self.message
        rendered = ", ".join(f"{key}={self.details[key]!r}" for key in sorted(self.details))
        return f"{self.message} ({rendered})"


# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------
class ConfigurationError(NasEngineError):
    """Raised when configuration is missing, malformed, or semantically invalid."""

    code = "configuration_error"


class ConfigVersionError(ConfigurationError):
    """Raised when a persisted configuration was written by an incompatible version."""

    code = "config_version_error"


# ---------------------------------------------------------------------------------
# Search space and architectures
# ---------------------------------------------------------------------------------
class SearchSpaceError(NasEngineError):
    """Raised when a search space definition is itself invalid or unusable."""

    code = "search_space_error"


class ArchitectureValidationError(NasEngineError):
    """Raised when an architecture violates schema, semantic, or constraint rules.

    This is a *permanent* failure: re-running the same specification cannot succeed.
    """

    code = "architecture_validation_error"


class ShapeInferenceError(ArchitectureValidationError):
    """Raised when tensor shapes cannot be reconciled for an architecture."""

    code = "shape_inference_error"


class ConstraintViolationError(ArchitectureValidationError):
    """Raised when a candidate breaks a hard constraint such as a parameter budget."""

    code = "constraint_violation_error"


class MutationError(NasEngineError):
    """Raised when no valid mutation can be produced from a parent architecture."""

    code = "mutation_error"


# ---------------------------------------------------------------------------------
# Model construction, training, evaluation
# ---------------------------------------------------------------------------------
class ModelBuildError(NasEngineError):
    """Raised when a validated architecture still fails to materialise as a module."""

    code = "model_build_error"


class TrainingError(NasEngineError):
    """Raised when training fails in a way that may be transient (e.g. OOM)."""

    code = "training_error"
    retriable = True


class NonFiniteLossError(TrainingError):
    """Raised when the loss becomes NaN or infinite.

    Divergence is treated as permanent: the same seed and architecture will diverge
    again, so retrying only wastes budget.
    """

    code = "non_finite_loss_error"
    retriable = False


class EvaluationTimeoutError(NasEngineError):
    """Raised when a candidate evaluation exceeds its wall-clock allowance."""

    code = "evaluation_timeout_error"
    retriable = True


class ResourceLimitError(NasEngineError):
    """Raised when a candidate would exceed a configured resource limit."""

    code = "resource_limit_error"


# ---------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------
class PersistenceError(NasEngineError):
    """Raised when the storage layer cannot complete an operation."""

    code = "persistence_error"
    retriable = True


class SchemaVersionError(PersistenceError):
    """Raised when the database schema version is newer than this package supports."""

    code = "schema_version_error"
    retriable = False


class RecordNotFoundError(PersistenceError):
    """Raised when a requested record does not exist."""

    code = "record_not_found_error"
    retriable = False


class DuplicateRecordError(PersistenceError):
    """Raised when a uniqueness constraint would be violated."""

    code = "duplicate_record_error"
    retriable = False


# ---------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------
class OrchestrationError(NasEngineError):
    """Raised for engine-level problems that are not attributable to one candidate."""

    code = "orchestration_error"


class InvalidStateTransitionError(OrchestrationError):
    """Raised when a candidate is moved between states along a forbidden edge."""

    code = "invalid_state_transition_error"


class CheckpointError(OrchestrationError):
    """Raised when a checkpoint cannot be written, read, or validated."""

    code = "checkpoint_error"


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint was produced by an incompatible checkpoint format."""

    code = "checkpoint_version_error"


class SearchExhaustedError(OrchestrationError):
    """Raised when a strategy can no longer propose novel valid candidates."""

    code = "search_exhausted_error"


class RetryExhaustedError(OrchestrationError):
    """Raised when a candidate has consumed its retry allowance."""

    code = "retry_exhausted_error"


class WorkerError(OrchestrationError):
    """Raised when a worker process dies or returns an unusable result."""

    code = "worker_error"
    retriable = True


# ---------------------------------------------------------------------------------
# Datasets, objectives, reporting, security
# ---------------------------------------------------------------------------------
class DatasetError(NasEngineError):
    """Raised when a dataset cannot be constructed or is unavailable offline."""

    code = "dataset_error"


class ObjectiveError(NasEngineError):
    """Raised when objectives are misconfigured or required metrics are missing."""

    code = "objective_error"


class ReportingError(NasEngineError):
    """Raised when a report or export cannot be produced."""

    code = "reporting_error"


class UnsafePathError(NasEngineError):
    """Raised when a path escapes its permitted root or is otherwise unsafe.

    See ``docs/architecture/security.md`` for the trust boundary this enforces.
    """

    code = "unsafe_path_error"


__all__ = [
    "ArchitectureValidationError",
    "CheckpointError",
    "CheckpointVersionError",
    "ConfigVersionError",
    "ConfigurationError",
    "ConstraintViolationError",
    "DatasetError",
    "DuplicateRecordError",
    "EvaluationTimeoutError",
    "InvalidStateTransitionError",
    "ModelBuildError",
    "MutationError",
    "NasEngineError",
    "NonFiniteLossError",
    "ObjectiveError",
    "OrchestrationError",
    "PersistenceError",
    "RecordNotFoundError",
    "ReportingError",
    "ResourceLimitError",
    "RetryExhaustedError",
    "SchemaVersionError",
    "SearchExhaustedError",
    "SearchSpaceError",
    "ShapeInferenceError",
    "TrainingError",
    "UnsafePathError",
    "WorkerError",
]
