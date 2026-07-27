"""The search-strategy contract.

This interface is the project's central extension point. The orchestration engine drives
the candidate lifecycle — validation, deduplication, persistence, evaluation, retries,
checkpointing — and knows nothing about *how* candidates are chosen. A strategy chooses
candidates and knows nothing about how they are evaluated or stored.

This is dependency inversion in the literal sense: both the engine and each strategy
depend on this abstraction, and neither depends on the other. Adding Bayesian optimisation
or an RL controller requires implementing this interface and registering it; not one line
of :mod:`nas_engine.orchestration.engine` changes. See
``docs/guides/adding-a-search-strategy.md``.

The contract in full
--------------------
``propose(count)``
    Return up to ``count`` proposals. Returning **fewer** is legal and meaningful: it says
    "I cannot usefully propose more right now". Returning an **empty** list while
    evaluations are outstanding means "I am waiting for results" — the engine will drain
    its in-flight work and ask again. Returning empty with nothing outstanding and
    ``is_finished()`` false means the strategy is stuck, and the engine raises rather than
    spinning.

``observe(observation)``
    Receive exactly one completed evaluation, successful or failed. Strategies must handle
    failures: a candidate that crashed still consumed budget and must not be proposed
    again.

``is_finished()``
    Report whether the strategy has nothing left to do. The engine also enforces its own
    budget; whichever triggers first wins.

``state_dict()`` / ``load_state_dict(payload)``
    Serialise and restore *everything* that affects future proposals, including the
    generator state. A strategy that re-seeds on resume replays proposals it already made.

``statistics()``
    Strategy-specific counters for reports and logs. Free-form, but keys should be stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from nas_engine.architectures.canonical import from_canonical_dict, to_canonical_dict
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.result import EvaluationResult

#: Version of the strategy state payload envelope. Individual strategies add their own
#: version inside the payload; this one covers the shared wrapper.
STRATEGY_STATE_VERSION: int = 1


@dataclass(frozen=True)
class Proposal:
    """One candidate a strategy wants evaluated.

    Attributes:
        spec: The architecture to evaluate.
        budget: Resources the strategy wants spent on it.
        parent_id: Candidate id of the parent, for lineage. ``None`` for founders.
        mutation: Description of the mutation that produced this candidate.
        origin: How the proposal was produced, e.g. ``"random"``, ``"mutation"``,
            ``"promotion"``. Used in reports to show where good candidates came from.
        metadata: Strategy-specific extra data persisted with the candidate.
    """

    spec: ArchitectureSpec
    budget: TrainingBudget
    parent_id: str | None = None
    mutation: str | None = None
    origin: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """One completed evaluation handed back to the strategy.

    Attributes:
        candidate_id: Identifier the engine assigned to the candidate.
        architecture_hash: Canonical architecture hash.
        spec: The architecture that was evaluated.
        result: The full evaluation result, successful or failed.
        objective_value: Time-stable scalar fitness, larger is better, or ``None`` when
            the candidate could not be scored. Computed by the engine via
            :func:`nas_engine.objectives.online.online_objective_value`.
        parent_id: Candidate id of the parent, if any.
    """

    candidate_id: str
    architecture_hash: str
    spec: ArchitectureSpec
    result: EvaluationResult
    objective_value: float | None
    parent_id: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the evaluation completed successfully."""
        return self.result.succeeded


@dataclass(frozen=True)
class StrategyStatistics:
    """Counters every strategy reports.

    Attributes:
        proposed: Proposals returned.
        observed: Observations received.
        succeeded: Observations that were successful evaluations.
        failed: Observations that were failures.
        duplicates_avoided: Proposals discarded internally for being duplicates.
        extra: Strategy-specific counters.
    """

    proposed: int = 0
    observed: int = 0
    succeeded: int = 0
    failed: int = 0
    duplicates_avoided: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "proposed": self.proposed,
            "observed": self.observed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "duplicates_avoided": self.duplicates_avoided,
            **self.extra,
        }


class SearchStrategy(ABC):
    """Abstract base class every search strategy implements.

    Class attributes:
        name: Registry key and the value persisted with search runs.
        requires_synchronous_observations: When ``True``, the engine limits itself to one
            outstanding evaluation at a time. Strategies with a barrier — successive
            halving cannot promote until an entire rung has reported — set this. Purely
            sequential strategies leave it ``False`` so they can use worker parallelism.
    """

    name: ClassVar[str] = "abstract"
    requires_synchronous_observations: ClassVar[bool] = False

    @abstractmethod
    def propose(self, count: int) -> list[Proposal]:
        """Return up to ``count`` candidate proposals.

        Args:
            count: Maximum number of proposals wanted.

        Returns:
            Between zero and ``count`` proposals.

        Raises:
            SearchExhaustedError: If the strategy can never propose again. Prefer
                returning an empty list and reporting ``is_finished()`` where that is the
                accurate description.
        """

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        """Record one completed evaluation.

        Args:
            observation: The evaluation outcome.
        """

    @abstractmethod
    def is_finished(self) -> bool:
        """Report whether the strategy has completed its plan.

        Returns:
            ``True`` when no further proposals will ever be produced.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all state affecting future proposals.

        Returns:
            The checkpoint payload.
        """

    @abstractmethod
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore state captured by :meth:`state_dict`.

        Args:
            payload: Previously captured state.

        Raises:
            CheckpointVersionError: If the payload version is unsupported.
            CheckpointError: If the payload is malformed.
        """

    @abstractmethod
    def statistics(self) -> StrategyStatistics:
        """Return current counters for reporting.

        Returns:
            The statistics.
        """

    # -- optional hooks -------------------------------------------------------------
    def on_duplicate(  # noqa: B027 - optional hook; the default is deliberately a no-op
        self, architecture_hash: str
    ) -> None:
        """Called when the engine rejects a proposal as a duplicate.

        The default implementation does nothing. Strategies that maintain their own
        novelty bookkeeping override it to stay in sync with the engine's view.

        Args:
            architecture_hash: Hash of the duplicate architecture.
        """

    def on_rejected(  # noqa: B027 - optional hook; the default is deliberately a no-op
        self, spec: ArchitectureSpec, reason: str
    ) -> None:
        """Called when the engine rejects a proposal as invalid or infeasible.

        Args:
            spec: The rejected architecture.
            reason: Human-readable rejection reason.
        """

    def describe(self) -> str:
        """Return a human-readable description of the strategy's configuration."""
        return f"{self.name} ({type(self).__name__})"


def serialize_spec(spec: ArchitectureSpec) -> dict[str, Any]:
    """Render an architecture for inclusion in a strategy state payload.

    Args:
        spec: Architecture to serialise.

    Returns:
        Plain JSON-compatible data.
    """
    return to_canonical_dict(spec)


def deserialize_spec(payload: dict[str, Any]) -> ArchitectureSpec:
    """Rebuild an architecture from a strategy state payload.

    The payload is validated as untrusted input, so a corrupt or tampered checkpoint
    produces a clear validation error rather than a half-built object.

    Args:
        payload: Serialised architecture.

    Returns:
        The validated architecture.

    Raises:
        ArchitectureValidationError: If the payload is not a valid architecture.
    """
    return from_canonical_dict(payload)


__all__ = [
    "STRATEGY_STATE_VERSION",
    "Observation",
    "Proposal",
    "SearchStrategy",
    "StrategyStatistics",
    "deserialize_spec",
    "serialize_spec",
]
