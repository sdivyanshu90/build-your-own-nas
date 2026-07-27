"""The candidate state machine.

Every candidate moves through an explicit sequence of states, and every transition is
checked. This is not ceremony: the recovery story depends on it.

When a search is interrupted, the database is the only record of what happened. A
candidate sitting in ``RUNNING`` means "a process was evaluating this when it died" —
which is exactly the case that needs retrying. A candidate in ``QUEUED`` was never
started, so it can simply be re-proposed. A candidate in ``FAILED`` already exhausted its
retries. Without distinct states these cases are indistinguishable, and resume either
loses work or repeats it.

.. code-block:: text

    PROPOSED ──validate──> VALIDATED ──enqueue──> QUEUED ──start──> RUNNING
        │                      │                     │                 │
        │                      │                     │                 ├─success──> COMPLETED
        │                      │                     │                 ├─retriable─> QUEUED
        │                      │                     │                 └─permanent─> FAILED
        │                      └──constraint────> PRUNED
        └──invalid─────────────────────────────> FAILED

    Any non-terminal state ──cancel──> CANCELLED

Terminal states are ``COMPLETED``, ``FAILED``, ``PRUNED``, and ``CANCELLED``. Nothing
leaves them; a retry creates a *new trial* on a candidate that returns to ``QUEUED``
before it reaches ``FAILED``.

This module imports nothing from the rest of the package except the exception taxonomy, so
both :mod:`nas_engine.persistence` and :mod:`nas_engine.orchestration.engine` can depend on
it without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nas_engine.exceptions import InvalidStateTransitionError


class CandidateState(str, Enum):
    """Lifecycle state of a single candidate.

    Members:
        PROPOSED: The strategy suggested it; nothing has been checked yet.
        VALIDATED: It passed schema, semantic, and membership validation.
        QUEUED: It is waiting for an evaluation slot.
        RUNNING: A worker is evaluating it now.
        COMPLETED: Evaluation finished and metrics were recorded.
        FAILED: Evaluation failed permanently, or retries were exhausted.
        PRUNED: Structurally valid but rejected by a resource constraint, or eliminated by
            a multi-fidelity rung. Distinct from ``FAILED`` because nothing went wrong.
        CANCELLED: The operator stopped the search before this candidate finished.
    """

    PROPOSED = "proposed"
    VALIDATED = "validated"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is possible from this state."""
        return self in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        """Whether the candidate still occupies search resources."""
        return self in {CandidateState.QUEUED, CandidateState.RUNNING}


class TrialState(str, Enum):
    """Outcome state of a single evaluation attempt.

    Members:
        RUNNING: The attempt is in progress.
        COMPLETED: The attempt succeeded.
        FAILED: The attempt raised.
        TIMEOUT: The attempt exceeded its wall-clock budget.
        INTERRUPTED: The process died before the attempt reported. Recorded during
            recovery, not by the attempt itself.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


#: States from which nothing further happens.
TERMINAL_STATES: frozenset[CandidateState] = frozenset(
    {
        CandidateState.COMPLETED,
        CandidateState.FAILED,
        CandidateState.PRUNED,
        CandidateState.CANCELLED,
    }
)

#: The complete transition table. Every legal edge is listed here and nowhere else, so the
#: diagram above and the code cannot drift apart.
ALLOWED_TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    CandidateState.PROPOSED: frozenset(
        {
            CandidateState.VALIDATED,
            CandidateState.FAILED,
            CandidateState.PRUNED,
            CandidateState.CANCELLED,
        }
    ),
    CandidateState.VALIDATED: frozenset(
        {CandidateState.QUEUED, CandidateState.PRUNED, CandidateState.CANCELLED}
    ),
    CandidateState.QUEUED: frozenset(
        {
            CandidateState.RUNNING,
            CandidateState.PRUNED,
            CandidateState.CANCELLED,
            CandidateState.FAILED,
        }
    ),
    CandidateState.RUNNING: frozenset(
        {
            CandidateState.COMPLETED,
            CandidateState.FAILED,
            CandidateState.QUEUED,  # a retriable failure returns the candidate to the queue
            CandidateState.PRUNED,
            CandidateState.CANCELLED,
        }
    ),
    CandidateState.COMPLETED: frozenset(),
    CandidateState.FAILED: frozenset(),
    CandidateState.PRUNED: frozenset(),
    CandidateState.CANCELLED: frozenset(),
}


def can_transition(source: CandidateState, target: CandidateState) -> bool:
    """Report whether a transition is legal.

    Args:
        source: Current state.
        target: Proposed next state.

    Returns:
        ``True`` when the edge exists in :data:`ALLOWED_TRANSITIONS`.
    """
    return target in ALLOWED_TRANSITIONS.get(source, frozenset())


def validate_transition(source: CandidateState, target: CandidateState) -> None:
    """Raise unless a transition is legal.

    Args:
        source: Current state.
        target: Proposed next state.

    Raises:
        InvalidStateTransitionError: If the transition is not permitted. The message lists
            the legal targets, so a caller with a bug learns what it should have done.
    """
    if can_transition(source, target):
        return
    allowed = sorted(state.value for state in ALLOWED_TRANSITIONS.get(source, frozenset()))
    if source.is_terminal:
        detail = f"'{source.value}' is a terminal state and admits no transitions"
    else:
        detail = f"legal transitions from '{source.value}' are {allowed}"
    msg = f"cannot move a candidate from '{source.value}' to '{target.value}': {detail}"
    raise InvalidStateTransitionError(
        msg,
        details={"source": source.value, "target": target.value, "allowed": allowed},
    )


@dataclass
class CandidateStateMachine:
    """Tracks one candidate's state and its transition history.

    The history is kept because "how did this candidate reach ``FAILED``?" is a question
    that comes up constantly when debugging a search, and reconstructing it from log lines
    is tedious.

    Attributes:
        state: Current state.
        history: Ordered ``(state, reason)`` pairs, starting with the initial state.

    Example:
        >>> machine = CandidateStateMachine()
        >>> machine.transition(CandidateState.VALIDATED)
        <CandidateState.VALIDATED: 'validated'>
        >>> machine.transition(CandidateState.COMPLETED)
        Traceback (most recent call last):
            ...
        nas_engine.exceptions.InvalidStateTransitionError: cannot move a candidate from ...
    """

    state: CandidateState = CandidateState.PROPOSED
    history: list[tuple[CandidateState, str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Seed the history with the initial state."""
        if not self.history:
            self.history.append((self.state, "initial"))

    def transition(self, target: CandidateState, *, reason: str | None = None) -> CandidateState:
        """Move to ``target`` after validating the edge.

        Args:
            target: Next state.
            reason: Human-readable explanation recorded in the history.

        Returns:
            The new state.

        Raises:
            InvalidStateTransitionError: If the transition is not permitted.
        """
        validate_transition(self.state, target)
        self.state = target
        self.history.append((target, reason))
        return target

    def try_transition(self, target: CandidateState, *, reason: str | None = None) -> bool:
        """Move to ``target`` if legal, reporting success instead of raising.

        Used by recovery paths that sweep many candidates and must not abort the sweep
        because one row is already terminal.

        Args:
            target: Next state.
            reason: Human-readable explanation.

        Returns:
            ``True`` when the transition happened.
        """
        if not can_transition(self.state, target):
            return False
        self.transition(target, reason=reason)
        return True

    @property
    def is_terminal(self) -> bool:
        """Whether the candidate has reached a terminal state."""
        return self.state.is_terminal

    def describe_history(self) -> str:
        """Return the transition history as a readable arrow chain."""
        return " -> ".join(
            f"{state.value}" + (f"({reason})" if reason and reason != "initial" else "")
            for state, reason in self.history
        )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "CandidateState",
    "CandidateStateMachine",
    "TrialState",
    "can_transition",
    "validate_transition",
]
