"""The closed vocabulary of structured search events.

Event names are a *public interface*: dashboards, log queries, and alerts are built on
them. They are therefore declared once, here, as an enumeration rather than being
typed as free-form strings at call sites. Adding an event means adding a member;
renaming one is a breaking change and requires a note in the changelog.

Each event is emitted with the ambient identifier context (search, candidate, trial,
worker) automatically attached by
:func:`nas_engine.observability.context.current_context`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from nas_engine.observability.logging import get_logger


class Event(str, Enum):
    """Canonical structured-event names.

    The value is the string written to logs. Inheriting from :class:`str` means the
    member can be used anywhere a string is expected without an explicit ``.value``.
    """

    SEARCH_STARTED = "search.started"
    SEARCH_RESUMED = "search.resumed"
    SEARCH_COMPLETED = "search.completed"
    SEARCH_FAILED = "search.failed"
    SEARCH_INTERRUPTED = "search.interrupted"

    CANDIDATE_PROPOSED = "candidate.proposed"
    CANDIDATE_REJECTED = "candidate.rejected"
    CANDIDATE_DUPLICATE = "candidate.duplicate"
    CANDIDATE_QUEUED = "candidate.queued"
    CANDIDATE_PROMOTED = "candidate.promoted"
    CANDIDATE_PRUNED = "candidate.pruned"
    CANDIDATE_CANCELLED = "candidate.cancelled"

    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    EVALUATION_TIMEOUT = "evaluation.timeout"

    POPULATION_UPDATED = "population.updated"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_RESTORED = "checkpoint.restored"
    RETRY_SCHEDULED = "retry.scheduled"
    RETRY_EXHAUSTED = "retry.exhausted"

    PARETO_UPDATED = "pareto.updated"
    REPORT_GENERATED = "report.generated"

    def __str__(self) -> str:
        """Return the wire value so f-strings render the event name, not the member."""
        return self.value


#: Events logged at WARNING rather than INFO, because each one signals lost work.
_WARNING_EVENTS = frozenset(
    {
        Event.CANDIDATE_REJECTED,
        Event.EVALUATION_FAILED,
        Event.EVALUATION_TIMEOUT,
        Event.RETRY_SCHEDULED,
        Event.SEARCH_INTERRUPTED,
    }
)

#: Events logged at ERROR: the search or a candidate is permanently damaged.
_ERROR_EVENTS = frozenset({Event.SEARCH_FAILED, Event.RETRY_EXHAUSTED})


def emit(event: Event, **fields: Any) -> None:
    """Emit a structured event at the level appropriate for its severity.

    Args:
        event: The event to emit.
        **fields: Additional structured context. Values must be JSON-serialisable if
            the JSON renderer is in use.
    """
    # Resolved per call rather than bound at import time, so that a reconfiguration after
    # this module was imported (the engine choosing JSON output, a worker reconfiguring
    # after spawn) actually takes effect.
    logger = get_logger("nas_engine.events")
    if event in _ERROR_EVENTS:
        logger.error(str(event), **fields)
    elif event in _WARNING_EVENTS:
        logger.warning(str(event), **fields)
    else:
        logger.info(str(event), **fields)


__all__ = ["Event", "emit"]
