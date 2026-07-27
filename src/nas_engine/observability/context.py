"""Ambient logging context carried by :mod:`contextvars`.

Every interesting log line should identify the search, candidate, trial, and worker it
belongs to. Threading those four identifiers through every function signature would be
invasive, so they live in context variables instead.

:mod:`contextvars` (rather than thread-locals) is used because the values are
correctly isolated per task in asynchronous code and per thread in threaded code, and
because entering a context returns a token that restores the exact previous value —
making nesting safe.

Worker processes start with an empty context and set their own; nothing is inherited
implicitly across a process boundary, which is deliberate.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# A ContextVar default must be immutable: a mutable default is shared by every context
# that never sets the variable, so one component could mutate another's view. `None` means
# "nothing bound", and `current_context` materialises a fresh dictionary each time.
_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("nas_engine_context", default=None)


def current_context() -> dict[str, Any]:
    """Return a copy of the ambient context dictionary."""
    return dict(_CONTEXT.get() or {})


@contextmanager
def bind_context(**values: Any) -> Iterator[dict[str, Any]]:
    """Bind key/value pairs into the ambient context for the duration of the block.

    Keys whose value is ``None`` are dropped so that optional identifiers do not
    pollute log output with nulls.

    Args:
        **values: Identifiers to bind.

    Yields:
        The merged context dictionary in effect inside the block.
    """
    merged = {
        **(_CONTEXT.get() or {}),
        **{key: val for key, val in values.items() if val is not None},
    }
    token = _CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _CONTEXT.reset(token)


@contextmanager
def search_context(search_id: str, *, strategy: str | None = None) -> Iterator[dict[str, Any]]:
    """Bind search-level identifiers.

    Args:
        search_id: Unique identifier of the search run.
        strategy: Name of the active search strategy.

    Yields:
        The merged context dictionary.
    """
    with bind_context(search_id=search_id, strategy=strategy) as context:
        yield context


@contextmanager
def candidate_context(
    candidate_id: str,
    *,
    architecture_hash: str | None = None,
    trial_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Bind candidate-level identifiers.

    Args:
        candidate_id: Unique identifier of the candidate.
        architecture_hash: Canonical architecture hash.
        trial_id: Identifier of the specific evaluation attempt.

    Yields:
        The merged context dictionary.
    """
    with bind_context(
        candidate_id=candidate_id,
        architecture_hash=architecture_hash,
        trial_id=trial_id,
    ) as context:
        yield context


@contextmanager
def worker_context(worker_id: int | str) -> Iterator[dict[str, Any]]:
    """Bind the worker identifier.

    Args:
        worker_id: Worker index or name.

    Yields:
        The merged context dictionary.
    """
    with bind_context(worker_id=str(worker_id)) as context:
        yield context


__all__ = [
    "bind_context",
    "candidate_context",
    "current_context",
    "search_context",
    "worker_context",
]
