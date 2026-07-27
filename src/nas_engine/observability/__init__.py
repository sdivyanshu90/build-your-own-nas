"""Structured logging, event vocabulary, and in-process counters."""

from nas_engine.observability.context import (
    bind_context,
    candidate_context,
    current_context,
    search_context,
    worker_context,
)
from nas_engine.observability.events import Event, emit
from nas_engine.observability.logging import configure_logging, get_logger, redact_mapping
from nas_engine.observability.metrics import CounterRegistry, MetricsSnapshot

__all__ = [
    "CounterRegistry",
    "Event",
    "MetricsSnapshot",
    "bind_context",
    "candidate_context",
    "configure_logging",
    "current_context",
    "emit",
    "get_logger",
    "redact_mapping",
    "search_context",
    "worker_context",
]
