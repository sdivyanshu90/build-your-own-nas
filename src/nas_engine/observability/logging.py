"""Structured logging configuration built on :mod:`structlog`.

Why structured logging
----------------------
A NAS run emits thousands of log lines across candidates, trials, and workers.
Free-text lines force downstream consumers to write brittle regexes. Structured
events carry typed key/value pairs instead, so ``search_id``, ``candidate_id``,
``architecture_hash``, and ``duration_seconds`` can be filtered and aggregated
directly, and the same event stream renders either as human-friendly console output
or as newline-delimited JSON for machine ingestion.

Redaction
---------
Configuration objects can legitimately contain paths, and a user may add fields whose
names suggest secrets (``token``, ``password``, ``api_key``). Rather than trusting
callers, a processor walks every event dictionary and replaces the value of any
allow-list-matching key with ``"***redacted***"``. This is defence in depth: the
framework itself never puts credentials in logs, but user configuration is untrusted.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Literal

import structlog

#: Placeholder substituted for sensitive values.
REDACTED = "***redacted***"

#: Substrings that mark a key as sensitive. Matching is case-insensitive and
#: substring-based so ``HF_API_TOKEN`` and ``db_password`` are both caught.
SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "auth",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)

LogFormat = Literal["console", "json"]

_CONFIGURED = False


def _is_sensitive(key: str) -> bool:
    """Return whether ``key`` looks like it holds a secret."""
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def redact_mapping(mapping: Mapping[str, Any], *, max_depth: int = 6) -> dict[str, Any]:
    """Return a copy of ``mapping`` with sensitive values replaced.

    Nested mappings and sequences are traversed up to ``max_depth`` levels. The depth
    cap prevents unbounded recursion on pathological or cyclic structures.

    Args:
        mapping: Mapping to redact.
        max_depth: Maximum recursion depth.

    Returns:
        A new dictionary safe to log.
    """

    def _walk(value: Any, depth: int) -> Any:
        if depth <= 0:
            return value
        if isinstance(value, Mapping):
            return {
                key: (REDACTED if _is_sensitive(str(key)) else _walk(item, depth - 1))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            walked = [_walk(item, depth - 1) for item in value]
            return type(value)(walked) if isinstance(value, tuple) else walked
        return value

    result = _walk(dict(mapping), max_depth)
    assert isinstance(result, dict)
    return result


def _redaction_processor(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that redacts sensitive keys anywhere in the event."""
    return redact_mapping(event_dict)


def _context_processor(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that merges the ambient identifier context into the event."""
    from nas_engine.observability.context import current_context

    for key, value in current_context().items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    log_format: LogFormat = "console",
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    """Configure process-wide structured logging.

    Safe to call repeatedly; subsequent calls are ignored unless ``force`` is set.
    Worker processes call this again after fork/spawn because logging configuration
    does not survive a ``spawn`` start method.

    Args:
        level: Minimum level name, e.g. ``"DEBUG"`` or ``"INFO"``.
        log_format: ``"console"`` for coloured human output, ``"json"`` for
            newline-delimited JSON.
        log_file: Optional file that receives the same records as stderr.
        force: Reconfigure even if logging was already configured.

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    global _CONFIGURED  # noqa: PLW0603 - module-level idempotency flag
    if _CONFIGURED and not force:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        msg = f"unknown log level {level!r}; expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
        raise ValueError(msg)

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(format="%(message)s", handlers=handlers, level=numeric_level, force=True)

    renderer: Any = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _context_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redaction_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Caching would freeze the processor chain into every logger created before the
        # first `configure_logging` call. Modules bind their logger at import time, so a
        # later reconfiguration — the engine switching to JSON, or a worker process
        # reconfiguring after spawn — would silently have no effect on them. The lookup
        # cost is negligible next to the work each event describes.
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger.

    Logging is configured lazily with defaults on first use so that library consumers
    who never call :func:`configure_logging` still get usable output.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A structlog bound logger.
    """
    if not _CONFIGURED:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


__all__ = [
    "REDACTED",
    "SENSITIVE_KEY_FRAGMENTS",
    "LogFormat",
    "configure_logging",
    "get_logger",
    "redact_mapping",
]
