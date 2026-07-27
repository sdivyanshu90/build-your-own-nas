"""Timing helpers and timezone-aware timestamps.

Two rules are enforced throughout the codebase:

* **Durations** use :func:`time.perf_counter`, a monotonic clock that is immune to
  NTP adjustments and daylight-saving jumps. Using wall-clock differences for
  durations can produce negative elapsed times.
* **Timestamps** are always timezone-aware UTC. Naive datetimes compare incorrectly
  across processes and are ambiguous once persisted, so :func:`utc_now` is the only
  sanctioned source of "now" in this project.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import TracebackType


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC :class:`datetime`."""
    return datetime.now(tz=timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with explicit offset."""
    return utc_now().isoformat()


class Stopwatch:
    """Monotonic elapsed-time measurement usable as a context manager.

    Example:
        >>> with Stopwatch() as watch:
        ...     _ = sum(range(1000))
        >>> watch.elapsed_seconds >= 0.0
        True
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._stop: float | None = None

    def start(self) -> Stopwatch:
        """Begin (or restart) measurement and return ``self``."""
        self._start = time.perf_counter()
        self._stop = None
        return self

    def stop(self) -> float:
        """Stop measurement and return the elapsed seconds.

        Raises:
            RuntimeError: If the stopwatch was never started.
        """
        if self._start is None:
            msg = "Stopwatch.stop() called before start()"
            raise RuntimeError(msg)
        self._stop = time.perf_counter()
        return self._stop - self._start

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed seconds; live while running, frozen once stopped.

        Raises:
            RuntimeError: If the stopwatch was never started.
        """
        if self._start is None:
            msg = "Stopwatch.elapsed_seconds accessed before start()"
            raise RuntimeError(msg)
        end = self._stop if self._stop is not None else time.perf_counter()
        return end - self._start

    def __enter__(self) -> Stopwatch:
        """Start the stopwatch on context entry."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the stopwatch on context exit, regardless of exceptions."""
        self.stop()


__all__ = ["Stopwatch", "utc_now", "utc_now_iso"]
