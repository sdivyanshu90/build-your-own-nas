"""In-process counters and gauges.

Requirements state that the framework must expose counters suitable for future
integration with a metrics system *without* depending on one. This module therefore
implements the smallest useful surface: monotonically increasing counters, last-value
gauges, and simple duration observations that report count / sum / min / max / mean.

The registry is thread-safe. It is deliberately *not* process-shared: each worker
keeps its own registry and returns a snapshot to the parent, which merges them. That
avoids shared-memory complexity and the associated locking bugs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DurationSummary:
    """Aggregate statistics for a set of duration observations.

    Attributes:
        count: Number of observations.
        total_seconds: Sum of observed durations.
        min_seconds: Smallest observation.
        max_seconds: Largest observation.
    """

    count: int
    total_seconds: float
    min_seconds: float
    max_seconds: float

    @property
    def mean_seconds(self) -> float:
        """Arithmetic mean, or ``0.0`` when there are no observations."""
        return self.total_seconds / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable representation."""
        return {
            "count": float(self.count),
            "total_seconds": self.total_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
        }


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable point-in-time view of a :class:`CounterRegistry`.

    Attributes:
        counters: Monotonic counters by name.
        gauges: Last observed value by name.
        durations: Duration aggregates by name.
    """

    counters: dict[str, int] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    durations: dict[str, DurationSummary] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "durations": {name: value.to_dict() for name, value in self.durations.items()},
        }

    def merge(self, other: MetricsSnapshot) -> MetricsSnapshot:
        """Combine two snapshots, summing counters and unioning duration statistics.

        Gauges take the *other* snapshot's value when both define the same name, since
        a gauge represents the most recent observation.

        Args:
            other: Snapshot to merge into a copy of this one.

        Returns:
            A new merged snapshot.
        """
        counters = dict(self.counters)
        for name, value in other.counters.items():
            counters[name] = counters.get(name, 0) + value

        gauges = {**self.gauges, **other.gauges}

        durations = dict(self.durations)
        for name, summary in other.durations.items():
            existing = durations.get(name)
            if existing is None:
                durations[name] = summary
            else:
                durations[name] = DurationSummary(
                    count=existing.count + summary.count,
                    total_seconds=existing.total_seconds + summary.total_seconds,
                    min_seconds=min(existing.min_seconds, summary.min_seconds),
                    max_seconds=max(existing.max_seconds, summary.max_seconds),
                )
        return MetricsSnapshot(counters=counters, gauges=gauges, durations=durations)


class CounterRegistry:
    """Thread-safe registry of counters, gauges, and duration observations.

    Example:
        >>> registry = CounterRegistry()
        >>> registry.increment("candidates.proposed")
        1
        >>> registry.observe_duration("evaluation", 1.5)
        >>> registry.snapshot().counters["candidates.proposed"]
        1
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._durations: dict[str, list[float]] = {}

    def increment(self, name: str, amount: int = 1) -> int:
        """Increase a counter and return its new value.

        Args:
            name: Counter name.
            amount: Non-negative increment.

        Returns:
            The counter value after incrementing.

        Raises:
            ValueError: If ``amount`` is negative; counters must be monotonic.
        """
        if amount < 0:
            msg = f"counter increments must be non-negative, received {amount}"
            raise ValueError(msg)
        with self._lock:
            value = self._counters.get(name, 0) + amount
            self._counters[name] = value
            return value

    def set_gauge(self, name: str, value: float) -> None:
        """Record the latest value of a gauge.

        Args:
            name: Gauge name.
            value: Latest observation.
        """
        with self._lock:
            self._gauges[name] = float(value)

    def observe_duration(self, name: str, seconds: float) -> None:
        """Record a duration observation.

        Args:
            name: Observation name.
            seconds: Non-negative duration.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            msg = f"duration observations must be non-negative, received {seconds}"
            raise ValueError(msg)
        with self._lock:
            self._durations.setdefault(name, []).append(float(seconds))

    def counter(self, name: str) -> int:
        """Return the current value of a counter, or ``0`` if unseen."""
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> MetricsSnapshot:
        """Return an immutable copy of the current registry contents."""
        with self._lock:
            durations = {
                name: DurationSummary(
                    count=len(values),
                    total_seconds=sum(values),
                    min_seconds=min(values),
                    max_seconds=max(values),
                )
                for name, values in self._durations.items()
                if values
            }
            return MetricsSnapshot(
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                durations=durations,
            )

    def reset(self) -> None:
        """Clear all recorded metrics. Intended for tests."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._durations.clear()


__all__ = ["CounterRegistry", "DurationSummary", "MetricsSnapshot"]
