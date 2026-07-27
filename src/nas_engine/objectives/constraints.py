"""Hard constraints on candidate metrics.

Constraints differ from objectives in kind, not degree. An objective says "smaller is
better"; a constraint says "larger than this is unacceptable at any price". Encoding a
hard requirement as a heavily weighted objective is a common mistake: with enough
accuracy, a weighted score will always eventually buy its way past the limit, so a model
that cannot fit on the target device gets recommended anyway.

Constraints are evaluated on the *measured* metrics of a completed candidate. They are
distinct from :class:`~nas_engine.search_space.space.SpaceConstraints`, which are analytic
and applied before training. Both exist because some quantities (measured latency, actual
serialised size) simply cannot be known without running the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from nas_engine.exceptions import ObjectiveError


class ComparisonOperator(str, Enum):
    """Comparison used by a constraint.

    Members:
        LE: less than or equal.
        LT: strictly less than.
        GE: greater than or equal.
        GT: strictly greater than.
    """

    LE = "le"
    LT = "lt"
    GE = "ge"
    GT = "gt"

    def compare(self, value: float, threshold: float) -> bool:
        """Evaluate ``value <op> threshold``.

        Args:
            value: Observed value.
            threshold: Constraint threshold.

        Returns:
            ``True`` when the constraint is satisfied.
        """
        if self is ComparisonOperator.LE:
            return value <= threshold
        if self is ComparisonOperator.LT:
            return value < threshold
        if self is ComparisonOperator.GE:
            return value >= threshold
        return value > threshold

    @property
    def symbol(self) -> str:
        """A human-readable symbol for messages and reports."""
        return {"le": "<=", "lt": "<", "ge": ">=", "gt": ">"}[self.value]


@dataclass(frozen=True)
class MetricConstraint:
    """A hard limit on one measured metric.

    Attributes:
        metric: Metric key to test.
        operator: Comparison to apply.
        threshold: Value compared against.
        required: Whether a candidate missing this metric is treated as infeasible.
            ``True`` is the safe default: an unmeasured latency is not a passing latency.

    Raises:
        ObjectiveError: If the metric name is empty.
    """

    metric: str
    operator: ComparisonOperator
    threshold: float
    required: bool = True

    def __post_init__(self) -> None:
        """Validate the constraint.

        Raises:
            ObjectiveError: If the metric name is empty.
        """
        if not self.metric.strip():
            msg = "constraint metric name must not be empty"
            raise ObjectiveError(msg)

    def describe(self) -> str:
        """Return a short human-readable description."""
        return f"{self.metric} {self.operator.symbol} {self.threshold:g}"

    def evaluate(self, metrics: Mapping[str, float]) -> str | None:
        """Test the constraint against a metric mapping.

        Args:
            metrics: Measured metrics for one candidate.

        Returns:
            ``None`` when satisfied, otherwise a human-readable violation description.
        """
        if self.metric not in metrics:
            if self.required:
                return (
                    f"{self.metric} is required by constraint '{self.describe()}' but was "
                    "not measured"
                )
            return None
        value = metrics[self.metric]
        if self.operator.compare(value, self.threshold):
            return None
        return f"{self.metric}={value:g} violates {self.describe()}"


@dataclass(frozen=True)
class ConstraintSet:
    """A collection of hard constraints evaluated together.

    Attributes:
        constraints: The constraints, in declaration order.
    """

    constraints: tuple[MetricConstraint, ...] = ()

    def violations(self, metrics: Mapping[str, float]) -> tuple[str, ...]:
        """Return every violation for one candidate.

        All constraints are evaluated rather than short-circuiting on the first failure,
        so a user fixing a configuration sees the complete picture at once.

        Args:
            metrics: Measured metrics for one candidate.

        Returns:
            Violation descriptions; empty when the candidate is feasible.
        """
        found = [constraint.evaluate(metrics) for constraint in self.constraints]
        return tuple(message for message in found if message is not None)

    def is_feasible(self, metrics: Mapping[str, float]) -> bool:
        """Report whether a candidate satisfies every constraint.

        Args:
            metrics: Measured metrics for one candidate.

        Returns:
            ``True`` when there are no violations.
        """
        return not self.violations(metrics)

    def describe(self) -> str:
        """Return a multi-line human-readable description."""
        if not self.constraints:
            return "Constraints: none"
        lines = ["Constraints:"]
        lines.extend(f"  {constraint.describe()}" for constraint in self.constraints)
        return "\n".join(lines)


__all__ = ["ComparisonOperator", "ConstraintSet", "MetricConstraint"]
