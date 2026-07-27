"""Objective definitions.

An objective names a metric, states which direction is better, and says how the metric
should be normalised before it can be combined with other metrics.

Why normalisation is not optional
---------------------------------
Consider "maximise validation accuracy" (range :math:`[0, 1]`) and "minimise parameter
count" (range :math:`[10^3, 10^7]`). A weighted sum of the raw values is dominated
entirely by the parameter count; the accuracy term contributes less than one part in a
million. Any weighting that "works" is really just an inverse-scale correction discovered
by trial and error, and it stops working the moment the parameter range shifts.

Normalisation puts every objective on a comparable scale first, so weights express
*preferences* rather than *unit conversions*.

The remaining honesty problem
-----------------------------
Even normalised, a weighted sum encodes a fixed exchange rate — "1% accuracy is worth
100k parameters" — that is rarely something anyone actually believes across the whole
range. This is why the framework computes a Pareto front as well: the front makes the
trade-off visible instead of resolving it silently. See
``docs/concepts/multi-objective-optimization.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nas_engine.exceptions import ObjectiveError


class ObjectiveDirection(str, Enum):
    """Whether larger or smaller values of a metric are better.

    Members:
        MAXIMIZE: Higher is better (accuracy).
        MINIMIZE: Lower is better (latency, parameters, model size).
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    @property
    def sign(self) -> float:
        """``+1`` for maximisation, ``-1`` for minimisation.

        Multiplying a raw value by this converts any objective into a maximisation
        problem, which is how the dominance and ranking code treats them uniformly.
        """
        return 1.0 if self is ObjectiveDirection.MAXIMIZE else -1.0


class NormalizationStrategy(str, Enum):
    """How a metric is rescaled before entering a weighted score.

    Members:
        NONE: Use the raw value. Correct only when every objective already shares a scale.
        MINMAX: Rescale to ``[0, 1]`` using the population's observed range. Interpretable
            and bounded, but *population-relative*: adding one extreme candidate changes
            every other candidate's normalised value, so scores are only comparable within
            a single ranking call.
        ZSCORE: Subtract the population mean and divide by the standard deviation.
            Unbounded, but far less sensitive to a single outlier than min-max.
        LOG: Take ``log10(1 + x)`` before min-max scaling. The right choice for quantities
            that span orders of magnitude, such as parameter counts, where a linear scale
            makes every small model look identical.
        REFERENCE: Divide by a fixed reference value supplied on the objective. The only
            strategy that is stable across runs, and therefore the one to use when scores
            must be compared between searches.
    """

    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    LOG = "log"
    REFERENCE = "reference"


@dataclass(frozen=True)
class Objective:
    """One optimisation objective.

    Attributes:
        metric: Key of the metric in a candidate's metric mapping.
        direction: Whether higher or lower is better.
        weight: Relative importance in the weighted scalar score. Weights are normalised
            to sum to one before use, so only their ratios matter.
        normalization: How to rescale the metric before weighting.
        reference: Divisor used by :attr:`NormalizationStrategy.REFERENCE`.
        required: Whether a candidate missing this metric can be scored at all.
        missing_value: Value substituted when the metric is absent and not required.
            Should be a deliberately pessimistic value, so a candidate is never rewarded
            for failing to report a metric.

    Raises:
        ObjectiveError: If the configuration is internally inconsistent.
    """

    metric: str
    direction: ObjectiveDirection
    weight: float = 1.0
    normalization: NormalizationStrategy = NormalizationStrategy.MINMAX
    reference: float | None = None
    required: bool = True
    missing_value: float | None = None

    def __post_init__(self) -> None:
        """Validate the objective.

        Raises:
            ObjectiveError: If the metric name is empty, the weight is negative, the
                reference strategy has no reference, or an optional objective has no
                fallback value.
        """
        if not self.metric.strip():
            msg = "objective metric name must not be empty"
            raise ObjectiveError(msg)
        if self.weight < 0:
            msg = f"objective '{self.metric}' has negative weight {self.weight}"
            raise ObjectiveError(msg, details={"metric": self.metric, "weight": self.weight})
        if self.normalization is NormalizationStrategy.REFERENCE and (
            self.reference is None or self.reference == 0
        ):
            msg = (
                f"objective '{self.metric}' uses reference normalisation but "
                f"reference={self.reference}; supply a non-zero reference value"
            )
            raise ObjectiveError(msg, details={"metric": self.metric, "reference": self.reference})
        if not self.required and self.missing_value is None:
            msg = (
                f"objective '{self.metric}' is optional but has no missing_value; "
                "supply a pessimistic default so absent metrics cannot be rewarded"
            )
            raise ObjectiveError(msg, details={"metric": self.metric})

    def describe(self) -> str:
        """Return a short human-readable description."""
        return (
            f"{self.direction.value} {self.metric} "
            f"(weight {self.weight:g}, {self.normalization.value})"
        )


@dataclass(frozen=True)
class ObjectiveSet:
    """An ordered collection of objectives with normalised weights.

    Attributes:
        objectives: The objectives, in a fixed order. Order matters for deterministic
            tie-breaking and for the column order of exported tables.
    """

    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        """Validate the set.

        Raises:
            ObjectiveError: If the set is empty, contains duplicate metrics, or has zero
                total weight.
        """
        if not self.objectives:
            msg = "an objective set must contain at least one objective"
            raise ObjectiveError(msg)
        metrics = [objective.metric for objective in self.objectives]
        duplicates = {metric for metric in metrics if metrics.count(metric) > 1}
        if duplicates:
            msg = (
                f"objective set contains duplicate metrics {sorted(duplicates)}; each "
                "metric may appear at most once"
            )
            raise ObjectiveError(msg, details={"duplicates": sorted(duplicates)})
        if sum(objective.weight for objective in self.objectives) <= 0:
            msg = "objective weights sum to zero; at least one objective must have weight > 0"
            raise ObjectiveError(msg)

    @property
    def primary(self) -> Objective:
        """The first objective, used as the default sort key and tie-breaker."""
        return self.objectives[0]

    @property
    def metrics(self) -> tuple[str, ...]:
        """The metric names, in declaration order."""
        return tuple(objective.metric for objective in self.objectives)

    def normalized_weights(self) -> tuple[float, ...]:
        """Return weights scaled to sum to one.

        Returns:
            One weight per objective, in declaration order.
        """
        total = sum(objective.weight for objective in self.objectives)
        return tuple(objective.weight / total for objective in self.objectives)

    def by_metric(self, metric: str) -> Objective:
        """Return the objective for ``metric``.

        Args:
            metric: Metric name.

        Returns:
            The matching objective.

        Raises:
            ObjectiveError: If no objective uses that metric.
        """
        for objective in self.objectives:
            if objective.metric == metric:
                return objective
        msg = f"no objective is defined for metric '{metric}'; defined metrics are {self.metrics}"
        raise ObjectiveError(msg, details={"metric": metric, "available": list(self.metrics)})

    def describe(self) -> str:
        """Return a multi-line human-readable description."""
        weights = self.normalized_weights()
        lines = ["Objectives:"]
        lines.extend(
            f"  {objective.describe()} -> normalised weight {weight:.3f}"
            for objective, weight in zip(self.objectives, weights, strict=True)
        )
        return "\n".join(lines)


def default_objectives() -> ObjectiveSet:
    """Return the project's default objective set.

    Maximise validation accuracy, and secondarily minimise parameter count, estimated
    inference latency, and serialised model size. Accuracy dominates the weighting
    because the secondary objectives exist to break ties among comparably accurate
    architectures, not to trade away accuracy wholesale.

    Returns:
        The default :class:`ObjectiveSet`.
    """
    return ObjectiveSet(
        objectives=(
            Objective(
                metric="validation_accuracy",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=1.0,
                normalization=NormalizationStrategy.MINMAX,
            ),
            Objective(
                metric="trainable_parameters",
                direction=ObjectiveDirection.MINIMIZE,
                weight=0.2,
                normalization=NormalizationStrategy.LOG,
            ),
            Objective(
                metric="latency_median_ms",
                direction=ObjectiveDirection.MINIMIZE,
                weight=0.1,
                normalization=NormalizationStrategy.MINMAX,
                required=False,
                missing_value=float("inf"),
            ),
            Objective(
                metric="model_size_bytes",
                direction=ObjectiveDirection.MINIMIZE,
                weight=0.1,
                normalization=NormalizationStrategy.LOG,
                required=False,
                missing_value=float("inf"),
            ),
        )
    )


__all__ = [
    "NormalizationStrategy",
    "Objective",
    "ObjectiveDirection",
    "ObjectiveSet",
    "default_objectives",
]
