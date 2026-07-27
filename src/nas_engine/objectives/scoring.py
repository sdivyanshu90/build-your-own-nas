"""Weighted scalar scoring with population-relative normalisation.

The score collapses several objectives into one number so that candidates can be totally
ordered. That is genuinely useful — a leaderboard needs a single column — but it is a
lossy summary, and this module is explicit about how the loss happens.

Procedure
---------
1. Gather each objective's values across the whole population.
2. Fit a normaliser per objective (min-max, z-score, log-min-max, reference, or none).
3. Map every value into a "higher is better" normalised space.
4. Take the weighted mean using weights normalised to sum to one.

Consequences worth knowing
--------------------------
* Population-relative strategies (min-max, z-score, log) mean **scores are only comparable
  within one scoring call**. Adding a candidate can change every other score. Scores are
  therefore recomputed from persisted metrics whenever a leaderboard is displayed, never
  stored as ground truth.
* A degenerate objective — every candidate has the same value — carries no information.
  Min-max would divide by zero; here the objective contributes a constant 0.5 to every
  candidate, so it neither helps nor distorts.
* Reference normalisation is the only strategy stable across runs. Use it when comparing
  two searches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nas_engine.exceptions import ObjectiveError
from nas_engine.objectives.objective import (
    NormalizationStrategy,
    Objective,
    ObjectiveSet,
)

#: Value assigned when an objective carries no information (zero spread across the
#: population). The midpoint is neutral: it cannot favour or penalise any candidate.
NEUTRAL_NORMALIZED_VALUE: float = 0.5


@dataclass(frozen=True)
class NormalizerStats:
    """Population statistics for one objective.

    Attributes:
        metric: Metric name.
        minimum: Smallest finite value observed.
        maximum: Largest finite value observed.
        mean: Arithmetic mean of finite values.
        std: Population standard deviation of finite values.
        count: Number of finite values.
    """

    metric: str
    minimum: float
    maximum: float
    mean: float
    std: float
    count: int

    @property
    def degenerate(self) -> bool:
        """Whether the objective has no spread and therefore no information."""
        return self.count == 0 or math.isclose(self.minimum, self.maximum, rel_tol=1e-12)


def _finite_values(values: Sequence[float]) -> list[float]:
    """Return only the finite entries of ``values``."""
    return [value for value in values if math.isfinite(value)]


def compute_stats(metric: str, values: Sequence[float]) -> NormalizerStats:
    """Summarise a population of values for one objective.

    Infinite values (used as pessimistic sentinels for missing optional metrics) are
    excluded from the statistics: a single ``inf`` would otherwise make the range
    infinite and collapse every real value onto zero.

    Args:
        metric: Metric name.
        values: Observed values across the population.

    Returns:
        The statistics.
    """
    finite = _finite_values(values)
    if not finite:
        return NormalizerStats(metric, 0.0, 0.0, 0.0, 0.0, 0)
    minimum = min(finite)
    maximum = max(finite)
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return NormalizerStats(metric, minimum, maximum, mean, math.sqrt(variance), len(finite))


def normalize_value(  # noqa: PLR0911, PLR0912 - one branch per normalisation strategy
    value: float, objective: Objective, stats: NormalizerStats
) -> float:
    """Map a raw metric value into a normalised "higher is better" score component.

    Args:
        value: Raw metric value.
        objective: Objective describing direction and normalisation.
        stats: Population statistics for this objective.

    Returns:
        A normalised value where larger is always better. Min-max and log strategies
        return values in ``[0, 1]``; z-score and none are unbounded.

    Raises:
        ObjectiveError: If reference normalisation is used without a reference.
    """
    if math.isinf(value):
        # An infinite value is a sentinel, not a measurement. After direction correction,
        # +inf means "best imaginable" and -inf means "worst imaginable"; both are clamped
        # to the ends of the normalised range so they cannot destabilise the arithmetic.
        return 1.0 if objective.direction.sign * value > 0 else 0.0
    if math.isnan(value):
        # NaN means the measurement failed. Scoring it as the worst possible value stops a
        # broken measurement from accidentally winning.
        return 0.0

    strategy = objective.normalization
    if strategy is NormalizationStrategy.NONE:
        return objective.direction.sign * value

    if strategy is NormalizationStrategy.REFERENCE:
        if objective.reference is None or objective.reference == 0:
            msg = (
                f"objective '{objective.metric}' uses reference normalisation but has no "
                "usable reference value"
            )
            raise ObjectiveError(msg, details={"metric": objective.metric})
        ratio = value / objective.reference
        return objective.direction.sign * ratio

    if stats.degenerate:
        return NEUTRAL_NORMALIZED_VALUE

    if strategy is NormalizationStrategy.ZSCORE:
        if stats.std <= 0:
            return NEUTRAL_NORMALIZED_VALUE
        return objective.direction.sign * (value - stats.mean) / stats.std

    if strategy is NormalizationStrategy.LOG:
        # log10(1 + x) keeps the transform defined at zero and monotonic for x >= 0.
        # Negative values fall back to linear min-max, since a log scale is meaningless
        # there and silently clamping would corrupt the ordering.
        if value < 0 or stats.minimum < 0:
            scaled = (value - stats.minimum) / (stats.maximum - stats.minimum)
        else:
            low = math.log10(1.0 + stats.minimum)
            high = math.log10(1.0 + stats.maximum)
            if math.isclose(low, high, rel_tol=1e-12):
                return NEUTRAL_NORMALIZED_VALUE
            scaled = (math.log10(1.0 + value) - low) / (high - low)
    else:  # NormalizationStrategy.MINMAX
        scaled = (value - stats.minimum) / (stats.maximum - stats.minimum)

    scaled = min(1.0, max(0.0, scaled))
    return scaled if objective.direction.sign > 0 else 1.0 - scaled


@dataclass(frozen=True)
class ScoringResult:
    """The outcome of scoring one candidate.

    Attributes:
        candidate_id: Identifier of the candidate.
        score: Weighted scalar score, or ``None`` when a required metric was missing.
        components: Normalised contribution per objective metric.
        missing_metrics: Required metrics that were absent.
    """

    candidate_id: str
    score: float | None
    components: dict[str, float]
    missing_metrics: tuple[str, ...] = ()

    @property
    def is_scored(self) -> bool:
        """Whether a score could be computed."""
        return self.score is not None


class WeightedScorer:
    """Computes weighted scalar scores for a population of candidates.

    The scorer is constructed against a *population* because most normalisation strategies
    need its statistics. Scoring a single candidate in isolation is possible only with
    ``NONE`` or ``REFERENCE`` normalisation.

    Args:
        objectives: Objectives to score against.
        population: ``(candidate_id, metrics)`` pairs used to fit the normalisers.
    """

    def __init__(
        self,
        objectives: ObjectiveSet,
        population: Sequence[tuple[str, Mapping[str, float]]],
    ) -> None:
        self._objectives = objectives
        self._weights = objectives.normalized_weights()
        self._stats: dict[str, NormalizerStats] = {}
        for objective in objectives.objectives:
            values = [
                float(metrics[objective.metric])
                for _, metrics in population
                if objective.metric in metrics
            ]
            self._stats[objective.metric] = compute_stats(objective.metric, values)

    @property
    def stats(self) -> dict[str, NormalizerStats]:
        """Fitted population statistics, keyed by metric."""
        return dict(self._stats)

    def score(self, candidate_id: str, metrics: Mapping[str, float]) -> ScoringResult:
        """Score one candidate.

        Args:
            candidate_id: Identifier of the candidate.
            metrics: Measured metrics.

        Returns:
            A :class:`ScoringResult`. ``score`` is ``None`` when a required metric was
            missing, which keeps unscoreable candidates out of the leaderboard rather than
            giving them a fabricated value.
        """
        components: dict[str, float] = {}
        missing: list[str] = []
        total = 0.0
        for objective, weight in zip(self._objectives.objectives, self._weights, strict=True):
            if objective.metric in metrics:
                value = float(metrics[objective.metric])
            elif objective.required:
                missing.append(objective.metric)
                continue
            else:
                assert objective.missing_value is not None  # guaranteed by validation
                value = float(objective.missing_value)
            normalized = normalize_value(value, objective, self._stats[objective.metric])
            components[objective.metric] = normalized
            total += weight * normalized

        if missing:
            return ScoringResult(
                candidate_id=candidate_id,
                score=None,
                components=components,
                missing_metrics=tuple(missing),
            )
        return ScoringResult(candidate_id=candidate_id, score=total, components=components)

    def score_all(
        self, population: Sequence[tuple[str, Mapping[str, float]]]
    ) -> list[ScoringResult]:
        """Score every candidate in ``population``.

        Args:
            population: ``(candidate_id, metrics)`` pairs.

        Returns:
            One result per candidate, in input order.
        """
        return [self.score(candidate_id, metrics) for candidate_id, metrics in population]


__all__ = [
    "NEUTRAL_NORMALIZED_VALUE",
    "NormalizerStats",
    "ScoringResult",
    "WeightedScorer",
    "compute_stats",
    "normalize_value",
]
