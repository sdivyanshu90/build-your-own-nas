"""Online scalarisation: a single fitness value available *during* a search.

The problem
-----------
Search strategies need one number per candidate. Regularized evolution compares
tournament entrants; successive halving promotes the top fraction of a rung. Both need a
scalar, and both need it *immediately* after each evaluation.

But the weighted scorer in :mod:`nas_engine.objectives.scoring` uses population-relative
normalisation. Its output for a candidate changes whenever another candidate is added, so
a value computed at evaluation 10 is not comparable with one computed at evaluation 200.
Storing such a value and comparing it later would silently corrupt every selection
decision, and the corruption would be invisible — the numbers all look reasonable.

The rule used here
------------------
An **online** objective value is computed from stable normalisations only:

* ``NONE`` and ``REFERENCE`` normalisation are absolute — the same metric always maps to
  the same normalised value — so those objectives contribute directly.
* ``MINMAX``, ``ZSCORE``, and ``LOG`` are population-relative and are *excluded*.
* If that leaves nothing, the direction-corrected **primary** metric is used alone.

The consequence is stated plainly rather than hidden: with the default objective set,
online selection is driven by validation accuracy, and the secondary objectives shape the
*final* ranking and the Pareto front but not the evolutionary trajectory. A user who wants
latency to steer the search should give the latency objective a ``REFERENCE``
normalisation with an explicit reference value — which forces them to say what a
millisecond is worth, which is exactly the decision that cannot be made for them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from nas_engine.objectives.objective import NormalizationStrategy, ObjectiveSet

#: Normalisation strategies whose output does not depend on the rest of the population.
STABLE_STRATEGIES: frozenset[NormalizationStrategy] = frozenset(
    {NormalizationStrategy.NONE, NormalizationStrategy.REFERENCE}
)


def uses_stable_scalarization(objectives: ObjectiveSet) -> bool:
    """Report whether any objective can contribute to an online scalar value.

    Args:
        objectives: The objective set.

    Returns:
        ``True`` when at least one objective uses a population-independent normalisation.
    """
    return any(
        objective.normalization in STABLE_STRATEGIES and objective.weight > 0
        for objective in objectives.objectives
    )


def online_objective_value(metrics: Mapping[str, float], objectives: ObjectiveSet) -> float | None:
    """Compute a time-stable scalar fitness for one candidate.

    Larger is always better, regardless of each objective's direction.

    Args:
        metrics: Measured metrics for the candidate.
        objectives: The objective set.

    Returns:
        The scalar value, or ``None`` when the required metrics are absent.
    """
    stable = [
        objective
        for objective in objectives.objectives
        if objective.normalization in STABLE_STRATEGIES and objective.weight > 0
    ]

    if not stable:
        primary = objectives.primary
        value = metrics.get(primary.metric)
        if value is None or not math.isfinite(float(value)):
            return None
        return primary.direction.sign * float(value)

    total_weight = sum(objective.weight for objective in stable)
    accumulated = 0.0
    for objective in stable:
        if objective.metric in metrics:
            raw = float(metrics[objective.metric])
        elif objective.required:
            return None
        else:
            assert objective.missing_value is not None  # guaranteed by Objective validation
            raw = float(objective.missing_value)
        if math.isnan(raw):
            return None
        if math.isinf(raw):
            # A sentinel: contribute the worst finite-equivalent contribution instead of
            # poisoning the whole sum with an infinity.
            return -math.inf if objective.direction.sign * raw < 0 else math.inf
        scaled = raw / objective.reference if objective.reference else raw
        accumulated += objective.weight * objective.direction.sign * scaled
    return accumulated / total_weight


__all__ = ["STABLE_STRATEGIES", "online_objective_value", "uses_stable_scalarization"]
