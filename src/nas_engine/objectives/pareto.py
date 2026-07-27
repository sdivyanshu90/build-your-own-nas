r"""Pareto dominance and front computation.

Definitions
-----------
Write each candidate's objective vector in *maximisation form*: multiply every
minimisation objective by :math:`-1` so that larger is better on every axis. Then
candidate :math:`a` **dominates** :math:`b`, written :math:`a \succ b`, when

.. math::

    \forall i:\; a_i \ge b_i \quad\text{and}\quad \exists j:\; a_j > b_j

— :math:`a` is at least as good everywhere and strictly better somewhere. Domination is a
strict partial order: irreflexive, asymmetric, and transitive, but *not* total. Two
candidates where each wins on a different axis are **incomparable**, and no amount of
argument resolves which is better without a stated preference.

The **Pareto front** is the set of non-dominated candidates. Every member represents a
trade-off that cannot be improved on one axis without giving something up on another, and
every non-member is strictly worse than some front member on every axis at once. Reporting
the front is the honest answer to a multi-objective search: it hands the decision back to
whoever owns the preference.

Numerical care
--------------
Floating-point metrics from short training runs are noisy. Comparing them with exact
equality means two candidates differing by :math:`10^{-15}` are treated as genuinely
different, and the front fills up with numerical dust. Comparisons therefore use a
relative tolerance, and :func:`pareto_front` returns candidates in a deterministic order
so that repeated calls on the same data give identical output.

Complexity
----------
The naive algorithm is :math:`O(n^2 m)` for :math:`n` candidates and :math:`m` objectives.
At NAS scale (hundreds to low thousands of candidates) that is microseconds to
milliseconds, and it is chosen over asymptotically better divide-and-conquer algorithms
because it is obviously correct and easy to test. The performance test in
``tests/performance`` pins the practical limits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nas_engine.exceptions import ObjectiveError
from nas_engine.objectives.objective import ObjectiveSet

#: Relative tolerance used when comparing objective values. Two values within this
#: relative distance are treated as equal, which keeps floating-point noise out of the
#: front.
DOMINANCE_RELATIVE_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class ObjectiveVector:
    """A candidate's objective values in maximisation form.

    Attributes:
        candidate_id: Identifier of the candidate.
        values: One value per objective, already sign-corrected so larger is better.
        raw: The original metric values, kept for display.
    """

    candidate_id: str
    values: tuple[float, ...]
    raw: tuple[float, ...]


def _approximately_equal(left: float, right: float) -> bool:
    """Return whether two values are equal within the dominance tolerance."""
    return math.isclose(left, right, rel_tol=DOMINANCE_RELATIVE_TOLERANCE, abs_tol=0.0)


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``.

    Both vectors must already be in maximisation form.

    Args:
        left: Candidate objective vector.
        right: Candidate objective vector.

    Returns:
        ``True`` when ``left`` is at least as good on every objective and strictly better
        on at least one.

    Raises:
        ObjectiveError: If the vectors have different lengths or are empty.
    """
    if len(left) != len(right):
        msg = f"cannot compare objective vectors of different lengths: {len(left)} and {len(right)}"
        raise ObjectiveError(msg, details={"left": len(left), "right": len(right)})
    if not left:
        msg = "cannot compare empty objective vectors"
        raise ObjectiveError(msg)

    strictly_better = False
    for left_value, right_value in zip(left, right, strict=True):
        if math.isnan(left_value) or math.isnan(right_value):
            # A NaN objective means the metric is unusable; refusing to let it participate
            # in dominance stops a failed measurement from silently winning.
            return False
        if _approximately_equal(left_value, right_value):
            continue
        if left_value < right_value:
            return False
        strictly_better = True
    return strictly_better


def to_objective_vector(
    candidate_id: str, metrics: Mapping[str, float], objectives: ObjectiveSet
) -> ObjectiveVector | None:
    """Convert a metric mapping into a maximisation-form objective vector.

    Args:
        candidate_id: Identifier of the candidate.
        metrics: Measured metrics.
        objectives: Objectives defining which metrics matter and in which direction.

    Returns:
        The vector, or ``None`` when a required metric is missing and the candidate
        therefore cannot participate in dominance comparisons.
    """
    values: list[float] = []
    raw: list[float] = []
    for objective in objectives.objectives:
        if objective.metric in metrics:
            value = float(metrics[objective.metric])
        elif objective.required:
            return None
        else:
            assert objective.missing_value is not None  # guaranteed by Objective validation
            value = float(objective.missing_value)
        raw.append(value)
        if math.isinf(value):
            # An infinite value is a legitimate "worst possible" sentinel for an optional
            # metric. Signed multiplication keeps it at the worst end after direction
            # correction.
            values.append(objective.direction.sign * value)
        else:
            values.append(objective.direction.sign * value)
    return ObjectiveVector(candidate_id=candidate_id, values=tuple(values), raw=tuple(raw))


def pareto_front(vectors: Sequence[ObjectiveVector]) -> list[ObjectiveVector]:
    """Return the non-dominated subset of ``vectors``.

    Args:
        vectors: Candidate objective vectors in maximisation form.

    Returns:
        The Pareto-optimal vectors, ordered by candidate id for determinism.
    """
    front: list[ObjectiveVector] = []
    for candidate in vectors:
        if any(
            dominates(other.values, candidate.values)
            for other in vectors
            if other.candidate_id != candidate.candidate_id
        ):
            continue
        front.append(candidate)
    return sorted(front, key=lambda vector: vector.candidate_id)


def non_dominated_sort(vectors: Sequence[ObjectiveVector]) -> list[list[ObjectiveVector]]:
    """Partition candidates into successive Pareto fronts.

    Front 0 is the Pareto front. Front 1 is the Pareto front of what remains after
    removing front 0, and so on. This "non-dominated sorting" gives every candidate a
    rank, which is what makes multi-objective selection possible in an evolutionary loop:
    a lower front rank is unambiguously better.

    Args:
        vectors: Candidate objective vectors in maximisation form.

    Returns:
        A list of fronts, best first. Each front is sorted by candidate id.
    """
    remaining = list(vectors)
    fronts: list[list[ObjectiveVector]] = []
    while remaining:
        current = pareto_front(remaining)
        if not current:  # pragma: no cover - a non-empty set always has a non-empty front
            break
        fronts.append(current)
        selected = {vector.candidate_id for vector in current}
        remaining = [vector for vector in remaining if vector.candidate_id not in selected]
    return fronts


def crowding_distance(front: Sequence[ObjectiveVector]) -> dict[str, float]:
    """Compute NSGA-II crowding distance for one front.

    Within a front no candidate dominates another, so a secondary criterion is needed to
    choose between them. Crowding distance measures how isolated a candidate is: for each
    objective, sort the front and sum each candidate's normalised gap to its two
    neighbours. Extreme candidates receive infinite distance so the ends of the trade-off
    curve are never discarded.

    Preferring high crowding distance spreads selection along the front instead of letting
    the population cluster in one region of the trade-off space.

    Args:
        front: A single Pareto front.

    Returns:
        Crowding distance keyed by candidate id.
    """
    distances: dict[str, float] = {vector.candidate_id: 0.0 for vector in front}
    if len(front) <= 2:
        return {vector.candidate_id: math.inf for vector in front}

    objective_count = len(front[0].values)
    for index in range(objective_count):
        ordered = sorted(front, key=lambda vector: vector.values[index])
        lowest = ordered[0].values[index]
        highest = ordered[-1].values[index]
        span = highest - lowest
        distances[ordered[0].candidate_id] = math.inf
        distances[ordered[-1].candidate_id] = math.inf
        if span <= 0 or math.isinf(span):
            continue
        for position in range(1, len(ordered) - 1):
            candidate_id = ordered[position].candidate_id
            if math.isinf(distances[candidate_id]):
                continue
            gap = ordered[position + 1].values[index] - ordered[position - 1].values[index]
            distances[candidate_id] += gap / span
    return distances


__all__ = [
    "DOMINANCE_RELATIVE_TOLERANCE",
    "ObjectiveVector",
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "pareto_front",
    "to_objective_vector",
]
