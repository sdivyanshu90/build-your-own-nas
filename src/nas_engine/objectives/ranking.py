"""Reproducible candidate ranking.

Ranking combines every signal the framework has about a candidate into one total order.
"Total" matters: a leaderboard, a "best candidate" query, and a report table must all agree
and must not change between identical runs. Ties are therefore broken all the way down to
the candidate identifier, so no two candidates ever compare equal.

Ordering rules, applied in sequence
------------------------------------
1. **Feasibility.** Candidates violating a hard constraint rank below every feasible one.
   A constraint is a requirement, not a preference.
2. **Pareto front rank.** Lower is better. This respects the multi-objective structure
   before any scalarisation happens, so a candidate on the front cannot be pushed below a
   dominated one by a weighting choice.
3. **Weighted score.** Higher is better. Breaks ties *within* a front, where by definition
   no candidate dominates another.
4. **Primary objective.** Direction-corrected raw value. Kept as a distinct step so that a
   ranking is still meaningful when a score could not be computed.
5. **Candidate id.** Lexicographic, guaranteeing a total order and byte-identical output
   across runs.

Candidates whose required metrics are missing keep a ``None`` score and sort last among
their feasibility class, rather than being dropped: a report that silently omits rows is
worse than one that shows them as unscored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nas_engine.objectives.constraints import ConstraintSet
from nas_engine.objectives.objective import ObjectiveSet
from nas_engine.objectives.pareto import (
    ObjectiveVector,
    crowding_distance,
    non_dominated_sort,
    to_objective_vector,
)
from nas_engine.objectives.scoring import WeightedScorer

#: Sentinel front rank for candidates that could not be placed on any front.
UNRANKED_FRONT: int = 10**6


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate's position in a ranking.

    Attributes:
        candidate_id: Identifier.
        architecture_hash: Canonical architecture hash, carried for display.
        metrics: The metrics the ranking was computed from.
        rank: Zero-based position in the total order.
        score: Weighted scalar score, or ``None`` when unscoreable.
        pareto_rank: Zero-based non-dominated front index, or :data:`UNRANKED_FRONT`.
        crowding: NSGA-II crowding distance within the candidate's front.
        feasible: Whether every hard constraint was satisfied.
        violations: Human-readable constraint violations.
        score_components: Normalised per-objective contributions to the score.
    """

    candidate_id: str
    architecture_hash: str
    metrics: Mapping[str, float]
    rank: int
    score: float | None
    pareto_rank: int
    crowding: float
    feasible: bool
    violations: tuple[str, ...] = ()
    score_components: Mapping[str, float] = field(default_factory=dict)

    @property
    def on_pareto_front(self) -> bool:
        """Whether the candidate is feasible and on the first non-dominated front."""
        return self.feasible and self.pareto_rank == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "candidate_id": self.candidate_id,
            "architecture_hash": self.architecture_hash,
            "rank": self.rank,
            "score": self.score,
            "pareto_rank": self.pareto_rank if self.pareto_rank != UNRANKED_FRONT else None,
            "crowding": None if math.isinf(self.crowding) else self.crowding,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "metrics": dict(self.metrics),
            "score_components": dict(self.score_components),
        }


@dataclass(frozen=True)
class RankingResult:
    """A complete ranking of a candidate population.

    Attributes:
        ranked: Candidates in rank order.
        pareto_front: The feasible, non-dominated candidates.
        objectives: The objective set used.
        unscored: Candidates whose required metrics were missing.
    """

    ranked: tuple[RankedCandidate, ...]
    pareto_front: tuple[RankedCandidate, ...]
    objectives: ObjectiveSet
    unscored: tuple[str, ...] = ()

    @property
    def best(self) -> RankedCandidate | None:
        """The top-ranked candidate, or ``None`` when the population is empty."""
        return self.ranked[0] if self.ranked else None

    def by_id(self, candidate_id: str) -> RankedCandidate | None:
        """Return one candidate's ranking entry.

        Args:
            candidate_id: Identifier to look up.

        Returns:
            The entry, or ``None`` when the candidate was not ranked.
        """
        for candidate in self.ranked:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "objectives": [objective.describe() for objective in self.objectives.objectives],
            "ranked": [candidate.to_dict() for candidate in self.ranked],
            "pareto_front": [candidate.candidate_id for candidate in self.pareto_front],
            "unscored": list(self.unscored),
        }


def rank_candidates(
    population: Sequence[tuple[str, str, Mapping[str, float]]],
    objectives: ObjectiveSet,
    *,
    constraints: ConstraintSet | None = None,
) -> RankingResult:
    """Rank a population of candidates.

    Args:
        population: ``(candidate_id, architecture_hash, metrics)`` triples.
        objectives: Objectives to optimise.
        constraints: Hard constraints; ``None`` means every candidate is feasible.

    Returns:
        A :class:`RankingResult` whose order is deterministic for a given input.
    """
    constraint_set = constraints if constraints is not None else ConstraintSet()

    feasibility: dict[str, tuple[bool, tuple[str, ...]]] = {}
    for candidate_id, _, metrics in population:
        violations = constraint_set.violations(metrics)
        feasibility[candidate_id] = (not violations, violations)

    scorer = WeightedScorer(objectives, [(cid, metrics) for cid, _, metrics in population])
    scores = {
        result.candidate_id: result
        for result in scorer.score_all([(cid, metrics) for cid, _, metrics in population])
    }

    # Only feasible candidates take part in dominance: an infeasible candidate on the
    # trade-off surface would otherwise be reported as an option the user cannot take.
    vectors: list[ObjectiveVector] = []
    for candidate_id, _, metrics in population:
        if not feasibility[candidate_id][0]:
            continue
        vector = to_objective_vector(candidate_id, metrics, objectives)
        if vector is not None:
            vectors.append(vector)

    fronts = non_dominated_sort(vectors)
    front_rank: dict[str, int] = {}
    crowding: dict[str, float] = {}
    for index, front in enumerate(fronts):
        distances = crowding_distance(front)
        for vector in front:
            front_rank[vector.candidate_id] = index
            crowding[vector.candidate_id] = distances[vector.candidate_id]

    primary = objectives.primary

    def sort_key(entry: tuple[str, str, Mapping[str, float]]) -> tuple[Any, ...]:
        candidate_id, _, metrics = entry
        feasible = feasibility[candidate_id][0]
        rank = front_rank.get(candidate_id, UNRANKED_FRONT)
        score = scores[candidate_id].score
        primary_value = metrics.get(primary.metric)
        primary_sorted = (
            -primary.direction.sign * float(primary_value)
            if primary_value is not None and math.isfinite(float(primary_value))
            else math.inf
        )
        return (
            0 if feasible else 1,
            rank,
            -score if score is not None else math.inf,
            primary_sorted,
            candidate_id,
        )

    ordered = sorted(population, key=sort_key)
    ranked: list[RankedCandidate] = []
    for position, (candidate_id, architecture_hash, metrics) in enumerate(ordered):
        feasible, violations = feasibility[candidate_id]
        result = scores[candidate_id]
        ranked.append(
            RankedCandidate(
                candidate_id=candidate_id,
                architecture_hash=architecture_hash,
                metrics=dict(metrics),
                rank=position,
                score=result.score,
                pareto_rank=front_rank.get(candidate_id, UNRANKED_FRONT),
                crowding=crowding.get(candidate_id, 0.0),
                feasible=feasible,
                violations=violations,
                score_components=dict(result.components),
            )
        )

    return RankingResult(
        ranked=tuple(ranked),
        pareto_front=tuple(candidate for candidate in ranked if candidate.on_pareto_front),
        objectives=objectives,
        unscored=tuple(sorted(cid for cid, result in scores.items() if not result.is_scored)),
    )


__all__ = ["UNRANKED_FRONT", "RankedCandidate", "RankingResult", "rank_candidates"]
