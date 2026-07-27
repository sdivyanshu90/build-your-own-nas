"""Multi-objective comparison: objectives, constraints, scoring, Pareto fronts, ranking.

Everything here operates on plain ``Mapping[str, float]`` metrics. The package therefore
has no dependency on PyTorch, on the evaluator, or on persistence, and its logic can be
tested exhaustively with hand-written numbers.
"""

from nas_engine.objectives.constraints import (
    ComparisonOperator,
    ConstraintSet,
    MetricConstraint,
)
from nas_engine.objectives.objective import (
    NormalizationStrategy,
    Objective,
    ObjectiveDirection,
    ObjectiveSet,
    default_objectives,
)
from nas_engine.objectives.pareto import (
    ObjectiveVector,
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front,
    to_objective_vector,
)
from nas_engine.objectives.ranking import (
    RankedCandidate,
    RankingResult,
    rank_candidates,
)
from nas_engine.objectives.scoring import (
    NormalizerStats,
    ScoringResult,
    WeightedScorer,
    normalize_value,
)

__all__ = [
    "ComparisonOperator",
    "ConstraintSet",
    "MetricConstraint",
    "NormalizationStrategy",
    "NormalizerStats",
    "Objective",
    "ObjectiveDirection",
    "ObjectiveSet",
    "ObjectiveVector",
    "RankedCandidate",
    "RankingResult",
    "ScoringResult",
    "WeightedScorer",
    "crowding_distance",
    "default_objectives",
    "dominates",
    "non_dominated_sort",
    "normalize_value",
    "pareto_front",
    "rank_candidates",
    "to_objective_vector",
]
