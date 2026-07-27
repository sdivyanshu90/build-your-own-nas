"""Search strategies and the interface every strategy implements.

This package depends on :mod:`nas_engine.search_space` (to sample and mutate) and on
:mod:`nas_engine.evaluation` (only for the :class:`~nas_engine.evaluation.budget.TrainingBudget`
and result types). It has no dependency on the orchestration engine, on persistence, or on
PyTorch training code — a strategy that could not be tested without a GPU would not be
testable at all.
"""

from nas_engine.search.evolution import PopulationMember, RegularizedEvolution
from nas_engine.search.random_search import RandomSearch
from nas_engine.search.registry import (
    available_strategies,
    build_strategy,
    register_strategy,
)
from nas_engine.search.strategy import (
    Observation,
    Proposal,
    SearchStrategy,
    StrategyStatistics,
)
from nas_engine.search.successive_halving import ResourceLadder, SuccessiveHalving

__all__ = [
    "Observation",
    "PopulationMember",
    "Proposal",
    "RandomSearch",
    "RegularizedEvolution",
    "ResourceLadder",
    "SearchStrategy",
    "StrategyStatistics",
    "SuccessiveHalving",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
