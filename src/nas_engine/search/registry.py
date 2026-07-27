"""Search-strategy registry.

Strategies are looked up by name so that a YAML configuration can select one without the
configuration loader importing every strategy module, and so that a third-party strategy
can be plugged in with :func:`register_strategy` rather than by editing this package.

The registry deliberately does **not** import modules named in configuration. Doing so
would make a configuration file into executable code, which the security model forbids —
see ``docs/architecture/security.md``. Registration is an explicit Python call the user
makes in their own process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import ConfigurationError
from nas_engine.search.evolution import RegularizedEvolution
from nas_engine.search.random_search import RandomSearch
from nas_engine.search.strategy import SearchStrategy
from nas_engine.search.successive_halving import ResourceLadder, SuccessiveHalving
from nas_engine.search_space.space import SearchSpace

#: A factory receives the space, a seed, the base budget, an evaluation cap, the dataset's
#: native resolution, and strategy-specific parameters.
StrategyFactory = Callable[..., SearchStrategy]


def _build_random_search(
    *,
    space: SearchSpace,
    seed: int,
    budget: TrainingBudget,
    max_evaluations: int,
    native_resolution: int | None,
    params: dict[str, Any],
) -> SearchStrategy:
    """Build a :class:`~nas_engine.search.random_search.RandomSearch`."""
    return RandomSearch(
        space,
        seed=seed,
        max_evaluations=max_evaluations,
        budget=budget,
        sample_attempts=int(params.get("sample_attempts", 200)),
        max_consecutive_exhaustions=int(params.get("max_consecutive_exhaustions", 3)),
    )


def _build_regularized_evolution(
    *,
    space: SearchSpace,
    seed: int,
    budget: TrainingBudget,
    max_evaluations: int,
    native_resolution: int | None,
    params: dict[str, Any],
) -> SearchStrategy:
    """Build a :class:`~nas_engine.search.evolution.RegularizedEvolution`."""
    population_size = int(params.get("population_size", 16))
    default_tournament = max(2, population_size // 4)
    return RegularizedEvolution(
        space,
        seed=seed,
        max_evaluations=max_evaluations,
        budget=budget,
        population_size=population_size,
        tournament_size=int(params.get("tournament_size", default_tournament)),
        allow_duplicate_children=bool(params.get("allow_duplicate_children", False)),
        mutation_attempts=int(params.get("mutation_attempts", 25)),
    )


def _build_successive_halving(
    *,
    space: SearchSpace,
    seed: int,
    budget: TrainingBudget,
    max_evaluations: int,
    native_resolution: int | None,
    params: dict[str, Any],
) -> SearchStrategy:
    """Build a :class:`~nas_engine.search.successive_halving.SuccessiveHalving`.

    ``initial_candidates`` defaults to a value derived from ``max_evaluations`` so that the
    whole bracket fits inside the configured evaluation budget rather than overrunning it.
    """
    reduction_factor = float(params.get("reduction_factor", 3.0))
    num_rungs = int(params.get("num_rungs", 3))
    ladder = ResourceLadder(
        base_budget=budget,
        num_rungs=num_rungs,
        reduction_factor=reduction_factor,
        scale_epochs=bool(params.get("scale_epochs", True)),
        scale_train_fraction=bool(params.get("scale_train_fraction", False)),
        scale_resolution=bool(params.get("scale_resolution", False)),
        native_resolution=native_resolution,
    )

    initial = params.get("initial_candidates")
    if initial is None:
        # Each rung costs roughly the same, so the bracket's evaluation count is about
        # `n * sum(eta^-r)`. Invert that to fit inside `max_evaluations`.
        geometric = sum(reduction_factor**-rung for rung in range(num_rungs))
        initial = max(1, int(max_evaluations / geometric))
    return SuccessiveHalving(
        space,
        seed=seed,
        ladder=ladder,
        initial_candidates=int(initial),
    )


_REGISTRY: dict[str, StrategyFactory] = {
    RandomSearch.name: _build_random_search,
    RegularizedEvolution.name: _build_regularized_evolution,
    SuccessiveHalving.name: _build_successive_halving,
}


def register_strategy(name: str, factory: StrategyFactory, *, overwrite: bool = False) -> None:
    """Register a strategy factory under ``name``.

    Args:
        name: Registry key used in configuration.
        factory: Callable accepting the keyword arguments listed in
            :data:`StrategyFactory` and returning a
            :class:`~nas_engine.search.strategy.SearchStrategy`.
        overwrite: Whether replacing an existing registration is permitted.

    Raises:
        ConfigurationError: If the name is taken and ``overwrite`` is ``False``.
    """
    if name in _REGISTRY and not overwrite:
        msg = f"search strategy '{name}' is already registered; pass overwrite=True to replace it"
        raise ConfigurationError(msg, details={"name": name})
    _REGISTRY[name] = factory


def available_strategies() -> list[str]:
    """Return the sorted names of every registered strategy."""
    return sorted(_REGISTRY)


def build_strategy(
    name: str,
    *,
    space: SearchSpace,
    seed: int,
    budget: TrainingBudget,
    max_evaluations: int,
    native_resolution: int | None = None,
    params: dict[str, Any] | None = None,
) -> SearchStrategy:
    """Construct a strategy by name.

    Args:
        name: Registry key.
        space: Search space.
        seed: Strategy seed.
        budget: Base training budget.
        max_evaluations: Cap on total evaluations.
        native_resolution: Dataset's native input resolution, needed by multi-fidelity
            strategies that scale resolution.
        params: Strategy-specific parameters from configuration.

    Returns:
        The constructed strategy.

    Raises:
        ConfigurationError: If the name is unknown or the parameters are invalid.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        msg = (
            f"unknown search strategy '{name}'; registered strategies are {available_strategies()}"
        )
        raise ConfigurationError(msg, details={"name": name, "available": available_strategies()})
    try:
        return factory(
            space=space,
            seed=seed,
            budget=budget,
            max_evaluations=max_evaluations,
            native_resolution=native_resolution,
            params=dict(params or {}),
        )
    except (TypeError, ValueError) as exc:
        msg = f"search strategy '{name}' rejected its parameters: {exc}"
        raise ConfigurationError(
            msg, details={"name": name, "params": sorted(params or {}), "error": str(exc)}
        ) from exc


__all__ = [
    "StrategyFactory",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
