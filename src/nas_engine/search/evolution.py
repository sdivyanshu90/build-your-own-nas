"""Regularized evolution, also known as aging evolution.

The algorithm
-------------
1. Initialise a population of ``P`` random architectures.
2. Repeat:

   a. Sample ``S`` distinct members of the population uniformly at random (a *tournament*).
   b. Take the best of those ``S`` as the parent.
   c. Mutate the parent to produce a child.
   d. Evaluate the child and append it to the population.
   e. Remove the **oldest** member of the population.

Everything interesting is in step (e).

Why remove the oldest, not the worst?
-------------------------------------
Classic (non-aging) evolution removes the worst member. That makes the population a
monotonically improving elite set — and creates a failure mode. Validation accuracy after
a short training run is a *noisy* estimate. Some architecture will eventually get a lucky
draw: favourable initial weights, a fortunate data order, a validation split that happens
to suit it. Under worst-removal, that lucky candidate is immortal. It stays in the
population forever, wins tournaments it does not deserve, and the search collapses onto
its neighbourhood. The search is then optimising *measurement noise*.

Aging removes the oldest member regardless of how good it is. Every architecture has a
bounded lifetime of exactly ``P`` subsequent evaluations. To persist in the gene pool a
lineage must keep producing children that score well — repeatedly, on independent training
runs. A one-off lucky measurement cannot do that. Aging therefore acts as an implicit
regulariser: it selects for architectures that are *reliably* good rather than
*once* lucky, and it maintains exploration because the population keeps turning over.

This is Real et al.'s "Regularized Evolution for Image Classifier Architecture Search"
(AAAI 2019), where aging evolution outperformed both non-aging evolution and RL under
equal compute.

The tournament size ``S`` sets the selection pressure. ``S = 1`` is a random walk;
``S = P`` always picks the current best and collapses diversity. Values around
``P / 4`` are a common compromise, and the default here follows that.

Failure handling
----------------
A candidate that failed to evaluate has no fitness, so it never enters the population.
If every initial candidate fails, the strategy falls back to random sampling rather than
deadlocking on an empty population — and logs that it did, because a fully failing
population means something is wrong with the configuration, not the search.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import CheckpointError, CheckpointVersionError, MutationError
from nas_engine.observability.logging import get_logger
from nas_engine.search.strategy import (
    Observation,
    Proposal,
    SearchStrategy,
    StrategyStatistics,
    deserialize_spec,
    serialize_spec,
)
from nas_engine.search_space.mutation import MutationOperator
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace
from nas_engine.utilities.seeding import derive_seed, rng_state_from_json, rng_state_to_json

_LOGGER = get_logger(__name__)

#: Version of this strategy's state payload.
EVOLUTION_STATE_VERSION: int = 1


@dataclass(frozen=True)
class PopulationMember:
    """One architecture currently in the population.

    Attributes:
        candidate_id: Engine-assigned candidate identifier.
        architecture_hash: Canonical hash.
        spec: The architecture.
        objective_value: Time-stable fitness; larger is better.
        generation: Generation index at which the member entered the population.
        parent_id: Candidate id of the parent, or ``None`` for founders.
    """

    candidate_id: str
    architecture_hash: str
    spec: ArchitectureSpec
    objective_value: float
    generation: int
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "candidate_id": self.candidate_id,
            "architecture_hash": self.architecture_hash,
            "spec": serialize_spec(self.spec),
            "objective_value": self.objective_value,
            "generation": self.generation,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PopulationMember:
        """Rebuild a member from :meth:`to_dict` output.

        Args:
            payload: Serialised member.

        Returns:
            The reconstructed member.
        """
        return cls(
            candidate_id=str(payload["candidate_id"]),
            architecture_hash=str(payload["architecture_hash"]),
            spec=deserialize_spec(payload["spec"]),
            objective_value=float(payload["objective_value"]),
            generation=int(payload.get("generation", 0)),
            parent_id=payload.get("parent_id"),
        )


class RegularizedEvolution(SearchStrategy):
    """Aging evolution over a fixed-size population.

    Args:
        space: Space to search.
        seed: Master seed; the sampler and mutation operator derive their own from it.
        max_evaluations: Total candidates to propose over the whole search.
        budget: Training budget applied to every candidate.
        population_size: Number of architectures kept alive.
        tournament_size: Members sampled per parent selection.
        allow_duplicate_children: Whether a mutation producing an already-seen
            architecture may still be proposed. Defaults to ``False``: re-evaluating a
            known architecture buys nothing.
        mutation_attempts: Mutation attempts per proposal before falling back to random
            sampling.

    Raises:
        ValueError: If population or tournament sizes are inconsistent.
    """

    name: ClassVar[str] = "regularized_evolution"
    requires_synchronous_observations: ClassVar[bool] = False

    def __init__(
        self,
        space: SearchSpace,
        *,
        seed: int,
        max_evaluations: int,
        budget: TrainingBudget,
        population_size: int = 16,
        tournament_size: int = 4,
        allow_duplicate_children: bool = False,
        mutation_attempts: int = 25,
    ) -> None:
        if max_evaluations < 1:
            msg = f"max_evaluations must be >= 1, received {max_evaluations}"
            raise ValueError(msg)
        if population_size < 2:
            msg = f"population_size must be >= 2, received {population_size}"
            raise ValueError(msg)
        if not 1 <= tournament_size <= population_size:
            msg = (
                f"tournament_size must lie in [1, population_size]; received "
                f"tournament_size={tournament_size}, population_size={population_size}"
            )
            raise ValueError(msg)

        self._space = space
        self._seed = seed
        self._max_evaluations = max_evaluations
        self._budget = budget
        self._population_size = population_size
        self._tournament_size = tournament_size
        self._allow_duplicate_children = allow_duplicate_children

        self._sampler = ArchitectureSampler(space, seed=derive_seed(seed, "evolution:sampler"))
        self._mutator = MutationOperator(
            space,
            seed=derive_seed(seed, "evolution:mutation"),
            max_attempts=mutation_attempts,
        )
        # Selection draws come from a third, independent stream so that changing the
        # mutation operator set cannot shift which parents get selected.
        self._rng = random.Random(  # noqa: S311 - reproducibility, not security
            derive_seed(seed, "evolution:selection")
        )

        self._population: deque[PopulationMember] = deque(maxlen=population_size)
        self._seen: set[str] = set()
        self._pending_parents: dict[str, str] = {}
        self._proposed = 0
        self._observed = 0
        self._succeeded = 0
        self._failed = 0
        self._duplicates = 0
        self._generation = 0
        self._random_fallbacks = 0
        self._mutation_failures = 0
        self._retired = 0

    # -- population -----------------------------------------------------------------
    @property
    def population(self) -> tuple[PopulationMember, ...]:
        """The current population, oldest first."""
        return tuple(self._population)

    def population_statistics(self) -> dict[str, float]:
        """Return summary statistics over the population's fitness values.

        Returns:
            Best, worst, mean, and spread of the population's objective values, plus its
            size and the number of distinct architectures it contains.
        """
        if not self._population:
            return {"size": 0.0, "unique": 0.0}
        values = [member.objective_value for member in self._population]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {
            "size": float(len(values)),
            "unique": float(len({member.architecture_hash for member in self._population})),
            "best": max(values),
            "worst": min(values),
            "mean": mean,
            "std": variance**0.5,
        }

    def _select_parent(self) -> PopulationMember:
        """Run one tournament and return the winner.

        Sampling is *without replacement* so that a tournament of size ``S`` really
        compares ``S`` distinct members. With replacement, the effective selection
        pressure would be lower than configured and would drift with population size.

        Returns:
            The fittest sampled member.

        Raises:
            RuntimeError: If the population is empty; callers must check first.
        """
        if not self._population:  # pragma: no cover - guarded by callers
            msg = "cannot run a tournament on an empty population"
            raise RuntimeError(msg)
        size = min(self._tournament_size, len(self._population))
        entrants = self._rng.sample(list(self._population), size)
        # Ties are broken by architecture hash so the winner is deterministic.
        return max(entrants, key=lambda member: (member.objective_value, member.architecture_hash))

    # -- strategy interface ---------------------------------------------------------
    def propose(self, count: int) -> list[Proposal]:
        """Return up to ``count`` proposals.

        During initialisation these are random draws; afterwards they are mutations of
        tournament winners.

        Args:
            count: Maximum proposals wanted.

        Returns:
            The proposals.
        """
        remaining = self._max_evaluations - self._proposed
        wanted = max(0, min(count, remaining))
        proposals: list[Proposal] = []

        for _ in range(wanted):
            proposal = self._propose_one()
            if proposal is None:
                break
            proposals.append(proposal)
        return proposals

    def _propose_one(self) -> Proposal | None:  # noqa: PLR0911 - one exit per fallback
        """Produce a single proposal, or ``None`` when none can be made.

        Returns:
            A proposal, or ``None``.
        """
        initialising = self._proposed < self._population_size
        if initialising or not self._population:
            if not initialising and not self._population:
                # Every observation so far failed or was unscoreable. Falling back to
                # random sampling keeps the search alive; the log line makes the
                # degradation visible rather than silent.
                self._random_fallbacks += 1
                _LOGGER.warning(
                    "evolution.empty_population",
                    proposed=self._proposed,
                    observed=self._observed,
                    failed=self._failed,
                    action="falling back to random sampling",
                )
            spec = self._sampler.sample_unique(self._seen)
            if spec is None:
                return None
            self._register(spec)
            return Proposal(spec=spec, budget=self._budget, origin="random")

        parent = self._select_parent()
        try:
            result = self._mutator.mutate(parent.spec)
        except MutationError:
            self._mutation_failures += 1
            spec = self._sampler.sample_unique(self._seen)
            if spec is None:
                return None
            self._random_fallbacks += 1
            self._register(spec)
            return Proposal(spec=spec, budget=self._budget, origin="random_fallback")

        child_hash = architecture_hash(result.child)
        if child_hash in self._seen and not self._allow_duplicate_children:
            # One retry with a fresh mutation, then fall back to random. Looping until a
            # novel child appears would stall on a saturated neighbourhood.
            retry = self._mutator.try_mutate(parent.spec)
            if retry is None or architecture_hash(retry.child) in self._seen:
                self._duplicates += 1
                spec = self._sampler.sample_unique(self._seen)
                if spec is None:
                    return None
                self._random_fallbacks += 1
                self._register(spec)
                return Proposal(spec=spec, budget=self._budget, origin="random_fallback")
            result = retry

        self._register(result.child)
        self._pending_parents[architecture_hash(result.child)] = parent.candidate_id
        return Proposal(
            spec=result.child,
            budget=self._budget,
            parent_id=parent.candidate_id,
            mutation=result.description,
            origin="mutation",
            metadata={
                "operator": result.operator,
                "parent_hash": result.parent_hash,
                "generation": self._generation,
            },
        )

    def _register(self, spec: ArchitectureSpec) -> None:
        """Record a proposal's hash and increment the proposal counter."""
        self._seen.add(architecture_hash(spec))
        self._proposed += 1

    def observe(self, observation: Observation) -> None:
        """Record an evaluation and update the population.

        Args:
            observation: The evaluation outcome.
        """
        self._observed += 1
        self._seen.add(observation.architecture_hash)
        self._pending_parents.pop(observation.architecture_hash, None)

        if not observation.succeeded or observation.objective_value is None:
            self._failed += 1
            _LOGGER.debug(
                "evolution.observation_skipped",
                candidate_id=observation.candidate_id,
                architecture_hash=observation.architecture_hash,
                reason="failed" if not observation.succeeded else "unscoreable",
            )
            return

        self._succeeded += 1
        self._generation += 1
        member = PopulationMember(
            candidate_id=observation.candidate_id,
            architecture_hash=observation.architecture_hash,
            spec=observation.spec,
            objective_value=float(observation.objective_value),
            generation=self._generation,
            parent_id=observation.parent_id,
        )
        # `deque(maxlen=P).append` discards from the left automatically, which is exactly
        # the aging rule: the oldest member leaves regardless of its fitness.
        was_full = len(self._population) == self._population_size
        self._population.append(member)
        if was_full:
            self._retired += 1

        _LOGGER.debug(
            "evolution.population_updated",
            candidate_id=observation.candidate_id,
            objective_value=observation.objective_value,
            population_size=len(self._population),
            generation=self._generation,
        )

    def is_finished(self) -> bool:
        """Report whether the evaluation budget is spent."""
        return self._proposed >= self._max_evaluations

    def on_duplicate(self, architecture_hash: str) -> None:
        """Record a duplicate rejected by the engine."""
        self._duplicates += 1
        self._seen.add(architecture_hash)

    def statistics(self) -> StrategyStatistics:
        """Return counters for reporting."""
        return StrategyStatistics(
            proposed=self._proposed,
            observed=self._observed,
            succeeded=self._succeeded,
            failed=self._failed,
            duplicates_avoided=self._duplicates,
            extra={
                "population": self.population_statistics(),
                "population_size": self._population_size,
                "tournament_size": self._tournament_size,
                "generation": self._generation,
                "retired": self._retired,
                "random_fallbacks": self._random_fallbacks,
                "mutation_failures": self._mutation_failures,
                "unique_architectures": len(self._seen),
                "mutation_operators": self._mutator.statistics.to_dict(),
            },
        )

    def describe(self) -> str:
        """Return a human-readable configuration summary."""
        return (
            f"regularized_evolution: population {self._population_size}, tournament "
            f"{self._tournament_size}, {self._max_evaluations} evaluations at "
            f"{self._budget.describe()}"
        )

    # -- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the strategy state."""
        return {
            "version": EVOLUTION_STATE_VERSION,
            "strategy": self.name,
            "seed": self._seed,
            "max_evaluations": self._max_evaluations,
            "population_size": self._population_size,
            "tournament_size": self._tournament_size,
            "population": [member.to_dict() for member in self._population],
            "seen": sorted(self._seen),
            "pending_parents": dict(self._pending_parents),
            "counters": {
                "proposed": self._proposed,
                "observed": self._observed,
                "succeeded": self._succeeded,
                "failed": self._failed,
                "duplicates": self._duplicates,
                "generation": self._generation,
                "random_fallbacks": self._random_fallbacks,
                "mutation_failures": self._mutation_failures,
                "retired": self._retired,
            },
            "sampler": self._sampler.state_dict(),
            "mutator": self._mutator.state_dict(),
            "selection_rng": rng_state_to_json(self._rng),
            "budget": self._budget.to_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore state captured by :meth:`state_dict`.

        Args:
            payload: Previously captured state.

        Raises:
            CheckpointVersionError: If the payload version is unsupported.
            CheckpointError: If the payload is malformed or the population size changed.
        """
        version = payload.get("version")
        if version != EVOLUTION_STATE_VERSION:
            msg = (
                f"regularized_evolution state version {version} is not supported by this "
                f"build (expected {EVOLUTION_STATE_VERSION})"
            )
            raise CheckpointVersionError(msg, details={"version": version})

        stored_population_size = int(payload.get("population_size", self._population_size))
        if stored_population_size != self._population_size:
            msg = (
                f"checkpoint was written with population_size={stored_population_size} but "
                f"the current configuration uses {self._population_size}; resuming would "
                "change the aging schedule and invalidate the comparison"
            )
            raise CheckpointError(
                msg,
                details={
                    "checkpoint": stored_population_size,
                    "configured": self._population_size,
                },
            )

        try:
            members = [PopulationMember.from_dict(entry) for entry in payload["population"]]
            self._population = deque(members, maxlen=self._population_size)
            self._seen = set(payload.get("seen", []))
            self._pending_parents = dict(payload.get("pending_parents", {}))
            counters = payload.get("counters", {})
            self._proposed = int(counters.get("proposed", 0))
            self._observed = int(counters.get("observed", 0))
            self._succeeded = int(counters.get("succeeded", 0))
            self._failed = int(counters.get("failed", 0))
            self._duplicates = int(counters.get("duplicates", 0))
            self._generation = int(counters.get("generation", 0))
            self._random_fallbacks = int(counters.get("random_fallbacks", 0))
            self._mutation_failures = int(counters.get("mutation_failures", 0))
            self._retired = int(counters.get("retired", 0))
            self._sampler.load_state_dict(payload["sampler"])
            self._mutator.load_state_dict(payload["mutator"])
            self._rng = rng_state_from_json(payload["selection_rng"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"regularized_evolution state payload is malformed: {exc}"
            raise CheckpointError(msg, details={"error": str(exc)}) from exc


__all__ = ["EVOLUTION_STATE_VERSION", "PopulationMember", "RegularizedEvolution"]
