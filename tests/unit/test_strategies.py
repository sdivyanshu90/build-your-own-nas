"""Unit tests for the search strategies.

Covers: the strategy contract, random search's reproducibility and exhaustion handling,
regularized evolution's aging rule and tournament selection, successive halving's ladder
arithmetic and promotion barrier, checkpoint round-trips, and the registry.
"""

from __future__ import annotations

from typing import Any

import pytest

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.result import EvaluationFailure, EvaluationResult
from nas_engine.exceptions import (
    CheckpointError,
    CheckpointVersionError,
    ConfigurationError,
    TrainingError,
)
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
    deserialize_spec,
    serialize_spec,
)
from nas_engine.search.successive_halving import ResourceLadder, SuccessiveHalving
from nas_engine.search_space.space import SearchSpace

pytestmark = pytest.mark.unit

BUDGET = TrainingBudget(epochs=1)


def _observation(
    proposal: Proposal, *, value: float | None = 0.5, succeeded: bool = True
) -> Observation:
    """Build an observation for a proposal without running an evaluation."""
    digest = architecture_hash(proposal.spec)
    result = EvaluationResult(
        candidate_id=digest,
        architecture_hash=digest,
        budget=proposal.budget,
        metrics={"validation_accuracy": value} if value is not None else {},
        succeeded=succeeded,
        failure=None if succeeded else EvaluationFailure.from_exception(TrainingError("x")),
    )
    return Observation(
        candidate_id=digest,
        architecture_hash=digest,
        spec=proposal.spec,
        result=result,
        objective_value=value if succeeded else None,
        parent_id=proposal.parent_id,
    )


class TestStrategyContract:
    def test_specification_serialisation_round_trips(self, sample_spec: ArchitectureSpec) -> None:
        assert deserialize_spec(serialize_spec(sample_spec)) == sample_spec

    def test_statistics_serialise_with_extras(self) -> None:
        payload = StrategyStatistics(proposed=3, extra={"custom": 1}).to_dict()
        assert payload["proposed"] == 3
        assert payload["custom"] == 1

    def test_default_hooks_are_no_ops(self, sample_spec: ArchitectureSpec) -> None:
        class Minimal(SearchStrategy):
            name = "minimal"

            def propose(self, count: int) -> list[Proposal]:
                return []

            def observe(self, observation: Observation) -> None:
                return None

            def is_finished(self) -> bool:
                return True

            def state_dict(self) -> dict[str, Any]:
                return {}

            def load_state_dict(self, payload: dict[str, Any]) -> None:
                return None

            def statistics(self) -> StrategyStatistics:
                return StrategyStatistics()

        strategy = Minimal()
        strategy.on_duplicate("hash")
        strategy.on_rejected(sample_spec, "reason")
        assert "minimal" in strategy.describe()


class TestRandomSearch:
    def test_proposes_up_to_the_budget(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=1, max_evaluations=5, budget=BUDGET)
        proposals = strategy.propose(10)
        assert len(proposals) == 5
        assert strategy.is_finished()

    def test_proposals_are_unique(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=2, max_evaluations=20, budget=BUDGET)
        hashes = [architecture_hash(p.spec) for p in strategy.propose(20)]
        assert len(set(hashes)) == len(hashes)

    def test_is_reproducible(self, tiny_space: SearchSpace) -> None:
        def run() -> list[str]:
            strategy = RandomSearch(tiny_space, seed=3, max_evaluations=6, budget=BUDGET)
            return [architecture_hash(p.spec) for p in strategy.propose(6)]

        assert run() == run()

    def test_records_observations(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=4, max_evaluations=3, budget=BUDGET)
        for proposal in strategy.propose(3):
            strategy.observe(_observation(proposal))
        statistics = strategy.statistics()
        assert statistics.observed == 3
        assert statistics.succeeded == 3

    def test_counts_failures_separately(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=5, max_evaluations=2, budget=BUDGET)
        for proposal in strategy.propose(2):
            strategy.observe(_observation(proposal, succeeded=False, value=None))
        assert strategy.statistics().failed == 2

    def test_exhausted_space_stops_the_search(self, micro_space: SearchSpace) -> None:
        strategy = RandomSearch(
            micro_space,
            seed=6,
            max_evaluations=100,
            budget=BUDGET,
            sample_attempts=40,
            max_consecutive_exhaustions=1,
        )
        collected: list[Proposal] = []
        for _ in range(10):
            batch = strategy.propose(5)
            collected.extend(batch)
            if strategy.is_finished():
                break
        assert strategy.is_finished()
        assert len(collected) < 100
        assert strategy.statistics().extra["exhausted"] is True

    def test_duplicates_reported_by_the_engine_are_recorded(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=7, max_evaluations=3, budget=BUDGET)
        strategy.on_duplicate("deadbeef")
        assert strategy.statistics().duplicates_avoided == 1

    def test_state_round_trip_continues_the_stream(self, tiny_space: SearchSpace) -> None:
        original = RandomSearch(tiny_space, seed=8, max_evaluations=10, budget=BUDGET)
        original.propose(3)
        state = original.state_dict()
        expected = [architecture_hash(p.spec) for p in original.propose(3)]

        restored = RandomSearch(tiny_space, seed=999, max_evaluations=10, budget=BUDGET)
        restored.load_state_dict(state)
        assert [architecture_hash(p.spec) for p in restored.propose(3)] == expected

    def test_rejects_a_future_state_version(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=9, max_evaluations=1, budget=BUDGET)
        with pytest.raises(CheckpointVersionError, match="state version"):
            strategy.load_state_dict({"version": 99})

    def test_rejects_malformed_state(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=9, max_evaluations=1, budget=BUDGET)
        with pytest.raises(CheckpointError, match="malformed"):
            strategy.load_state_dict({"version": 1})

    def test_budget_must_be_positive(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ValueError, match="max_evaluations"):
            RandomSearch(tiny_space, seed=1, max_evaluations=0, budget=BUDGET)

    def test_describes_itself(self, tiny_space: SearchSpace) -> None:
        strategy = RandomSearch(tiny_space, seed=1, max_evaluations=4, budget=BUDGET)
        assert "random_search" in strategy.describe()


class TestRegularizedEvolution:
    @staticmethod
    def _strategy(space: SearchSpace, **kwargs: Any) -> RegularizedEvolution:
        defaults: dict[str, Any] = {
            "seed": 11,
            "max_evaluations": 30,
            "budget": BUDGET,
            "population_size": 4,
            "tournament_size": 2,
        }
        defaults.update(kwargs)
        return RegularizedEvolution(space, **defaults)

    def test_initial_proposals_are_random(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        proposals = strategy.propose(4)
        assert all(proposal.origin == "random" for proposal in proposals)
        assert all(proposal.parent_id is None for proposal in proposals)

    def test_switches_to_mutation_once_the_population_is_seeded(
        self, tiny_space: SearchSpace
    ) -> None:
        strategy = self._strategy(tiny_space)
        for proposal in strategy.propose(4):
            strategy.observe(_observation(proposal))
        child = strategy.propose(1)[0]
        assert child.origin in {"mutation", "random_fallback"}
        if child.origin == "mutation":
            assert child.parent_id is not None
            assert child.mutation

    def test_population_is_capped(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space, population_size=3)
        for index in range(9):
            for proposal in strategy.propose(1):
                strategy.observe(_observation(proposal, value=0.1 * index))
        assert len(strategy.population) == 3

    def test_aging_removes_the_oldest_not_the_worst(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space, population_size=2)
        proposals = strategy.propose(2)
        # The first candidate is by far the best; aging must still evict it once two more
        # candidates arrive, which is the entire point of the algorithm.
        strategy.observe(_observation(proposals[0], value=10.0))
        strategy.observe(_observation(proposals[1], value=0.1))
        best_hash = architecture_hash(proposals[0].spec)
        assert best_hash in {member.architecture_hash for member in strategy.population}

        for _ in range(2):
            for proposal in strategy.propose(1):
                strategy.observe(_observation(proposal, value=0.2))
        assert best_hash not in {member.architecture_hash for member in strategy.population}

    def test_failed_candidates_never_enter_the_population(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        for proposal in strategy.propose(4):
            strategy.observe(_observation(proposal, succeeded=False, value=None))
        assert strategy.population == ()
        assert strategy.statistics().failed == 4

    def test_empty_population_falls_back_to_random_sampling(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space, population_size=2)
        for proposal in strategy.propose(2):
            strategy.observe(_observation(proposal, succeeded=False, value=None))
        proposal = strategy.propose(1)[0]
        assert proposal.origin == "random"
        assert strategy.statistics().extra["random_fallbacks"] >= 1

    def test_tournament_prefers_fitter_parents(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space, population_size=4, tournament_size=4)
        proposals = strategy.propose(4)
        values = [0.1, 0.2, 0.3, 0.9]
        for proposal, value in zip(proposals, values, strict=True):
            strategy.observe(_observation(proposal, value=value))
        best_id = architecture_hash(proposals[3].spec)
        # A tournament covering the whole population always selects the fittest member.
        # Some draws fall back to random sampling when every neighbour of that parent has
        # already been seen, and those carry no parent; only the mutations are asserted on.
        children = [strategy.propose(1)[0] for _ in range(6)]
        mutations = [child for child in children if child.origin == "mutation"]
        assert mutations, "expected at least one mutation among six proposals"
        assert {child.parent_id for child in mutations} == {best_id}

    def test_population_statistics_are_reported(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        for index, proposal in enumerate(strategy.propose(4)):
            strategy.observe(_observation(proposal, value=0.1 * (index + 1)))
        statistics = strategy.statistics().extra["population"]
        assert statistics["size"] == 4
        assert statistics["best"] == pytest.approx(0.4)
        assert statistics["worst"] == pytest.approx(0.1)

    def test_empty_population_statistics_are_safe(self, tiny_space: SearchSpace) -> None:
        assert self._strategy(tiny_space).population_statistics() == {"size": 0.0, "unique": 0.0}

    def test_state_round_trip_preserves_the_population(self, tiny_space: SearchSpace) -> None:
        original = self._strategy(tiny_space)
        for proposal in original.propose(4):
            original.observe(_observation(proposal))
        state = original.state_dict()

        restored = self._strategy(tiny_space, seed=999)
        restored.load_state_dict(state)
        assert [member.architecture_hash for member in restored.population] == [
            member.architecture_hash for member in original.population
        ]

    def test_state_round_trip_continues_the_stream(self, tiny_space: SearchSpace) -> None:
        original = self._strategy(tiny_space)
        for proposal in original.propose(4):
            original.observe(_observation(proposal))
        state = original.state_dict()
        expected = architecture_hash(original.propose(1)[0].spec)

        restored = self._strategy(tiny_space, seed=999)
        restored.load_state_dict(state)
        assert architecture_hash(restored.propose(1)[0].spec) == expected

    def test_changing_population_size_across_a_resume_is_rejected(
        self, tiny_space: SearchSpace
    ) -> None:
        original = self._strategy(tiny_space, population_size=4)
        state = original.state_dict()
        restored = self._strategy(tiny_space, population_size=8)
        with pytest.raises(CheckpointError, match="population_size"):
            restored.load_state_dict(state)

    def test_rejects_a_future_state_version(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(CheckpointVersionError, match="state version"):
            self._strategy(tiny_space).load_state_dict({"version": 99})

    def test_configuration_is_validated(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ValueError, match="population_size"):
            self._strategy(tiny_space, population_size=1)
        with pytest.raises(ValueError, match="tournament_size"):
            self._strategy(tiny_space, population_size=4, tournament_size=9)
        with pytest.raises(ValueError, match="max_evaluations"):
            self._strategy(tiny_space, max_evaluations=0)

    def test_population_member_round_trips(self, sample_spec: ArchitectureSpec) -> None:
        member = PopulationMember(
            candidate_id="c",
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
            objective_value=0.5,
            generation=3,
        )
        assert PopulationMember.from_dict(member.to_dict()) == member

    def test_describes_itself(self, tiny_space: SearchSpace) -> None:
        assert "population" in self._strategy(tiny_space).describe()


class TestResourceLadder:
    def test_epochs_grow_geometrically(self) -> None:
        ladder = ResourceLadder(base_budget=BUDGET, num_rungs=3, reduction_factor=3.0)
        assert [budget.epochs for budget in ladder.budgets()] == [1, 3, 9]

    def test_rung_sizes_shrink_geometrically(self) -> None:
        ladder = ResourceLadder(base_budget=BUDGET, num_rungs=3, reduction_factor=3.0)
        assert ladder.rung_sizes(9) == (9, 3, 1)

    def test_rung_sizes_never_reach_zero(self) -> None:
        ladder = ResourceLadder(base_budget=BUDGET, num_rungs=4, reduction_factor=3.0)
        assert all(size >= 1 for size in ladder.rung_sizes(2))

    def test_each_rung_costs_about_the_same(self) -> None:
        ladder = ResourceLadder(base_budget=BUDGET, num_rungs=3, reduction_factor=3.0)
        costs = [
            size * budget.epochs
            for size, budget in zip(ladder.rung_sizes(9), ladder.budgets(), strict=True)
        ]
        assert costs == [9, 9, 9]

    def test_total_evaluations_sums_the_rungs(self) -> None:
        ladder = ResourceLadder(base_budget=BUDGET, num_rungs=3, reduction_factor=3.0)
        assert ladder.total_evaluations(9) == 13

    def test_data_fraction_scaling_caps_at_one(self) -> None:
        ladder = ResourceLadder(
            base_budget=TrainingBudget(epochs=1, train_fraction=0.5),
            num_rungs=3,
            reduction_factor=3.0,
            scale_epochs=False,
            scale_train_fraction=True,
        )
        assert [budget.train_fraction for budget in ladder.budgets()] == [0.5, 1.0, 1.0]

    def test_resolution_scaling_caps_at_native(self) -> None:
        ladder = ResourceLadder(
            base_budget=TrainingBudget(epochs=1, resolution=8),
            num_rungs=3,
            reduction_factor=2.0,
            scale_epochs=False,
            scale_resolution=True,
            native_resolution=16,
        )
        assert [budget.resolution for budget in ladder.budgets()] == [8, None, None]

    def test_configuration_is_validated(self) -> None:
        with pytest.raises(ConfigurationError, match="num_rungs"):
            ResourceLadder(base_budget=BUDGET, num_rungs=0)
        with pytest.raises(ConfigurationError, match="reduction_factor"):
            ResourceLadder(base_budget=BUDGET, reduction_factor=1.0)
        with pytest.raises(ConfigurationError, match="at least one resource dimension"):
            ResourceLadder(base_budget=BUDGET, num_rungs=2, scale_epochs=False)
        with pytest.raises(ConfigurationError, match="native_resolution"):
            ResourceLadder(base_budget=BUDGET, scale_resolution=True)


class TestSuccessiveHalving:
    @staticmethod
    def _strategy(space: SearchSpace, **kwargs: Any) -> SuccessiveHalving:
        ladder = kwargs.pop(
            "ladder",
            ResourceLadder(base_budget=BUDGET, num_rungs=3, reduction_factor=2.0),
        )
        return SuccessiveHalving(
            space,
            seed=kwargs.pop("seed", 21),
            ladder=ladder,
            initial_candidates=kwargs.pop("initial_candidates", 4),
        )

    def test_first_rung_proposes_random_candidates(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        proposals = strategy.propose(10)
        assert len(proposals) == 4
        assert all(proposal.origin == "random" for proposal in proposals)
        assert all(proposal.budget.rung == 0 for proposal in proposals)

    def test_promotion_waits_for_the_whole_rung(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        proposals = strategy.propose(4)
        strategy.observe(_observation(proposals[0], value=0.9))
        # Three results are still outstanding, so the barrier must hold.
        assert strategy.propose(4) == []
        for proposal in proposals[1:]:
            strategy.observe(_observation(proposal, value=0.1))
        promoted = strategy.propose(4)
        assert promoted
        assert all(item.origin == "promotion" for item in promoted)

    def test_promotes_the_best_candidates(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        proposals = strategy.propose(4)
        values = [0.1, 0.9, 0.2, 0.8]
        for proposal, value in zip(proposals, values, strict=True):
            strategy.observe(_observation(proposal, value=value))
        promoted = strategy.propose(4)
        promoted_hashes = {architecture_hash(item.spec) for item in promoted}
        expected = {
            architecture_hash(proposals[1].spec),
            architecture_hash(proposals[3].spec),
        }
        assert promoted_hashes == expected

    def test_later_rungs_use_larger_budgets(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        proposals = strategy.propose(4)
        for proposal in proposals:
            strategy.observe(_observation(proposal))
        promoted = strategy.propose(4)
        assert all(item.budget.epochs == 2 for item in promoted)
        assert all(item.budget.rung == 1 for item in promoted)

    def test_bracket_completes(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        for _ in range(20):
            proposals = strategy.propose(4)
            if not proposals and strategy.is_finished():
                break
            for proposal in proposals:
                strategy.observe(_observation(proposal))
        assert strategy.is_finished()
        summary = strategy.rung_summary()
        assert summary[-1]["completed"] >= 1

    def test_all_failures_end_the_bracket(self, tiny_space: SearchSpace) -> None:
        strategy = self._strategy(tiny_space)
        for proposal in strategy.propose(4):
            strategy.observe(_observation(proposal, succeeded=False, value=None))
        assert strategy.propose(4) == []
        assert strategy.is_finished()

    def test_state_round_trip_preserves_rung_progress(self, tiny_space: SearchSpace) -> None:
        original = self._strategy(tiny_space)
        for proposal in original.propose(4):
            original.observe(_observation(proposal))
        state = original.state_dict()

        restored = self._strategy(tiny_space, seed=999)
        restored.load_state_dict(state)
        assert restored.current_rung == original.current_rung
        assert restored.rung_summary() == original.rung_summary()

    def test_changing_the_ladder_across_a_resume_is_rejected(self, tiny_space: SearchSpace) -> None:
        original = self._strategy(tiny_space)
        state = original.state_dict()
        restored = self._strategy(
            tiny_space,
            ladder=ResourceLadder(base_budget=BUDGET, num_rungs=2, reduction_factor=2.0),
        )
        with pytest.raises(CheckpointError, match="rungs"):
            restored.load_state_dict(state)

    def test_rejects_a_future_state_version(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(CheckpointVersionError, match="state version"):
            self._strategy(tiny_space).load_state_dict({"version": 99})

    def test_initial_candidates_must_be_positive(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ValueError, match="initial_candidates"):
            self._strategy(tiny_space, initial_candidates=0)

    def test_describes_its_ladder(self, tiny_space: SearchSpace) -> None:
        assert "eta=2" in self._strategy(tiny_space).describe()


class TestRegistry:
    def test_built_in_strategies_are_registered(self) -> None:
        assert set(available_strategies()) >= {
            "random_search",
            "regularized_evolution",
            "successive_halving",
        }

    @pytest.mark.parametrize(
        "name", ["random_search", "regularized_evolution", "successive_halving"]
    )
    def test_every_strategy_builds_from_the_registry(
        self, name: str, tiny_space: SearchSpace
    ) -> None:
        strategy = build_strategy(
            name,
            space=tiny_space,
            seed=1,
            budget=BUDGET,
            max_evaluations=8,
            native_resolution=16,
        )
        assert strategy.name == name
        assert isinstance(strategy.statistics(), StrategyStatistics)

    def test_unknown_strategy_is_reported(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ConfigurationError, match="unknown search strategy"):
            build_strategy("nope", space=tiny_space, seed=1, budget=BUDGET, max_evaluations=1)

    def test_bad_parameters_are_reported(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ConfigurationError, match="rejected its parameters"):
            build_strategy(
                "regularized_evolution",
                space=tiny_space,
                seed=1,
                budget=BUDGET,
                max_evaluations=4,
                params={"population_size": 1},
            )

    def test_successive_halving_sizes_itself_to_the_budget(self, tiny_space: SearchSpace) -> None:
        strategy = build_strategy(
            "successive_halving",
            space=tiny_space,
            seed=1,
            budget=BUDGET,
            max_evaluations=13,
            native_resolution=16,
            params={"num_rungs": 3, "reduction_factor": 3.0},
        )
        assert isinstance(strategy, SuccessiveHalving)
        assert sum(strategy.rung_sizes) <= 14

    def test_custom_strategies_can_be_registered(self, tiny_space: SearchSpace) -> None:
        def factory(**_: Any) -> SearchStrategy:
            return RandomSearch(tiny_space, seed=1, max_evaluations=1, budget=BUDGET)

        register_strategy("custom-test-strategy", factory, overwrite=True)
        assert "custom-test-strategy" in available_strategies()
        with pytest.raises(ConfigurationError, match="already registered"):
            register_strategy("custom-test-strategy", factory)
