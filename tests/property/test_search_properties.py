"""Property-based tests for sampling, mutation, objectives, and state transitions.

Invariants covered:

* every sampled architecture is a valid member of its space;
* mutation stays inside the space and never modifies its parent;
* repair is idempotent and produces valid architectures;
* Pareto-front members are never dominated, and dominance is a strict partial order;
* weighted scores stay bounded when every objective is bounded;
* checkpoint round-trips preserve strategy state;
* the state machine rejects every transition outside its table.
"""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import InvalidStateTransitionError
from nas_engine.objectives.objective import (
    NormalizationStrategy,
    Objective,
    ObjectiveDirection,
    ObjectiveSet,
)
from nas_engine.objectives.pareto import (
    ObjectiveVector,
    dominates,
    non_dominated_sort,
    pareto_front,
)
from nas_engine.objectives.ranking import rank_candidates
from nas_engine.objectives.scoring import WeightedScorer
from nas_engine.orchestration.lifecycle import (
    ALLOWED_TRANSITIONS,
    CandidateState,
    can_transition,
    validate_transition,
)
from nas_engine.search.random_search import RandomSearch
from nas_engine.search_space.mutation import MutationOperator
from nas_engine.search_space.presets import default_cnn_space, tiny_cnn_space
from nas_engine.search_space.repair import repair_architecture
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace
from nas_engine.search_space.validation import check_architecture
from tests.profiles import scaled

pytestmark = [pytest.mark.property]

SETTINGS = settings(
    max_examples=scaled(40),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

seeds = st.integers(min_value=0, max_value=2**16)


def _space(name: str) -> SearchSpace:
    """Return a preset space by name without a fixture, so Hypothesis can call it freely."""
    return tiny_cnn_space() if name == "tiny" else default_cnn_space()


class TestSampling:
    @given(seed=seeds, space_name=st.sampled_from(["tiny", "default"]))
    @SETTINGS
    def test_every_sample_is_a_valid_member(self, seed: int, space_name: str) -> None:
        space = _space(space_name)
        spec = ArchitectureSampler(space, seed=seed).sample()
        assert check_architecture(spec, space).is_valid

    @given(seed=seeds)
    @SETTINGS
    def test_sampling_is_a_pure_function_of_the_seed(self, seed: int) -> None:
        space = tiny_cnn_space()
        first = architecture_hash(ArchitectureSampler(space, seed=seed).sample())
        second = architecture_hash(ArchitectureSampler(space, seed=seed).sample())
        assert first == second

    @given(seed=seeds, count=st.integers(min_value=1, max_value=8))
    @SETTINGS
    def test_unique_sampling_never_repeats(self, seed: int, count: int) -> None:
        space = default_cnn_space()
        sampler = ArchitectureSampler(space, seed=seed)
        seen: set[str] = set()
        for _ in range(count):
            spec = sampler.sample_unique(seen)
            if spec is None:
                break
            digest = architecture_hash(spec)
            assert digest not in seen
            seen.add(digest)

    @given(seed=seeds)
    @SETTINGS
    def test_stage_widths_are_non_decreasing_when_required(self, seed: int) -> None:
        from nas_engine.search_space.repair import stage_widths

        space = default_cnn_space()
        widths = stage_widths(ArchitectureSampler(space, seed=seed).sample())
        assert list(widths) == sorted(widths)


class TestMutation:
    @given(seed=seeds)
    @SETTINGS
    def test_children_stay_inside_the_space(self, seed: int) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=seed).sample()
        child = MutationOperator(space, seed=seed + 1).mutate(parent).child
        assert check_architecture(child, space).is_valid

    @given(seed=seeds)
    @SETTINGS
    def test_the_parent_is_never_modified(self, seed: int) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=seed).sample()
        before = architecture_hash(parent)
        MutationOperator(space, seed=seed + 1).mutate(parent)
        assert architecture_hash(parent) == before

    @given(seed=seeds)
    @SETTINGS
    def test_children_differ_from_their_parent(self, seed: int) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=seed).sample()
        result = MutationOperator(space, seed=seed + 1).mutate(parent)
        assert architecture_hash(result.child) != result.parent_hash

    @given(seed=seeds, steps=st.integers(min_value=1, max_value=5))
    @SETTINGS
    def test_repeated_mutation_stays_valid(self, seed: int, steps: int) -> None:
        space = default_cnn_space()
        mutator = MutationOperator(space, seed=seed + 1)
        current = ArchitectureSampler(space, seed=seed).sample()
        for _ in range(steps):
            current = mutator.mutate(current).child
            assert check_architecture(current, space).is_valid

    @given(seed=seeds)
    @SETTINGS
    def test_mutation_is_a_pure_function_of_its_seed(self, seed: int) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=seed).sample()
        first = MutationOperator(space, seed=7).mutate(parent)
        second = MutationOperator(space, seed=7).mutate(parent)
        assert architecture_hash(first.child) == architecture_hash(second.child)


class TestRepair:
    @given(seed=seeds)
    @SETTINGS
    def test_repair_is_idempotent(self, seed: int) -> None:
        space = default_cnn_space()
        spec = ArchitectureSampler(space, seed=seed).sample()
        once, _ = repair_architecture(spec)
        twice, report = repair_architecture(once)
        assert architecture_hash(once) == architecture_hash(twice)
        assert not report.changed

    @given(seed=seeds)
    @SETTINGS
    def test_repairing_a_valid_architecture_changes_nothing(self, seed: int) -> None:
        space = default_cnn_space()
        spec = ArchitectureSampler(space, seed=seed).sample()
        repaired, report = repair_architecture(spec)
        assert repaired is spec
        assert not report.changed

    @given(seed=seeds, data=st.data())
    @SETTINGS
    def test_forced_widths_produce_buildable_architectures(
        self, seed: int, data: st.DataObject
    ) -> None:
        from nas_engine.architectures.shapes import infer_shapes

        space = default_cnn_space()
        spec = ArchitectureSampler(space, seed=seed).sample()
        # Draw the widths *after* sampling, so their length matches the stage count by
        # construction. Drawing first and filtering with assume would discard most
        # examples, since the sampler picks its own stage count.
        widths = data.draw(
            st.lists(
                st.sampled_from([16, 32, 64]),
                min_size=spec.num_stages,
                max_size=spec.num_stages,
            )
        )
        repaired, _ = repair_architecture(spec, target_widths=tuple(widths))
        infer_shapes(repaired)


@st.composite
def objective_vectors(draw: st.DrawFn) -> list[ObjectiveVector]:
    """Draw a small population of two-dimensional objective vectors."""
    count = draw(st.integers(min_value=1, max_value=8))
    values = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            ),
            min_size=count,
            max_size=count,
        )
    )
    return [ObjectiveVector(f"c{index}", pair, pair) for index, pair in enumerate(values)]


class TestParetoProperties:
    @given(vectors=objective_vectors())
    @SETTINGS
    def test_front_members_are_never_dominated(self, vectors: list[ObjectiveVector]) -> None:
        front = pareto_front(vectors)
        for member in front:
            assert not any(
                dominates(other.values, member.values)
                for other in vectors
                if other.candidate_id != member.candidate_id
            )

    @given(vectors=objective_vectors())
    @SETTINGS
    def test_the_front_is_never_empty_for_a_non_empty_population(
        self, vectors: list[ObjectiveVector]
    ) -> None:
        assert pareto_front(vectors)

    @given(vectors=objective_vectors())
    @SETTINGS
    def test_non_dominated_sort_partitions_the_population(
        self, vectors: list[ObjectiveVector]
    ) -> None:
        fronts = non_dominated_sort(vectors)
        assigned = [vector.candidate_id for front in fronts for vector in front]
        assert sorted(assigned) == sorted(vector.candidate_id for vector in vectors)

    @given(vectors=objective_vectors())
    @SETTINGS
    def test_front_computation_is_order_independent(self, vectors: list[ObjectiveVector]) -> None:
        forward = [vector.candidate_id for vector in pareto_front(vectors)]
        backward = [vector.candidate_id for vector in pareto_front(list(reversed(vectors)))]
        assert forward == backward

    @given(
        left=st.tuples(st.floats(-100, 100), st.floats(-100, 100)),
        right=st.tuples(st.floats(-100, 100), st.floats(-100, 100)),
    )
    @SETTINGS
    def test_dominance_is_asymmetric(
        self, left: tuple[float, float], right: tuple[float, float]
    ) -> None:
        assume(not any(math.isnan(value) for value in (*left, *right)))
        assert not (dominates(left, right) and dominates(right, left))

    @given(values=st.tuples(st.floats(-100, 100), st.floats(-100, 100)))
    @SETTINGS
    def test_dominance_is_irreflexive(self, values: tuple[float, float]) -> None:
        assume(not any(math.isnan(value) for value in values))
        assert not dominates(values, values)


class TestScoringProperties:
    @given(
        population=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.floats(min_value=1.0, max_value=1e7),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @SETTINGS
    def test_minmax_scores_stay_within_the_unit_interval(
        self, population: list[tuple[float, float]]
    ) -> None:
        objectives = ObjectiveSet(
            (
                Objective(
                    metric="accuracy",
                    direction=ObjectiveDirection.MAXIMIZE,
                    normalization=NormalizationStrategy.MINMAX,
                ),
                Objective(
                    metric="parameters",
                    direction=ObjectiveDirection.MINIMIZE,
                    normalization=NormalizationStrategy.MINMAX,
                ),
            )
        )
        entries = [
            (f"c{index}", {"accuracy": accuracy, "parameters": parameters})
            for index, (accuracy, parameters) in enumerate(population)
        ]
        scorer = WeightedScorer(objectives, entries)
        for result in scorer.score_all(entries):
            assert result.score is not None
            assert 0.0 <= result.score <= 1.0

    @given(
        population=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=10,
            unique=True,
        )
    )
    @SETTINGS
    def test_ranking_is_a_total_order(self, population: list[float]) -> None:
        objectives = ObjectiveSet(
            (Objective(metric="accuracy", direction=ObjectiveDirection.MAXIMIZE),)
        )
        entries = [
            (f"c{index}", f"h{index}", {"accuracy": value})
            for index, value in enumerate(population)
        ]
        result = rank_candidates(entries, objectives)
        ranks = [candidate.rank for candidate in result.ranked]
        assert ranks == list(range(len(population)))

    @given(
        population=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=8,
        )
    )
    @SETTINGS
    def test_ranking_is_independent_of_input_order(self, population: list[float]) -> None:
        objectives = ObjectiveSet(
            (Objective(metric="accuracy", direction=ObjectiveDirection.MAXIMIZE),)
        )
        entries = [
            (f"c{index}", f"h{index}", {"accuracy": value})
            for index, value in enumerate(population)
        ]
        forward = [c.candidate_id for c in rank_candidates(entries, objectives).ranked]
        backward = [
            c.candidate_id for c in rank_candidates(list(reversed(entries)), objectives).ranked
        ]
        assert forward == backward


class TestStrategyCheckpoints:
    @given(seed=seeds, drawn=st.integers(min_value=0, max_value=4))
    @SETTINGS
    def test_random_search_state_round_trips(self, seed: int, drawn: int) -> None:
        space = tiny_cnn_space()
        original = RandomSearch(
            space, seed=seed, max_evaluations=20, budget=TrainingBudget(epochs=1)
        )
        original.propose(drawn)
        state = original.state_dict()
        expected = [architecture_hash(p.spec) for p in original.propose(2)]

        restored = RandomSearch(
            space, seed=seed + 1, max_evaluations=20, budget=TrainingBudget(epochs=1)
        )
        restored.load_state_dict(state)
        assert [architecture_hash(p.spec) for p in restored.propose(2)] == expected

    @given(seed=seeds)
    @SETTINGS
    def test_sampler_state_round_trips(self, seed: int) -> None:
        space = tiny_cnn_space()
        original = ArchitectureSampler(space, seed=seed)
        original.sample()
        state = original.state_dict()
        expected = architecture_hash(original.sample())

        restored = ArchitectureSampler(space, seed=seed + 1)
        restored.load_state_dict(state)
        assert architecture_hash(restored.sample()) == expected

    @given(seed=seeds)
    @SETTINGS
    def test_mutation_state_round_trips(self, seed: int) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=seed).sample()
        original = MutationOperator(space, seed=seed)
        original.mutate(parent)
        state = original.state_dict()
        expected = architecture_hash(original.mutate(parent).child)

        restored = MutationOperator(space, seed=seed + 1)
        restored.load_state_dict(state)
        assert architecture_hash(restored.mutate(parent).child) == expected


class TestStateMachineProperties:
    @given(
        source=st.sampled_from(list(CandidateState)),
        target=st.sampled_from(list(CandidateState)),
    )
    @SETTINGS
    def test_only_table_edges_are_permitted(
        self, source: CandidateState, target: CandidateState
    ) -> None:
        if target in ALLOWED_TRANSITIONS[source]:
            validate_transition(source, target)
        else:
            with pytest.raises(InvalidStateTransitionError):
                validate_transition(source, target)

    @given(state=st.sampled_from(list(CandidateState)))
    @SETTINGS
    def test_no_state_transitions_to_itself(self, state: CandidateState) -> None:
        assert not can_transition(state, state)

    @given(state=st.sampled_from(list(CandidateState)))
    @SETTINGS
    def test_terminal_states_are_absorbing(self, state: CandidateState) -> None:
        if state.is_terminal:
            assert all(not can_transition(state, other) for other in CandidateState)

    @given(path=st.lists(st.sampled_from(list(CandidateState)), min_size=1, max_size=6))
    @SETTINGS
    def test_a_machine_never_leaves_a_terminal_state(self, path: list[CandidateState]) -> None:
        from nas_engine.orchestration.lifecycle import CandidateStateMachine

        machine = CandidateStateMachine()
        for target in path:
            machine.try_transition(target)
            if machine.is_terminal:
                break
        if machine.is_terminal:
            for target in CandidateState:
                assert not machine.try_transition(target)


def test_random_module_is_never_used_implicitly() -> None:
    """Sampling must not depend on the global RNG.

    Seeding the global generator differently between two runs must not change what a
    seeded sampler produces. If it did, unrelated code drawing random numbers would
    silently perturb a search.
    """
    space = tiny_cnn_space()
    random.seed(1)
    first = architecture_hash(ArchitectureSampler(space, seed=5).sample())
    random.seed(999)
    second = architecture_hash(ArchitectureSampler(space, seed=5).sample())
    assert first == second
