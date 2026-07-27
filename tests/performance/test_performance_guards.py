"""Coarse performance guards.

These are **guard rails, not benchmarks**. Each threshold is set roughly an order of
magnitude above the observed time on a modest laptop CPU, so ordinary machine-to-machine
variation, CI noise, and a busy scheduler cannot make them fail. What they catch is an
accidental algorithmic regression — an O(n) operation becoming O(n²), a per-call
allocation becoming a per-call model build, a query losing its index.

They are environment sensitive by nature. If one fails, the first question is "is this
machine loaded?", not "is the code broken". Timing is measured with
:func:`time.perf_counter`, a monotonic clock, and every measurement warms up first so
that import and allocator behaviour is excluded.

Run the real benchmark (``scripts/benchmark.py``) for numbers worth quoting.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nas_engine.architectures.cost import compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.shapes import infer_shapes
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.models.builder import ModelBuilder
from nas_engine.objectives.objective import (
    Objective,
    ObjectiveDirection,
    ObjectiveSet,
)
from nas_engine.objectives.pareto import ObjectiveVector, pareto_front
from nas_engine.objectives.ranking import rank_candidates
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import ensure_schema
from nas_engine.persistence.repository import SearchRepository
from nas_engine.search.random_search import RandomSearch
from nas_engine.search_space.presets import default_cnn_space
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.validation import check_architecture

pytestmark = pytest.mark.performance


def _timed(operation: object, *, repeats: int) -> float:
    """Return the mean seconds per call, after one warm-up call.

    Args:
        operation: Zero-argument callable to time.
        repeats: Number of timed calls.

    Returns:
        Mean seconds per call.
    """
    operation()  # type: ignore[operator]
    start = time.perf_counter()
    for _ in range(repeats):
        operation()  # type: ignore[operator]
    return (time.perf_counter() - start) / repeats


class TestArchitectureOperations:
    def test_sampling_is_fast(self) -> None:
        space = default_cnn_space()
        sampler = ArchitectureSampler(space, seed=1)
        per_call = _timed(sampler.sample, repeats=200)
        # Observed around 0.1 ms; the guard is 20 ms.
        assert per_call < 0.020, f"sampling took {per_call * 1000:.2f} ms per call"

    def test_validation_is_fast(self) -> None:
        space = default_cnn_space()
        spec = ArchitectureSampler(space, seed=2).sample()
        per_call = _timed(lambda: check_architecture(spec, space), repeats=500)
        assert per_call < 0.010, f"validation took {per_call * 1000:.2f} ms per call"

    def test_hashing_is_fast(self) -> None:
        spec = ArchitectureSampler(default_cnn_space(), seed=3).sample()
        per_call = _timed(lambda: architecture_hash(spec), repeats=1000)
        assert per_call < 0.005, f"hashing took {per_call * 1000:.2f} ms per call"

    def test_shape_inference_is_fast(self) -> None:
        spec = ArchitectureSampler(default_cnn_space(), seed=4).sample()
        per_call = _timed(lambda: infer_shapes(spec), repeats=1000)
        assert per_call < 0.005, f"shape inference took {per_call * 1000:.2f} ms per call"

    def test_the_cost_model_is_far_cheaper_than_building(self) -> None:
        spec = ArchitectureSampler(default_cnn_space(), seed=5).sample()
        builder = ModelBuilder(initialize=False)
        analytic = _timed(lambda: compute_cost(spec), repeats=500)
        built = _timed(lambda: builder.build(spec), repeats=20)
        # The whole point of the analytic model is that it is cheap enough to run on every
        # proposal. A 5x margin is a weak claim; the real ratio is usually 100x or more.
        assert analytic * 5 < built, (
            f"analytic cost {analytic * 1000:.3f} ms is not clearly cheaper than "
            f"building {built * 1000:.3f} ms"
        )

    def test_model_construction_is_bounded(self) -> None:
        spec = ArchitectureSampler(default_cnn_space(), seed=6).sample()
        builder = ModelBuilder()
        per_call = _timed(lambda: builder.build(spec), repeats=20)
        assert per_call < 2.0, f"model construction took {per_call:.3f} s"


class TestParetoScaling:
    @pytest.mark.parametrize("count", [50, 200])
    def test_front_computation_stays_usable(self, count: int) -> None:
        vectors = [
            ObjectiveVector(
                f"c{index}", (index / count, 1 - index / count), (index / count, 1 - index / count)
            )
            for index in range(count)
        ]
        start = time.perf_counter()
        front = pareto_front(vectors)
        elapsed = time.perf_counter() - start
        assert front
        assert elapsed < 5.0, f"Pareto front over {count} candidates took {elapsed:.2f} s"

    def test_ranking_a_realistic_population_is_fast(self) -> None:
        objectives = ObjectiveSet(
            (
                Objective(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE),
                Objective(
                    metric="trainable_parameters", direction=ObjectiveDirection.MINIMIZE, weight=0.3
                ),
            )
        )
        population = [
            (
                f"c{index}",
                f"h{index}",
                {
                    "validation_accuracy": (index * 37 % 100) / 100,
                    "trainable_parameters": float(1000 * (index + 1)),
                },
            )
            for index in range(200)
        ]
        start = time.perf_counter()
        rank_candidates(population, objectives)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"ranking 200 candidates took {elapsed:.2f} s"


class TestPersistenceScaling:
    def test_bulk_insertion_is_usable(self, tmp_path: Path) -> None:
        database = Database.from_path(tmp_path / "perf.db")
        ensure_schema(database)
        repository = SearchRepository(database)
        try:
            search_id = repository.create_search(
                name="perf",
                strategy="random_search",
                config={"version": 1},
                config_hash="h",
                config_version=1,
                search_space={},
                seed=1,
                seeds={},
                environment={},
                planned_evaluations=100,
            )
            sampler = ArchitectureSampler(default_cnn_space(), seed=7)
            specs = [sampler.sample() for _ in range(50)]

            start = time.perf_counter()
            for spec in specs:
                repository.add_candidate(
                    search_id=search_id,
                    architecture_hash=architecture_hash(spec),
                    spec=spec,
                )
            elapsed = time.perf_counter() - start
            assert elapsed < 20.0, f"inserting 50 candidates took {elapsed:.2f} s"
        finally:
            database.dispose()

    def test_listing_candidates_is_fast(self, tmp_path: Path) -> None:
        database = Database.from_path(tmp_path / "perf.db")
        ensure_schema(database)
        repository = SearchRepository(database)
        try:
            search_id = repository.create_search(
                name="perf",
                strategy="random_search",
                config={"version": 1},
                config_hash="h",
                config_version=1,
                search_space={},
                seed=1,
                seeds={},
                environment={},
                planned_evaluations=100,
            )
            sampler = ArchitectureSampler(default_cnn_space(), seed=8)
            for _ in range(30):
                spec = sampler.sample()
                repository.add_candidate(
                    search_id=search_id,
                    architecture_hash=architecture_hash(spec),
                    spec=spec,
                )
            per_call = _timed(lambda: repository.list_candidates(search_id), repeats=10)
            assert per_call < 2.0, f"listing took {per_call:.3f} s"
        finally:
            database.dispose()

    def test_status_counting_does_not_scan_linearly_in_python(self, tmp_path: Path) -> None:
        database = Database.from_path(tmp_path / "perf.db")
        ensure_schema(database)
        repository = SearchRepository(database)
        try:
            search_id = repository.create_search(
                name="perf",
                strategy="random_search",
                config={"version": 1},
                config_hash="h",
                config_version=1,
                search_space={},
                seed=1,
                seeds={},
                environment={},
                planned_evaluations=100,
            )
            sampler = ArchitectureSampler(default_cnn_space(), seed=9)
            for _ in range(40):
                spec = sampler.sample()
                repository.add_candidate(
                    search_id=search_id,
                    architecture_hash=architecture_hash(spec),
                    spec=spec,
                    status=CandidateState.PROPOSED,
                )
            per_call = _timed(lambda: repository.count_candidates_by_status(search_id), repeats=20)
            # Counting is a single GROUP BY; it must not materialise every candidate.
            assert per_call < 0.5, f"counting took {per_call * 1000:.1f} ms"
        finally:
            database.dispose()


class TestResumeScaling:
    def test_restoring_strategy_state_is_fast(self) -> None:
        space = default_cnn_space()
        strategy = RandomSearch(space, seed=1, max_evaluations=200, budget=TrainingBudget(epochs=1))
        strategy.propose(100)
        state = strategy.state_dict()

        def restore() -> None:
            fresh = RandomSearch(
                space, seed=2, max_evaluations=200, budget=TrainingBudget(epochs=1)
            )
            fresh.load_state_dict(state)

        per_call = _timed(restore, repeats=20)
        assert per_call < 1.0, f"restoring state took {per_call * 1000:.1f} ms"

    def test_checkpoint_serialisation_is_fast(self) -> None:
        space = default_cnn_space()
        strategy = RandomSearch(space, seed=1, max_evaluations=200, budget=TrainingBudget(epochs=1))
        strategy.propose(100)
        per_call = _timed(strategy.state_dict, repeats=50)
        assert per_call < 0.5, f"serialising state took {per_call * 1000:.1f} ms"
