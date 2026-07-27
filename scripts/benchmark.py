"""Micro-benchmarks for the operations a search performs thousands of times.

Unlike the guards in ``tests/performance``, this script reports numbers rather than
asserting thresholds. Use it to answer "did that change make sampling slower?" and to
generate the figures quoted in ``docs/operations/production-runbook.md``.

Every measurement warms up first, uses a monotonic clock, and reports the median across
repeats so a single scheduler hiccup does not dominate. Results are machine specific;
the environment is printed alongside them for exactly that reason.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --repeats 20 --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nas_engine.architectures.cost import compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.shapes import infer_shapes
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.models.builder import ModelBuilder
from nas_engine.objectives.objective import default_objectives
from nas_engine.objectives.pareto import ObjectiveVector, pareto_front
from nas_engine.objectives.ranking import rank_candidates
from nas_engine.search.random_search import RandomSearch
from nas_engine.search_space.mutation import MutationOperator
from nas_engine.search_space.presets import default_cnn_space
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.validation import check_architecture
from nas_engine.utilities.environment import collect_environment


def measure(name: str, operation: Callable[[], Any], *, repeats: int) -> dict[str, Any]:
    """Time an operation and return its statistics.

    Args:
        name: Label for the measurement.
        operation: Zero-argument callable to time.
        repeats: Number of timed calls.

    Returns:
        A result record with median, mean, min, and max timings in milliseconds.
    """
    operation()  # warm up: exclude import, allocation, and first-call overhead
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "name": name,
        "repeats": repeats,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def build_benchmarks(repeats: int) -> list[dict[str, Any]]:
    """Run every benchmark and return the results.

    Args:
        repeats: Timed calls per benchmark.

    Returns:
        One record per benchmark.
    """
    space = default_cnn_space()
    sampler = ArchitectureSampler(space, seed=42)
    spec = sampler.sample()
    builder = ModelBuilder()
    uninitialised = ModelBuilder(initialize=False)
    mutator = MutationOperator(space, seed=7)

    vectors = [
        ObjectiveVector(f"c{index}", (index / 200, 1 - index / 200), (0.0, 0.0))
        for index in range(200)
    ]
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
    objectives = default_objectives()

    strategy = RandomSearch(space, seed=1, max_evaluations=500, budget=TrainingBudget(epochs=1))
    strategy.propose(100)
    state = strategy.state_dict()

    def restore_state() -> None:
        fresh = RandomSearch(space, seed=2, max_evaluations=500, budget=TrainingBudget(epochs=1))
        fresh.load_state_dict(state)

    return [
        measure("sample_architecture", sampler.sample, repeats=repeats),
        measure("hash_architecture", lambda: architecture_hash(spec), repeats=repeats),
        measure("infer_shapes", lambda: infer_shapes(spec), repeats=repeats),
        measure("compute_cost", lambda: compute_cost(spec), repeats=repeats),
        measure("validate_architecture", lambda: check_architecture(spec, space), repeats=repeats),
        measure("mutate_architecture", lambda: mutator.mutate(spec), repeats=repeats),
        measure(
            "build_model_uninitialised",
            lambda: uninitialised.build(spec),
            repeats=max(5, repeats // 10),
        ),
        measure(
            "build_model_initialised", lambda: builder.build(spec), repeats=max(5, repeats // 10)
        ),
        measure("pareto_front_200", lambda: pareto_front(vectors), repeats=max(5, repeats // 10)),
        measure(
            "rank_200_candidates",
            lambda: rank_candidates(population, objectives),
            repeats=max(5, repeats // 10),
        ),
        measure("serialise_strategy_state", strategy.state_dict, repeats=max(5, repeats // 5)),
        measure("restore_strategy_state", restore_state, repeats=max(5, repeats // 5)),
    ]


def main() -> int:
    """Parse arguments, run the benchmarks, and print the results.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=100, help="timed calls per benchmark (default: 100)"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the results to this JSON file"
    )
    arguments = parser.parse_args()

    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")

    environment = collect_environment()
    print("Environment")
    for line in environment.summary_lines():
        print(f"  {line}")
    print(
        "\nTimings are specific to this machine and load. Compare runs on the same host, "
        "never across hosts.\n"
    )

    results = build_benchmarks(arguments.repeats)
    width = max(len(result["name"]) for result in results)
    print(f"{'benchmark':<{width}}  {'median':>10}  {'mean':>10}  {'min':>10}  {'max':>10}")
    print("-" * (width + 48))
    for result in results:
        print(
            f"{result['name']:<{width}}  "
            f"{result['median_ms']:>9.4f}m  "
            f"{result['mean_ms']:>9.4f}m  "
            f"{result['min_ms']:>9.4f}m  "
            f"{result['max_ms']:>9.4f}m"
        )
    print("\n(all times in milliseconds; 'm' suffix denotes ms)")

    if arguments.json is not None:
        payload = {"environment": environment.to_dict(), "results": results}
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
