"""Define objectives and constraints, and see how they change the answer.

The same set of evaluated candidates produces different "best" architectures depending on
what you asked for. This example runs one search, then ranks its results four different
ways to make that concrete.

It also shows the difference between an *objective* ("smaller is better") and a
*constraint* ("larger than this is unacceptable at any price") — a distinction that is easy
to blur and expensive to get wrong.

Run it with::

    python examples/custom_objective.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nas_engine import SearchConfig, SearchEngine
from nas_engine.objectives import (
    ComparisonOperator,
    ConstraintSet,
    MetricConstraint,
    NormalizationStrategy,
    Objective,
    ObjectiveDirection,
    ObjectiveSet,
    RankingResult,
    rank_candidates,
)

ACCURACY_ONLY = ObjectiveSet(
    (Objective(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE),)
)

ACCURACY_AND_SIZE = ObjectiveSet(
    (
        Objective(
            metric="validation_accuracy",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.0,
        ),
        Objective(
            metric="trainable_parameters",
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.5,
            # A log scale, because parameter counts span orders of magnitude and a linear
            # scale would make every small model look identical.
            normalization=NormalizationStrategy.LOG,
        ),
    )
)

SIZE_DOMINANT = ObjectiveSet(
    (
        Objective(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE, weight=0.2),
        Objective(
            metric="trainable_parameters",
            direction=ObjectiveDirection.MINIMIZE,
            weight=2.0,
            normalization=NormalizationStrategy.LOG,
        ),
    )
)

#: A hard limit, not a preference. No amount of accuracy buys past it.
SIZE_LIMIT = ConstraintSet(
    (MetricConstraint("trainable_parameters", ComparisonOperator.LE, 5_000),)
)


def build_config(output_dir: Path) -> SearchConfig:
    """Build a small search whose candidates vary widely in size.

    Args:
        output_dir: Where results are written.

    Returns:
        A validated configuration.
    """
    return SearchConfig.from_mapping(
        {
            "project": {"name": "objectives", "output_dir": str(output_dir)},
            "dataset": {
                "provider": "synthetic",
                "batch_size": 32,
                "options": {
                    "num_classes": 4,
                    "input_size": 16,
                    "train_samples": 256,
                    "validation_samples": 128,
                    "test_samples": 128,
                    "seed": 42,
                },
            },
            "search_space": {"preset": "tiny_cnn"},
            "budget": {"max_evaluations": 8, "epochs": 1},
            "evaluation": {"measure_latency": False},
            "hardware": {"device": "cpu"},
            "logging": {"level": "WARNING"},
        }
    )


def show(title: str, ranking: RankingResult) -> None:
    """Print the top of a ranking with the metrics that drove it.

    Args:
        title: Heading for this ranking.
        ranking: The ranking to display.
    """
    print(f"\n{title}")
    print(f"  {'rank':<5}{'architecture':<16}{'accuracy':>10}{'params':>10}{'score':>9}  front")
    for candidate in ranking.ranked[:5]:
        accuracy = candidate.metrics.get("validation_accuracy", float("nan"))
        parameters = int(candidate.metrics.get("trainable_parameters", 0))
        score = f"{candidate.score:.4f}" if candidate.score is not None else "n/a"
        front = (
            "yes"
            if candidate.on_pareto_front
            else ("infeasible" if not candidate.feasible else "no")
        )
        print(
            f"  {candidate.rank:<5}{candidate.architecture_hash[:12]:<16}"
            f"{accuracy:>10.4f}{parameters:>10,}{score:>9}  {front}"
        )


def main() -> int:
    """Run one search and rank its results under four different preferences.

    Returns:
        A process exit code.
    """
    with tempfile.TemporaryDirectory(prefix="nas-objectives-") as temporary:
        config = build_config(Path(temporary))
        engine = SearchEngine(config)
        try:
            result = engine.run()
            population = engine.repository.completed_metrics(result.search_id)
        finally:
            engine.close()

        print(f"Evaluated {len(population)} candidates. The same results, ranked four ways:")

        show("1. Accuracy only", rank_candidates(population, ACCURACY_ONLY))
        show(
            "2. Accuracy, with size as a secondary objective",
            rank_candidates(population, ACCURACY_AND_SIZE),
        )
        show("3. Size-dominant weighting", rank_candidates(population, SIZE_DOMINANT))
        show(
            "4. Accuracy, with a hard 5,000-parameter limit",
            rank_candidates(population, ACCURACY_AND_SIZE, constraints=SIZE_LIMIT),
        )

        print(
            "\nThe winner changes with the preference, which is the point: a weighted score "
            "encodes an exchange rate, and there is no universally correct one. The Pareto "
            "front is the answer that does not require choosing:"
        )
        front = rank_candidates(population, ACCURACY_AND_SIZE).pareto_front
        for candidate in front:
            print(
                f"  {candidate.architecture_hash[:12]}  "
                f"accuracy={candidate.metrics['validation_accuracy']:.4f}  "
                f"parameters={int(candidate.metrics['trainable_parameters']):,}"
            )

        infeasible = [
            candidate
            for candidate in rank_candidates(
                population, ACCURACY_AND_SIZE, constraints=SIZE_LIMIT
            ).ranked
            if not candidate.feasible
        ]
        print(
            f"\n{len(infeasible)} candidate(s) violate the hard limit and rank below every "
            "feasible one, regardless of accuracy."
        )
        for candidate in infeasible[:3]:
            print(f"  {candidate.architecture_hash[:12]}: {candidate.violations[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
