"""Unit tests for multi-objective comparison.

Covers: objective and constraint validation, normalisation strategies, weighted scoring,
Pareto dominance and fronts, crowding distance, deterministic ranking, and the online
scalarisation rule that search strategies rely on.
"""

from __future__ import annotations

import math

import pytest

from nas_engine.exceptions import ObjectiveError
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
from nas_engine.objectives.online import (
    online_objective_value,
    uses_stable_scalarization,
)
from nas_engine.objectives.pareto import (
    ObjectiveVector,
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front,
    to_objective_vector,
)
from nas_engine.objectives.ranking import UNRANKED_FRONT, rank_candidates
from nas_engine.objectives.scoring import (
    NEUTRAL_NORMALIZED_VALUE,
    WeightedScorer,
    compute_stats,
    normalize_value,
)

pytestmark = pytest.mark.unit


ACCURACY = Objective(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE)
PARAMETERS = Objective(
    metric="trainable_parameters", direction=ObjectiveDirection.MINIMIZE, weight=0.5
)


class TestObjectiveValidation:
    def test_direction_signs(self) -> None:
        assert ObjectiveDirection.MAXIMIZE.sign == 1.0
        assert ObjectiveDirection.MINIMIZE.sign == -1.0

    def test_empty_metric_name_is_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="must not be empty"):
            Objective(metric="  ", direction=ObjectiveDirection.MAXIMIZE)

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="negative weight"):
            Objective(metric="x", direction=ObjectiveDirection.MAXIMIZE, weight=-1)

    def test_reference_normalisation_requires_a_reference(self) -> None:
        with pytest.raises(ObjectiveError, match="reference normalisation"):
            Objective(
                metric="x",
                direction=ObjectiveDirection.MAXIMIZE,
                normalization=NormalizationStrategy.REFERENCE,
            )

    def test_optional_objective_requires_a_fallback(self) -> None:
        with pytest.raises(ObjectiveError, match="no missing_value"):
            Objective(metric="x", direction=ObjectiveDirection.MAXIMIZE, required=False)

    def test_empty_objective_set_is_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="at least one objective"):
            ObjectiveSet(())

    def test_duplicate_metrics_are_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="duplicate metrics"):
            ObjectiveSet((ACCURACY, ACCURACY))

    def test_zero_total_weight_is_rejected(self) -> None:
        zero = Objective(metric="x", direction=ObjectiveDirection.MAXIMIZE, weight=0.0)
        with pytest.raises(ObjectiveError, match="sum to zero"):
            ObjectiveSet((zero,))

    def test_weights_normalise_to_one(self) -> None:
        weights = ObjectiveSet((ACCURACY, PARAMETERS)).normalized_weights()
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] > weights[1]

    def test_lookup_by_metric(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        assert objectives.by_metric("trainable_parameters") is PARAMETERS
        with pytest.raises(ObjectiveError, match="no objective is defined"):
            objectives.by_metric("missing")

    def test_primary_is_the_first_objective(self) -> None:
        assert ObjectiveSet((ACCURACY, PARAMETERS)).primary is ACCURACY

    def test_describe_lists_every_objective(self) -> None:
        text = ObjectiveSet((ACCURACY, PARAMETERS)).describe()
        assert "maximize validation_accuracy" in text
        assert "minimize trainable_parameters" in text

    def test_default_objective_set_is_valid(self) -> None:
        objectives = default_objectives()
        assert objectives.primary.metric == "validation_accuracy"
        assert len(objectives.objectives) == 4


class TestConstraints:
    @pytest.mark.parametrize(
        ("operator", "value", "threshold", "satisfied"),
        [
            (ComparisonOperator.LE, 5.0, 5.0, True),
            (ComparisonOperator.LT, 5.0, 5.0, False),
            (ComparisonOperator.GE, 5.0, 5.0, True),
            (ComparisonOperator.GT, 6.0, 5.0, True),
        ],
    )
    def test_operators_compare_correctly(
        self, operator: ComparisonOperator, value: float, threshold: float, satisfied: bool
    ) -> None:
        assert operator.compare(value, threshold) is satisfied

    def test_symbols_are_readable(self) -> None:
        assert ComparisonOperator.LE.symbol == "<="
        assert ComparisonOperator.GT.symbol == ">"

    def test_satisfied_constraint_reports_nothing(self) -> None:
        constraint = MetricConstraint("params", ComparisonOperator.LE, 100)
        assert constraint.evaluate({"params": 50}) is None

    def test_violation_is_described(self) -> None:
        constraint = MetricConstraint("params", ComparisonOperator.LE, 100)
        message = constraint.evaluate({"params": 500})
        assert message is not None
        assert "params=500" in message

    def test_missing_required_metric_is_a_violation(self) -> None:
        constraint = MetricConstraint("latency", ComparisonOperator.LE, 10)
        assert "was not measured" in str(constraint.evaluate({}))

    def test_missing_optional_metric_is_tolerated(self) -> None:
        constraint = MetricConstraint("latency", ComparisonOperator.LE, 10, required=False)
        assert constraint.evaluate({}) is None

    def test_empty_metric_name_is_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="must not be empty"):
            MetricConstraint("", ComparisonOperator.LE, 1)

    def test_set_reports_every_violation(self) -> None:
        constraints = ConstraintSet(
            (
                MetricConstraint("a", ComparisonOperator.LE, 1),
                MetricConstraint("b", ComparisonOperator.GE, 10),
            )
        )
        violations = constraints.violations({"a": 5, "b": 1})
        assert len(violations) == 2
        assert not constraints.is_feasible({"a": 5, "b": 1})

    def test_empty_set_is_always_feasible(self) -> None:
        assert ConstraintSet().is_feasible({})
        assert ConstraintSet().describe() == "Constraints: none"


class TestNormalisation:
    def test_stats_ignore_infinities(self) -> None:
        stats = compute_stats("x", [1.0, 2.0, float("inf")])
        assert stats.minimum == 1.0
        assert stats.maximum == 2.0
        assert stats.count == 2

    def test_degenerate_stats_are_detected(self) -> None:
        assert compute_stats("x", [3.0, 3.0]).degenerate
        assert compute_stats("x", []).degenerate

    def test_minmax_maps_to_unit_interval(self) -> None:
        stats = compute_stats("m", [0.0, 10.0])
        assert normalize_value(0.0, ACCURACY, stats) == 0.0
        assert normalize_value(10.0, ACCURACY, stats) == 1.0
        assert normalize_value(5.0, ACCURACY, stats) == pytest.approx(0.5)

    def test_minimisation_inverts_the_scale(self) -> None:
        objective = Objective(metric="m", direction=ObjectiveDirection.MINIMIZE)
        stats = compute_stats("m", [0.0, 10.0])
        assert normalize_value(0.0, objective, stats) == 1.0
        assert normalize_value(10.0, objective, stats) == 0.0

    def test_degenerate_objective_is_neutral(self) -> None:
        stats = compute_stats("m", [5.0, 5.0])
        assert normalize_value(5.0, ACCURACY, stats) == NEUTRAL_NORMALIZED_VALUE

    def test_log_scale_compresses_wide_ranges(self) -> None:
        objective = Objective(
            metric="m",
            direction=ObjectiveDirection.MAXIMIZE,
            normalization=NormalizationStrategy.LOG,
        )
        stats = compute_stats("m", [1.0, 1_000_000.0])
        midpoint = normalize_value(1000.0, objective, stats)
        assert 0.4 < midpoint < 0.6

    def test_zscore_centres_on_the_mean(self) -> None:
        objective = Objective(
            metric="m",
            direction=ObjectiveDirection.MAXIMIZE,
            normalization=NormalizationStrategy.ZSCORE,
        )
        stats = compute_stats("m", [0.0, 10.0])
        assert normalize_value(5.0, objective, stats) == pytest.approx(0.0)

    def test_reference_normalisation_is_population_independent(self) -> None:
        objective = Objective(
            metric="m",
            direction=ObjectiveDirection.MAXIMIZE,
            normalization=NormalizationStrategy.REFERENCE,
            reference=100.0,
        )
        stats = compute_stats("m", [1.0, 2.0])
        assert normalize_value(50.0, objective, stats) == pytest.approx(0.5)

    def test_none_normalisation_passes_the_raw_value(self) -> None:
        objective = Objective(
            metric="m",
            direction=ObjectiveDirection.MINIMIZE,
            normalization=NormalizationStrategy.NONE,
        )
        stats = compute_stats("m", [1.0, 2.0])
        assert normalize_value(3.0, objective, stats) == -3.0

    def test_infinite_sentinels_clamp_to_the_extremes(self) -> None:
        minimise = Objective(metric="m", direction=ObjectiveDirection.MINIMIZE)
        stats = compute_stats("m", [1.0, 2.0])
        assert normalize_value(float("inf"), minimise, stats) == 0.0
        assert normalize_value(float("-inf"), minimise, stats) == 1.0

    def test_nan_scores_as_the_worst_value(self) -> None:
        stats = compute_stats("m", [1.0, 2.0])
        assert normalize_value(float("nan"), ACCURACY, stats) == 0.0


class TestWeightedScoring:
    @staticmethod
    def _population() -> list[tuple[str, dict[str, float]]]:
        return [
            ("a", {"validation_accuracy": 0.9, "trainable_parameters": 1_000_000.0}),
            ("b", {"validation_accuracy": 0.5, "trainable_parameters": 1_000.0}),
            ("c", {"validation_accuracy": 0.7, "trainable_parameters": 100_000.0}),
        ]

    def test_scores_every_candidate(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        results = WeightedScorer(objectives, self._population()).score_all(self._population())
        assert len(results) == 3
        assert all(result.is_scored for result in results)

    def test_score_reflects_the_weighting(self) -> None:
        heavy_accuracy = ObjectiveSet(
            (
                Objective(
                    metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE, weight=10.0
                ),
                PARAMETERS,
            )
        )
        scorer = WeightedScorer(heavy_accuracy, self._population())
        scores = {r.candidate_id: r.score for r in scorer.score_all(self._population())}
        assert scores["a"] is not None and scores["b"] is not None
        assert scores["a"] > scores["b"]

    def test_missing_required_metric_leaves_the_score_unset(self) -> None:
        objectives = ObjectiveSet((ACCURACY,))
        scorer = WeightedScorer(objectives, self._population())
        result = scorer.score("d", {"trainable_parameters": 10.0})
        assert result.score is None
        assert result.missing_metrics == ("validation_accuracy",)

    def test_optional_metric_uses_its_fallback(self) -> None:
        optional = Objective(
            metric="latency",
            direction=ObjectiveDirection.MINIMIZE,
            required=False,
            missing_value=float("inf"),
        )
        objectives = ObjectiveSet((ACCURACY, optional))
        scorer = WeightedScorer(objectives, self._population())
        result = scorer.score("a", {"validation_accuracy": 0.9})
        assert result.is_scored
        assert result.components["latency"] == 0.0

    def test_stats_are_exposed_for_inspection(self) -> None:
        scorer = WeightedScorer(ObjectiveSet((ACCURACY,)), self._population())
        assert scorer.stats["validation_accuracy"].maximum == 0.9


class TestParetoDominance:
    def test_strictly_better_dominates(self) -> None:
        assert dominates([1.0, 1.0], [0.0, 0.0])

    def test_equal_vectors_do_not_dominate(self) -> None:
        assert not dominates([1.0, 1.0], [1.0, 1.0])

    def test_mixed_comparisons_are_incomparable(self) -> None:
        assert not dominates([1.0, 0.0], [0.0, 1.0])
        assert not dominates([0.0, 1.0], [1.0, 0.0])

    def test_better_on_one_axis_and_tied_on_another_dominates(self) -> None:
        assert dominates([1.0, 1.0], [1.0, 0.0])

    def test_nan_never_dominates(self) -> None:
        assert not dominates([float("nan"), 1.0], [0.0, 0.0])
        assert not dominates([1.0, 1.0], [float("nan"), 0.0])

    def test_tiny_differences_are_treated_as_ties(self) -> None:
        assert not dominates([1.0, 1.0], [1.0 - 1e-15, 1.0])

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="different lengths"):
            dominates([1.0], [1.0, 2.0])

    def test_empty_vectors_are_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="empty objective vectors"):
            dominates([], [])

    def test_dominance_is_transitive(self) -> None:
        assert dominates([3.0, 3.0], [2.0, 2.0])
        assert dominates([2.0, 2.0], [1.0, 1.0])
        assert dominates([3.0, 3.0], [1.0, 1.0])


class TestParetoFront:
    @staticmethod
    def _vectors() -> list[ObjectiveVector]:
        return [
            ObjectiveVector("a", (0.9, -1_000_000.0), (0.9, 1_000_000.0)),
            ObjectiveVector("b", (0.5, -1_000.0), (0.5, 1_000.0)),
            ObjectiveVector("c", (0.4, -10_000.0), (0.4, 10_000.0)),
        ]

    def test_front_excludes_dominated_members(self) -> None:
        front = {vector.candidate_id for vector in pareto_front(self._vectors())}
        assert front == {"a", "b"}

    def test_front_members_are_mutually_non_dominated(self) -> None:
        front = pareto_front(self._vectors())
        for first in front:
            for second in front:
                if first.candidate_id != second.candidate_id:
                    assert not dominates(first.values, second.values)

    def test_front_ordering_is_deterministic(self) -> None:
        first = [vector.candidate_id for vector in pareto_front(self._vectors())]
        shuffled = list(reversed(self._vectors()))
        second = [vector.candidate_id for vector in pareto_front(shuffled)]
        assert first == second

    def test_non_dominated_sort_partitions_everything(self) -> None:
        fronts = non_dominated_sort(self._vectors())
        assigned = [vector.candidate_id for front in fronts for vector in front]
        assert sorted(assigned) == ["a", "b", "c"]
        assert {vector.candidate_id for vector in fronts[0]} == {"a", "b"}

    def test_empty_input_yields_no_fronts(self) -> None:
        assert non_dominated_sort([]) == []

    def test_crowding_distance_favours_the_extremes(self) -> None:
        vectors = [
            ObjectiveVector("a", (0.0, 1.0), (0.0, 1.0)),
            ObjectiveVector("b", (0.5, 0.5), (0.5, 0.5)),
            ObjectiveVector("c", (1.0, 0.0), (1.0, 0.0)),
        ]
        distances = crowding_distance(vectors)
        assert math.isinf(distances["a"])
        assert math.isinf(distances["c"])
        assert distances["b"] < math.inf

    def test_small_fronts_are_all_extreme(self) -> None:
        vectors = [ObjectiveVector("a", (1.0,), (1.0,)), ObjectiveVector("b", (2.0,), (2.0,))]
        assert all(math.isinf(value) for value in crowding_distance(vectors).values())

    def test_vector_conversion_applies_direction(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        vector = to_objective_vector(
            "a", {"validation_accuracy": 0.9, "trainable_parameters": 100.0}, objectives
        )
        assert vector is not None
        assert vector.values == (0.9, -100.0)

    def test_missing_required_metric_yields_no_vector(self) -> None:
        objectives = ObjectiveSet((ACCURACY,))
        assert to_objective_vector("a", {}, objectives) is None


class TestRanking:
    @staticmethod
    def _population() -> list[tuple[str, str, dict[str, float]]]:
        return [
            ("a", "hash-a", {"validation_accuracy": 0.9, "trainable_parameters": 1e6}),
            ("b", "hash-b", {"validation_accuracy": 0.5, "trainable_parameters": 1e3}),
            ("c", "hash-c", {"validation_accuracy": 0.4, "trainable_parameters": 1e4}),
        ]

    def test_produces_a_total_order(self) -> None:
        result = rank_candidates(self._population(), ObjectiveSet((ACCURACY, PARAMETERS)))
        assert [candidate.rank for candidate in result.ranked] == [0, 1, 2]

    def test_ranking_is_deterministic_under_reordering(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        first = [c.candidate_id for c in rank_candidates(self._population(), objectives).ranked]
        shuffled = list(reversed(self._population()))
        second = [c.candidate_id for c in rank_candidates(shuffled, objectives).ranked]
        assert first == second

    def test_infeasible_candidates_rank_last(self) -> None:
        constraints = ConstraintSet(
            (MetricConstraint("trainable_parameters", ComparisonOperator.LE, 1e4),)
        )
        result = rank_candidates(
            self._population(), ObjectiveSet((ACCURACY, PARAMETERS)), constraints=constraints
        )
        assert result.ranked[-1].candidate_id == "a"
        assert not result.ranked[-1].feasible
        assert result.ranked[-1].violations

    def test_pareto_front_only_contains_feasible_candidates(self) -> None:
        constraints = ConstraintSet(
            (MetricConstraint("trainable_parameters", ComparisonOperator.LE, 1e4),)
        )
        result = rank_candidates(
            self._population(), ObjectiveSet((ACCURACY, PARAMETERS)), constraints=constraints
        )
        assert all(candidate.feasible for candidate in result.pareto_front)
        assert "a" not in {candidate.candidate_id for candidate in result.pareto_front}

    def test_unscoreable_candidates_are_kept_but_marked(self) -> None:
        population = [*self._population(), ("d", "hash-d", {})]
        result = rank_candidates(population, ObjectiveSet((ACCURACY, PARAMETERS)))
        assert "d" in result.unscored
        entry = result.by_id("d")
        assert entry is not None
        assert entry.score is None
        assert entry.pareto_rank == UNRANKED_FRONT

    def test_best_is_the_top_ranked_candidate(self) -> None:
        result = rank_candidates(self._population(), ObjectiveSet((ACCURACY,)))
        assert result.best is not None
        assert result.best.candidate_id == "a"

    def test_empty_population_has_no_best(self) -> None:
        assert rank_candidates([], ObjectiveSet((ACCURACY,))).best is None

    def test_lookup_of_unknown_candidate_returns_none(self) -> None:
        result = rank_candidates(self._population(), ObjectiveSet((ACCURACY,)))
        assert result.by_id("nope") is None

    def test_serialises_to_plain_data(self) -> None:
        payload = rank_candidates(self._population(), ObjectiveSet((ACCURACY,))).to_dict()
        assert len(payload["ranked"]) == 3
        assert isinstance(payload["pareto_front"], list)


class TestOnlineScalarisation:
    def test_falls_back_to_the_primary_metric(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        assert not uses_stable_scalarization(objectives)
        value = online_objective_value(
            {"validation_accuracy": 0.8, "trainable_parameters": 1e6}, objectives
        )
        assert value == pytest.approx(0.8)

    def test_minimisation_primary_is_negated(self) -> None:
        objectives = ObjectiveSet((PARAMETERS,))
        assert online_objective_value({"trainable_parameters": 100.0}, objectives) == -100.0

    def test_stable_objectives_are_combined(self) -> None:
        objectives = ObjectiveSet(
            (
                Objective(
                    metric="validation_accuracy",
                    direction=ObjectiveDirection.MAXIMIZE,
                    normalization=NormalizationStrategy.NONE,
                ),
                Objective(
                    metric="latency_median_ms",
                    direction=ObjectiveDirection.MINIMIZE,
                    weight=1.0,
                    normalization=NormalizationStrategy.REFERENCE,
                    reference=10.0,
                ),
            )
        )
        assert uses_stable_scalarization(objectives)
        value = online_objective_value(
            {"validation_accuracy": 0.8, "latency_median_ms": 5.0}, objectives
        )
        assert value == pytest.approx((0.8 - 0.5) / 2)

    def test_missing_primary_metric_yields_none(self) -> None:
        assert online_objective_value({}, ObjectiveSet((ACCURACY,))) is None

    def test_non_finite_primary_metric_yields_none(self) -> None:
        objectives = ObjectiveSet((ACCURACY,))
        assert online_objective_value({"validation_accuracy": float("nan")}, objectives) is None

    def test_missing_required_stable_metric_yields_none(self) -> None:
        objectives = ObjectiveSet(
            (
                Objective(
                    metric="a",
                    direction=ObjectiveDirection.MAXIMIZE,
                    normalization=NormalizationStrategy.NONE,
                ),
            )
        )
        assert online_objective_value({}, objectives) is None

    def test_value_is_stable_regardless_of_population(self) -> None:
        objectives = ObjectiveSet((ACCURACY, PARAMETERS))
        first = online_objective_value({"validation_accuracy": 0.6}, objectives)
        second = online_objective_value({"validation_accuracy": 0.6}, objectives)
        assert first == second
