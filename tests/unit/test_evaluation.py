"""Unit tests for the evaluation layer.

Covers: budget arithmetic and serialisation, failure classification and the retry
decision it drives, latency benchmarking methodology, model-size measurement, and the
candidate evaluator's success and failure paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.loaders import LoaderSettings
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import (
    CandidateEvaluator,
    EvaluationContext,
    EvaluationSettings,
)
from nas_engine.evaluation.latency import (
    LATENCY_WARNING,
    LatencyMeasurement,
    measure_latency,
)
from nas_engine.evaluation.model_size import measure_model_size, save_model_weights
from nas_engine.evaluation.result import (
    EvaluationFailure,
    EvaluationResult,
    FailureKind,
    classify_failure,
)
from nas_engine.exceptions import (
    ArchitectureValidationError,
    CheckpointError,
    ConfigurationError,
    ConstraintViolationError,
    EvaluationTimeoutError,
    ModelBuildError,
    NonFiniteLossError,
    PersistenceError,
    ResourceLimitError,
    TrainingError,
    WorkerError,
)
from nas_engine.models.builder import build_model
from nas_engine.training.optimizers import OptimizerSettings
from nas_engine.training.trainer import TrainingSettings

pytestmark = pytest.mark.unit


class TestTrainingBudget:
    def test_key_is_stable_and_filename_safe(self) -> None:
        budget = TrainingBudget(epochs=3, train_fraction=0.5, resolution=16, rung=1)
        assert budget.key == "e3_f0.5_r16_rung1"
        assert "/" not in budget.key

    def test_native_resolution_is_labelled(self) -> None:
        assert "native" in TrainingBudget(epochs=1).key

    def test_relative_cost_scales_with_epochs_and_data(self) -> None:
        cheap = TrainingBudget(epochs=1, train_fraction=0.5)
        expensive = TrainingBudget(epochs=4, train_fraction=1.0)
        assert expensive.relative_cost == 8 * cheap.relative_cost

    def test_round_trips_through_plain_data(self) -> None:
        budget = TrainingBudget(
            epochs=2, train_fraction=0.25, resolution=8, max_seconds=5.0, rung=2
        )
        assert TrainingBudget.from_dict(budget.to_dict()) == budget

    def test_describes_itself(self) -> None:
        text = TrainingBudget(epochs=2, train_fraction=0.5, resolution=8, max_seconds=30).describe()
        assert "2 epochs" in text
        assert "50%" in text
        assert "resolution 8" in text

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"epochs": 1, "train_fraction": 0.0},
            {"epochs": 1, "train_fraction": 1.5},
            {"epochs": 1, "resolution": 2},
            {"epochs": 1, "max_seconds": 0.0},
            {"epochs": 1, "rung": -1},
        ],
    )
    def test_fields_are_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ConfigurationError):
            TrainingBudget(**kwargs)  # type: ignore[arg-type]

    def test_deserialisation_requires_epochs(self) -> None:
        with pytest.raises(ConfigurationError, match="missing the required"):
            TrainingBudget.from_dict({})


class TestFailureClassification:
    @pytest.mark.parametrize(
        ("error", "kind", "retriable"),
        [
            (NonFiniteLossError("diverged"), FailureKind.DIVERGENCE, False),
            (ConstraintViolationError("too big"), FailureKind.CONSTRAINT, False),
            (ArchitectureValidationError("bad"), FailureKind.VALIDATION, False),
            (ModelBuildError("nope"), FailureKind.BUILD, False),
            (EvaluationTimeoutError("slow"), FailureKind.TIMEOUT, True),
            (ResourceLimitError("too big"), FailureKind.RESOURCE, False),
            (PersistenceError("locked"), FailureKind.PERSISTENCE, True),
            (WorkerError("died"), FailureKind.WORKER, True),
            (TrainingError("oops"), FailureKind.TRAINING, True),
            (MemoryError("oom"), FailureKind.RESOURCE, True),
        ],
    )
    def test_known_errors_are_classified(
        self, error: BaseException, kind: FailureKind, retriable: bool
    ) -> None:
        assert classify_failure(error) == (kind, retriable)

    def test_cuda_oom_is_recognised_by_message(self) -> None:
        kind, retriable = classify_failure(RuntimeError("CUDA out of memory"))
        assert kind is FailureKind.RESOURCE
        assert retriable

    def test_unknown_errors_are_permanent(self) -> None:
        kind, retriable = classify_failure(ValueError("mystery"))
        assert kind is FailureKind.UNKNOWN
        assert not retriable

    def test_failure_record_captures_context(self) -> None:
        failure = EvaluationFailure.from_exception(
            NonFiniteLossError("diverged", details={"epoch": 2}), traceback_text="tb"
        )
        assert failure.kind is FailureKind.DIVERGENCE
        assert failure.details["epoch"] == 2
        assert failure.traceback_text == "tb"

    def test_failure_record_round_trips(self) -> None:
        failure = EvaluationFailure.from_exception(WorkerError("died"))
        assert EvaluationFailure.from_dict(failure.to_dict()) == failure


class TestEvaluationResult:
    def test_round_trips_through_plain_data(self) -> None:
        result = EvaluationResult(
            candidate_id="c",
            architecture_hash="h",
            budget=TrainingBudget(epochs=1),
            metrics={"validation_accuracy": 0.5},
            artifacts={"weights": "w.pt"},
            artifact_bytes={"weights": 100},
            notes=("caveat",),
        )
        restored = EvaluationResult.from_dict(result.to_dict())
        assert restored.metrics == result.metrics
        assert restored.artifacts == result.artifacts
        assert restored.artifact_bytes == {"weights": 100}
        assert restored.notes == ("caveat",)

    def test_failed_result_round_trips(self) -> None:
        result = EvaluationResult(
            candidate_id="c",
            architecture_hash="h",
            budget=TrainingBudget(epochs=1),
            succeeded=False,
            failure=EvaluationFailure.from_exception(TrainingError("oops")),
        )
        restored = EvaluationResult.from_dict(result.to_dict())
        assert restored.failure is not None
        assert restored.failure.kind is FailureKind.TRAINING

    def test_primary_metric_accessor(self) -> None:
        result = EvaluationResult(
            candidate_id="c",
            architecture_hash="h",
            budget=TrainingBudget(epochs=1),
            metrics={"validation_accuracy": 0.7},
        )
        assert result.primary_metric == 0.7


class TestLatency:
    def test_reports_positive_statistics(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        measurement = measure_latency(
            model,
            input_shape=(3, 8, 8),
            warmup_iterations=1,
            timed_iterations=2,
            repeats=3,
        )
        assert measurement.median_ms > 0
        assert measurement.min_ms <= measurement.median_ms <= measurement.p99_ms
        assert measurement.repeats == 3
        assert len(measurement.samples_ms) == 3

    def test_carries_the_portability_warning(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        measurement = measure_latency(
            model, input_shape=(3, 8, 8), warmup_iterations=0, timed_iterations=1, repeats=2
        )
        assert measurement.warning == LATENCY_WARNING
        assert "device_name" in measurement.to_dict()

    def test_per_image_latency_divides_by_batch_size(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        measurement = measure_latency(
            model,
            input_shape=(3, 8, 8),
            batch_size=4,
            warmup_iterations=0,
            timed_iterations=1,
            repeats=2,
        )
        assert measurement.per_image_ms == pytest.approx(measurement.median_ms / 4)

    def test_restores_the_training_mode(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        model.train()
        measure_latency(
            model, input_shape=(3, 8, 8), warmup_iterations=0, timed_iterations=1, repeats=1
        )
        assert model.training

    def test_metrics_use_documented_keys(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        metrics = measure_latency(
            model, input_shape=(3, 8, 8), warmup_iterations=0, timed_iterations=1, repeats=1
        ).to_metrics()
        assert set(metrics) == {
            "latency_median_ms",
            "latency_mean_ms",
            "latency_p90_ms",
            "latency_p99_ms",
            "latency_per_image_ms",
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_size": 0},
            {"timed_iterations": 0},
            {"repeats": 0},
            {"warmup_iterations": -1},
        ],
    )
    def test_arguments_are_validated(self, kwargs: dict[str, int]) -> None:
        model = nn.Linear(2, 2)
        with pytest.raises(ConfigurationError):
            measure_latency(model, input_shape=(3, 8, 8), **kwargs)  # type: ignore[arg-type]

    def test_measurement_serialises(self) -> None:
        measurement = LatencyMeasurement(
            median_ms=1.0,
            mean_ms=1.0,
            std_ms=0.0,
            min_ms=1.0,
            p90_ms=1.0,
            p99_ms=1.0,
            per_image_ms=1.0,
            batch_size=1,
            input_shape=(1, 3, 8, 8),
            warmup_iterations=1,
            timed_iterations=1,
            repeats=1,
            device="cpu",
            device_name="test",
            torch_threads=1,
        )
        assert measurement.to_dict()["device"] == "cpu"


class TestModelSize:
    def test_serialised_size_exceeds_the_payload(self, manual_spec: ArchitectureSpec) -> None:
        measurement = measure_model_size(build_model(manual_spec))
        assert measurement.serialized_bytes >= measurement.state_dict_bytes
        assert measurement.overhead_bytes >= 0

    def test_counts_parameters_and_buffers_separately(self, manual_spec: ArchitectureSpec) -> None:
        measurement = measure_model_size(build_model(manual_spec))
        assert measurement.parameter_bytes > 0
        assert measurement.buffer_bytes > 0

    def test_metrics_use_documented_keys(self, manual_spec: ArchitectureSpec) -> None:
        metrics = measure_model_size(build_model(manual_spec)).to_metrics()
        assert set(metrics) == {"model_size_bytes", "parameter_bytes"}

    def test_saved_weights_reload(self, manual_spec: ArchitectureSpec, tmp_path: Path) -> None:
        model = build_model(manual_spec)
        size = save_model_weights(model, tmp_path / "w.pt")
        assert size > 0
        state = torch.load(tmp_path / "w.pt", map_location="cpu", weights_only=True)
        rebuilt = build_model(manual_spec)
        rebuilt.load_state_dict(state)
        assert all(torch.equal(state[key], value) for key, value in rebuilt.state_dict().items())

    def test_saving_leaves_no_temporary_file(
        self, manual_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        save_model_weights(build_model(manual_spec), tmp_path / "w.pt")
        assert [path.name for path in tmp_path.iterdir()] == ["w.pt"]


class TestCandidateEvaluator:
    @staticmethod
    def _evaluator(bundle: DatasetBundle, root: Path, **settings: object) -> CandidateEvaluator:
        return CandidateEvaluator(
            dataset=bundle,
            loader_settings=LoaderSettings(batch_size=32),
            training_settings=TrainingSettings(
                epochs=1, optimizer=OptimizerSettings(learning_rate=0.01), topk=2
            ),
            settings=EvaluationSettings(measure_latency=False, **settings),  # type: ignore[arg-type]
            artifact_root=root,
            device="cpu",
            seed=99,
        )

    def test_successful_evaluation_reports_metrics(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        assert result.succeeded
        assert "validation_accuracy" in result.metrics
        assert "trainable_parameters" in result.metrics
        assert result.duration_seconds > 0
        assert result.artifacts["weights"]

    def test_candidate_seed_depends_only_on_identity(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        budget = TrainingBudget(epochs=1)
        assert evaluator.candidate_seed(sample_spec, budget) == evaluator.candidate_seed(
            sample_spec, budget
        )

    def test_candidate_seed_differs_between_rungs(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        first = evaluator.candidate_seed(sample_spec, TrainingBudget(epochs=1, rung=0))
        second = evaluator.candidate_seed(sample_spec, TrainingBudget(epochs=1, rung=1))
        assert first != second

    def test_repeated_evaluation_is_reproducible(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        budget = TrainingBudget(epochs=1)
        context = EvaluationContext(candidate_id="c", trial_id="t")
        first = self._evaluator(synthetic_bundle, tmp_path / "a").evaluate(
            sample_spec, budget, context
        )
        second = self._evaluator(synthetic_bundle, tmp_path / "b").evaluate(
            sample_spec, budget, context
        )
        assert first.metrics["validation_accuracy"] == second.metrics["validation_accuracy"]

    def test_parameter_limit_prunes_before_building(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path, max_parameters=1)
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        assert not result.succeeded
        assert result.failure is not None
        assert result.failure.kind is FailureKind.RESOURCE

    def test_failures_are_returned_not_raised(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)

        class _ExplodingBuilder:
            def build(self, *args: object, **kwargs: object) -> object:
                raise ModelBuildError("synthetic build failure")

        evaluator._model_builder = _ExplodingBuilder()  # type: ignore[assignment]
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        assert not result.succeeded
        assert result.failure is not None
        assert result.failure.kind is FailureKind.BUILD
        assert result.failure.traceback_text

    def test_keyboard_interrupt_propagates(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)

        class _InterruptingBuilder:
            def build(self, *args: object, **kwargs: object) -> object:
                raise KeyboardInterrupt

        evaluator._model_builder = _InterruptingBuilder()  # type: ignore[assignment]
        with pytest.raises(KeyboardInterrupt):
            evaluator.evaluate(
                sample_spec,
                TrainingBudget(epochs=1),
                EvaluationContext(candidate_id="c", trial_id="t"),
            )

    def test_artifact_paths_stay_inside_the_root(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        for relative in result.artifacts.values():
            assert not Path(relative).is_absolute()
            assert (evaluator.artifact_root / relative).is_file()

    def test_low_fidelity_uses_less_data(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1, train_fraction=0.5),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        assert result.metrics["train_examples"] == 48

    def test_test_evaluation_needs_weights_to_exist(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        with pytest.raises(CheckpointError, match="weights file not found"):
            evaluator.evaluate_on_test(sample_spec, weights_path=tmp_path / "absent.pt")

    def test_test_evaluation_reports_test_metrics(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = self._evaluator(synthetic_bundle, tmp_path)
        result = evaluator.evaluate(
            sample_spec,
            TrainingBudget(epochs=1),
            EvaluationContext(candidate_id="c", trial_id="t"),
        )
        weights = evaluator.artifact_root / result.artifacts["weights"]
        metrics = evaluator.evaluate_on_test(sample_spec, weights_path=weights)
        assert set(metrics) == {
            "test_accuracy",
            "test_loss",
            "test_topk_accuracy",
            "test_examples",
        }
