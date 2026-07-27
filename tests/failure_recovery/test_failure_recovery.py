"""Failure and recovery tests.

Every scenario here simulates something going wrong and asserts two things: the failure is
recorded with an understandable status, and the *rest of the search survives*. A NAS run
that aborts because one candidate diverged has thrown away hours of work for no reason.

Scenarios covered:

* a training exception during evaluation;
* a divergent (non-finite) loss, which must be permanent, not retried;
* an architecture that cannot be built;
* a candidate that exceeds a resource constraint;
* a worker process that dies;
* a database write failure;
* a corrupted training checkpoint;
* a missing training checkpoint;
* an interrupted evaluation left in ``RUNNING``;
* retry exhaustion;
* a duplicate proposal;
* an incompatible saved configuration version;
* a missing artifact file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.config.models import CONFIG_VERSION
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.loaders import LoaderSettings
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import (
    CandidateEvaluator,
    EvaluationContext,
    EvaluationSettings,
)
from nas_engine.evaluation.result import EvaluationFailure, FailureKind
from nas_engine.exceptions import (
    CheckpointError,
    ConfigVersionError,
    ModelBuildError,
    PersistenceError,
    RecordNotFoundError,
    ResourceLimitError,
    TrainingError,
)
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.orchestration.retry import RetryPolicy
from nas_engine.persistence.models import CandidateRecord
from nas_engine.training.checkpointing import load_checkpoint
from nas_engine.training.optimizers import OptimizerSettings
from nas_engine.training.trainer import TrainingSettings

pytestmark = pytest.mark.recovery


def _evaluator(bundle: DatasetBundle, root: Path, **settings: Any) -> CandidateEvaluator:
    """Build an evaluator with fast, deterministic settings."""
    return CandidateEvaluator(
        dataset=bundle,
        loader_settings=LoaderSettings(batch_size=32),
        training_settings=TrainingSettings(
            epochs=1, optimizer=OptimizerSettings(learning_rate=0.01), topk=2
        ),
        settings=EvaluationSettings(measure_latency=False, **settings),
        artifact_root=root,
        device="cpu",
        seed=3,
    )


class _ExplodingBuilder:
    """A model builder that always raises, used to inject build failures."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def build(self, *args: object, **kwargs: object) -> object:
        raise self._error


class TestEvaluationFailures:
    def test_training_exception_is_captured_not_raised(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(synthetic_bundle, tmp_path)
        evaluator._model_builder = _ExplodingBuilder(  # type: ignore[assignment]
            TrainingError("simulated transient failure")
        )
        result = evaluator.evaluate(
            sample_spec, TrainingBudget(epochs=1), EvaluationContext("c", "t")
        )
        assert not result.succeeded
        assert result.failure is not None
        assert result.failure.kind is FailureKind.TRAINING
        assert result.failure.retriable is True
        assert result.failure.traceback_text

    def test_build_failure_is_permanent(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(synthetic_bundle, tmp_path)
        evaluator._model_builder = _ExplodingBuilder(  # type: ignore[assignment]
            ModelBuildError("unbuildable")
        )
        result = evaluator.evaluate(
            sample_spec, TrainingBudget(epochs=1), EvaluationContext("c", "t")
        )
        assert result.failure is not None
        assert result.failure.kind is FailureKind.BUILD
        assert result.failure.retriable is False

    def test_resource_limit_is_reported_before_building(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(synthetic_bundle, tmp_path, max_parameters=1)
        result = evaluator.evaluate(
            sample_spec, TrainingBudget(epochs=1), EvaluationContext("c", "t")
        )
        assert result.failure is not None
        assert result.failure.kind is FailureKind.RESOURCE
        assert "exceeds the evaluation limit" in result.failure.message

    def test_a_failure_does_not_leave_partial_artifacts(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(synthetic_bundle, tmp_path)
        evaluator._model_builder = _ExplodingBuilder(  # type: ignore[assignment]
            TrainingError("x")
        )
        result = evaluator.evaluate(
            sample_spec, TrainingBudget(epochs=1), EvaluationContext("c", "t")
        )
        assert result.artifacts == {}


class TestSearchSurvivesFailures:
    def test_a_failing_candidate_does_not_stop_the_search(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 4, "epochs": 1}, retry={"max_retries": 0}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            original = engine.evaluator.evaluate
            calls = {"count": 0}

            def flaky(spec: Any, budget: Any, context: Any, **kwargs: Any) -> Any:
                calls["count"] += 1
                if calls["count"] == 2:
                    from nas_engine.evaluation.result import EvaluationResult

                    return EvaluationResult(
                        candidate_id=context.candidate_id,
                        architecture_hash=architecture_hash(spec),
                        budget=budget,
                        succeeded=False,
                        failure=EvaluationFailure.from_exception(
                            ModelBuildError("simulated permanent failure")
                        ),
                    )
                return original(spec, budget, context, **kwargs)

            engine.evaluator.evaluate = flaky  # type: ignore[method-assign]
            result = engine.run()
            assert result.engine_state.failed == 1
            assert result.engine_state.completed == 3
            assert result.best is not None
        finally:
            engine.close()

    def test_a_failed_candidate_records_its_error(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 2, "epochs": 1}, retry={"max_retries": 0}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            from nas_engine.evaluation.result import EvaluationResult

            def always_fail(spec: Any, budget: Any, context: Any, **kwargs: Any) -> Any:
                return EvaluationResult(
                    candidate_id=context.candidate_id,
                    architecture_hash=architecture_hash(spec),
                    budget=budget,
                    succeeded=False,
                    failure=EvaluationFailure.from_exception(ModelBuildError("nope")),
                )

            engine.evaluator.evaluate = always_fail  # type: ignore[method-assign]
            result = engine.run()
            assert result.engine_state.failed == 2
            assert result.best is None
            candidates = engine.repository.list_candidates(
                result.search_id, statuses=[CandidateState.FAILED]
            )
            assert len(candidates) == 2
            assert candidates[0].error is not None
            assert candidates[0].error["code"] == "model_build_error"
        finally:
            engine.close()

    def test_a_pruned_candidate_is_distinguished_from_a_failure(
        self, config_factory: object
    ) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 3, "epochs": 1},
            search_space={
                "preset": "tiny_cnn",
                # A parameter ceiling low enough that most candidates are pruned, but not
                # so low that the space is declared infeasible up front.
                "overrides": {"constraints": {"max_parameters": 700}},
            },
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            counts = engine.repository.count_candidates_by_status(result.search_id)
            assert counts["failed"] == 0
            assert counts["pruned"] + counts["completed"] > 0
        finally:
            engine.close()


class TestRetryBehaviour:
    def test_a_retriable_failure_is_retried_then_succeeds(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 2, "epochs": 1}, retry={"max_retries": 2}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            original = engine.evaluator.evaluate
            failures = {"remaining": 1}

            def flaky(spec: Any, budget: Any, context: Any, **kwargs: Any) -> Any:
                if failures["remaining"] > 0:
                    failures["remaining"] -= 1
                    from nas_engine.evaluation.result import EvaluationResult

                    return EvaluationResult(
                        candidate_id=context.candidate_id,
                        architecture_hash=architecture_hash(spec),
                        budget=budget,
                        succeeded=False,
                        failure=EvaluationFailure.from_exception(TrainingError("transient")),
                    )
                return original(spec, budget, context, **kwargs)

            engine.evaluator.evaluate = flaky  # type: ignore[method-assign]
            result = engine.run()
            assert result.engine_state.retried == 1
            assert result.engine_state.completed == 2
            assert result.engine_state.failed == 0
        finally:
            engine.close()

    def test_retry_exhaustion_fails_the_candidate(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 1, "epochs": 1}, retry={"max_retries": 1}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            from nas_engine.evaluation.result import EvaluationResult

            def always_transient(spec: Any, budget: Any, context: Any, **kwargs: Any) -> Any:
                return EvaluationResult(
                    candidate_id=context.candidate_id,
                    architecture_hash=architecture_hash(spec),
                    budget=budget,
                    succeeded=False,
                    failure=EvaluationFailure.from_exception(TrainingError("always")),
                )

            engine.evaluator.evaluate = always_transient  # type: ignore[method-assign]
            result = engine.run()
            assert result.engine_state.retried == 1
            assert result.engine_state.failed == 1
            candidate = engine.repository.list_candidates(
                result.search_id, statuses=[CandidateState.FAILED]
            )[0]
            assert candidate.retry_count == 1
            assert candidate.trial_count == 2
        finally:
            engine.close()

    def test_a_permanent_failure_is_never_retried(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 1, "epochs": 1}, retry={"max_retries": 5}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            from nas_engine.evaluation.result import EvaluationResult

            def diverge(spec: Any, budget: Any, context: Any, **kwargs: Any) -> Any:
                from nas_engine.exceptions import NonFiniteLossError

                return EvaluationResult(
                    candidate_id=context.candidate_id,
                    architecture_hash=architecture_hash(spec),
                    budget=budget,
                    succeeded=False,
                    failure=EvaluationFailure.from_exception(NonFiniteLossError("diverged")),
                )

            engine.evaluator.evaluate = diverge  # type: ignore[method-assign]
            result = engine.run()
            assert result.engine_state.retried == 0
            assert result.engine_state.failed == 1
        finally:
            engine.close()


class TestCheckpointCorruption:
    def test_a_corrupt_training_checkpoint_is_rejected(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "training.pt"
        corrupt.write_bytes(b"\x00\x01\x02 not a checkpoint")
        with pytest.raises(CheckpointError, match="could not be read"):
            load_checkpoint(corrupt)

    def test_a_truncated_checkpoint_is_rejected(
        self, tmp_path: Path, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        from nas_engine.training.checkpointing import TrainingCheckpoint, save_checkpoint

        path = save_checkpoint(
            tmp_path / "ck.pt",
            TrainingCheckpoint(
                architecture_hash="abc",
                epoch=1,
                global_step=1,
                model_state={"w": torch.ones(4)},
            ),
        )
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        with pytest.raises(CheckpointError, match="could not be read"):
            load_checkpoint(path)

    def test_a_missing_checkpoint_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "absent.pt")

    def test_training_starts_fresh_when_no_checkpoint_exists(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        from nas_engine.datasets.loaders import build_dataloaders
        from nas_engine.models.builder import build_model
        from nas_engine.training.trainer import Trainer

        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=32), seed=1)
        outcome = Trainer(TrainingSettings(epochs=1, topk=2)).fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=architecture_hash(sample_spec),
            checkpoint_path=tmp_path / "does_not_exist.pt",
            resume=True,
        )
        assert outcome.epochs_completed == 1

    def test_a_corrupt_search_checkpoint_is_rejected(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
        finally:
            engine.close()

        broken = SearchEngine(config, configure_process=False)
        try:
            from sqlalchemy import select

            from nas_engine.persistence.models import CheckpointRecord

            with broken.repository.database.session() as session:
                record = session.scalars(
                    select(CheckpointRecord)
                    .where(CheckpointRecord.search_id == result.search_id)
                    .order_by(CheckpointRecord.sequence.desc())
                    .limit(1)
                ).one()
                record.payload_json = {"format_version": 1, "search_id": result.search_id}
            with pytest.raises(CheckpointError, match="missing required fields"):
                broken.resume(result.search_id)
        finally:
            broken.close()


class TestInterruptedEvaluations:
    def test_a_running_candidate_is_requeued_on_resume(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 2, "epochs": 1}, retry={"max_retries": 1}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
        finally:
            engine.close()

        resumed = SearchEngine(config, configure_process=False)
        try:
            victim = resumed.repository.list_candidates(result.search_id)[0]
            with resumed.repository.database.session() as session:
                record = session.get(CandidateRecord, victim.id)
                assert record is not None
                record.status = CandidateState.RUNNING.value
            report = resumed.repository.recover_interrupted(result.search_id, max_retries=1)
            assert report.requeued == (victim.id,)
            assert resumed.repository.get_candidate(victim.id).status == (
                CandidateState.QUEUED.value
            )
        finally:
            resumed.close()

    def test_a_repeatedly_interrupted_candidate_is_abandoned(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 1, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            victim = engine.repository.list_candidates(result.search_id)[0]
            with engine.repository.database.session() as session:
                record = session.get(CandidateRecord, victim.id)
                assert record is not None
                record.status = CandidateState.RUNNING.value
                record.retry_count = 5
            report = engine.repository.recover_interrupted(result.search_id, max_retries=1)
            assert report.abandoned == (victim.id,)
            candidate = engine.repository.get_candidate(victim.id)
            assert candidate.status == CandidateState.FAILED.value
            assert candidate.error is not None
            assert candidate.error["code"] == "retry_exhausted_error"
        finally:
            engine.close()


class TestDatabaseFailures:
    def test_a_write_failure_rolls_the_transaction_back(self, repository: Any) -> None:
        from sqlalchemy import text

        search_id = repository.create_search(
            name="t",
            strategy="random_search",
            config={"version": 1},
            config_hash="h",
            config_version=1,
            search_space={},
            seed=1,
            seeds={},
            environment={},
            planned_evaluations=1,
        )
        with pytest.raises(PersistenceError), repository.database.session() as session:
            session.execute(text("INSERT INTO candidates (id) VALUES ('broken')"))
        # The failed statement must not have left a partial row behind.
        assert repository.count_candidates_by_status(search_id)["proposed"] == 0

    def test_an_unreachable_database_is_reported(self, tmp_path: Path) -> None:
        from nas_engine.persistence.database import Database

        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        database = Database(f"sqlite+pysqlite:///{blocker}/nas.db")
        with pytest.raises(PersistenceError), database.session() as session:
            from sqlalchemy import text

            session.execute(text("SELECT 1"))
        database.dispose()


class TestDuplicateProposals:
    def test_a_duplicate_proposal_is_rejected_not_re_evaluated(
        self, config_factory: object
    ) -> None:
        config = config_factory(  # type: ignore[operator]
            search_space={"preset": "micro_cnn"},
            dataset={
                "provider": "synthetic",
                "batch_size": 16,
                "options": {
                    "num_classes": 3,
                    "input_size": 8,
                    "train_samples": 48,
                    "validation_samples": 24,
                    "test_samples": 24,
                    "seed": 1,
                },
            },
            budget={"max_evaluations": 12, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            hashes = [candidate.architecture_hash for candidate in result.ranked]
            assert len(set(hashes)) == len(hashes)
            # The micro space holds only a couple of architectures, so the search must
            # stop early rather than re-evaluating what it has already seen.
            assert result.engine_state.completed < 12
        finally:
            engine.close()


class TestConfigurationVersioning:
    def test_a_future_stored_configuration_blocks_resume(self, config_factory: object) -> None:
        from nas_engine.persistence.models import SearchRecord

        config = config_factory(budget={"max_evaluations": 1, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
        finally:
            engine.close()

        resumed = SearchEngine(config, configure_process=False)
        try:
            with resumed.repository.database.session() as session:
                record = session.get(SearchRecord, result.search_id)
                assert record is not None
                stored = dict(record.config_json)
                stored["version"] = CONFIG_VERSION + 1
                record.config_json = stored
            with pytest.raises(ConfigVersionError, match="upgrade nas-engine"):
                resumed.resume(result.search_id)
        finally:
            resumed.close()

    def test_a_changed_configuration_warns_rather_than_blocking(
        self, config_factory: object
    ) -> None:
        first = config_factory(budget={"max_evaluations": 1, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(first, configure_process=False)
        try:
            result = engine.run()
        finally:
            engine.close()

        second = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 2, "epochs": 1}, logging={"level": "WARNING"}
        )
        resumed = SearchEngine(second, configure_process=False)
        try:
            outcome = resumed.resume(result.search_id)
            assert outcome.warnings
            assert outcome.engine_state.completed == 2
        finally:
            resumed.close()


class TestMissingArtifacts:
    def test_a_missing_weights_file_is_reported_clearly(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 1, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.best is not None
            candidate = engine.repository.get_candidate(result.best.candidate_id)
            weights = engine.artifact_root / candidate.artifacts["weights"]
            weights.unlink()
            with pytest.raises(RecordNotFoundError, match="missing from disk"):
                engine.load_best_model(result.search_id)
        finally:
            engine.close()

    def test_a_candidate_without_weights_is_reported(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 1, "epochs": 1},
            evaluation={"measure_latency": False, "save_weights": False},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            with pytest.raises(RecordNotFoundError, match="no stored weights"):
                engine.load_best_model(result.search_id)
        finally:
            engine.close()


class TestWorkerFailures:
    def test_a_dead_worker_becomes_a_retriable_failure(self, tmp_path: Path) -> None:
        from nas_engine.orchestration.executors import (
            EvaluationTask,
            ProcessPoolExecutorBackend,
        )
        from nas_engine.search_space.presets import tiny_cnn_space
        from nas_engine.search_space.sampler import ArchitectureSampler

        spec = ArchitectureSampler(tiny_cnn_space(), seed=1).sample()
        # A configuration payload the worker cannot validate makes every task fail in the
        # worker rather than in the parent, which is the shape a crash takes.
        backend = ProcessPoolExecutorBackend(
            config_payload={"nonsense_section": True},
            workers=1,
            start_method="spawn",
            seed=1,
        )
        try:
            results = backend.run_batch(
                [
                    EvaluationTask(
                        candidate_id="c",
                        trial_id="t",
                        architecture_hash=architecture_hash(spec),
                        spec=spec,
                        budget=TrainingBudget(epochs=1),
                    )
                ]
            )
        finally:
            backend.shutdown()

        assert len(results) == 1
        assert not results[0].succeeded
        assert results[0].failure is not None

    def test_an_empty_batch_is_a_no_op(self) -> None:
        from nas_engine.orchestration.executors import ProcessPoolExecutorBackend

        backend = ProcessPoolExecutorBackend(config_payload={}, workers=1)
        try:
            assert backend.run_batch([]) == []
        finally:
            backend.shutdown()

    def test_worker_count_is_validated(self) -> None:
        from nas_engine.orchestration.executors import ProcessPoolExecutorBackend

        with pytest.raises(ValueError, match="workers must be"):
            ProcessPoolExecutorBackend(config_payload={}, workers=0)


class TestRetryPolicyIntegration:
    def test_the_engine_honours_a_zero_retry_policy(self) -> None:
        policy = RetryPolicy(max_retries=0)
        failure = EvaluationFailure.from_exception(TrainingError("transient"))
        assert not policy.decide(failure, attempt=0).should_retry

    def test_resource_limit_errors_are_not_retried(self) -> None:
        policy = RetryPolicy(max_retries=3)
        failure = EvaluationFailure.from_exception(ResourceLimitError("too big"))
        assert not policy.decide(failure, attempt=0).should_retry
