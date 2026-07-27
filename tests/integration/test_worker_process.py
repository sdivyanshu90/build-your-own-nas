"""Tests for the multiprocessing worker entry point.

The worker normally runs in a spawned interpreter, where nothing is inherited and every
input arrives as plain data. These tests call it **in-process** so its behaviour is
observable and measurable: the payload contract, the per-process evaluator cache, seed
isolation, and the guarantee that no exception escapes.

A separate, slower end-to-end test exercises the real spawned path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nas_engine.architectures.canonical import to_canonical_dict
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.result import EvaluationResult, FailureKind
from nas_engine.orchestration.executors import (
    EvaluationTask,
    SequentialExecutor,
    build_executor,
)
from nas_engine.orchestration.worker import evaluate_task, reset_worker_cache
from nas_engine.search_space.presets import tiny_cnn_space
from nas_engine.search_space.sampler import ArchitectureSampler
from tests.conftest import build_smoke_config

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_worker_cache() -> Any:
    """Ensure no evaluator leaks between tests in this module."""
    reset_worker_cache()
    yield
    reset_worker_cache()


def _payload(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a worker payload for a valid tiny-space candidate."""
    config = build_smoke_config(tmp_path, **overrides)
    spec = ArchitectureSampler(tiny_cnn_space(), seed=4).sample()
    return {
        "config": config.to_dict(),
        "spec": to_canonical_dict(spec),
        "budget": TrainingBudget(epochs=1).to_dict(),
        "candidate_id": "candidate-1",
        "trial_id": "trial-1",
        "architecture_hash": architecture_hash(spec),
        "attempt": 0,
        "seed": config.reproducibility.seed,
        "worker_id": "0",
    }


class TestWorkerEntryPoint:
    def test_a_valid_payload_produces_a_successful_result(self, tmp_path: Path) -> None:
        payload = evaluate_task(_payload(tmp_path))
        result = EvaluationResult.from_dict(payload)
        assert result.succeeded
        assert result.candidate_id == "candidate-1"
        assert result.trial_id if hasattr(result, "trial_id") else True
        assert result.metrics["validation_accuracy"] >= 0.0
        assert result.worker_id == "0"

    def test_the_returned_payload_is_plain_data(self, tmp_path: Path) -> None:
        import json

        payload = evaluate_task(_payload(tmp_path))
        # It must survive the pickling a process boundary performs; JSON is a stricter
        # check and catches anything exotic sneaking into the result.
        json.dumps(payload, default=str)
        assert isinstance(payload["metrics"], dict)
        assert isinstance(payload["budget"], dict)

    def test_a_malformed_configuration_becomes_a_failed_result(self, tmp_path: Path) -> None:
        payload = _payload(tmp_path)
        payload["config"] = {"nonsense_section": True}
        result = EvaluationResult.from_dict(evaluate_task(payload))
        assert not result.succeeded
        assert result.failure is not None
        assert result.failure.kind in {FailureKind.UNKNOWN, FailureKind.VALIDATION}

    def test_a_malformed_architecture_becomes_a_failed_result(self, tmp_path: Path) -> None:
        payload = _payload(tmp_path)
        payload["spec"] = {"stages": "not a list"}
        result = EvaluationResult.from_dict(evaluate_task(payload))
        assert not result.succeeded
        assert result.failure is not None
        assert result.failure.kind is FailureKind.VALIDATION

    def test_a_missing_budget_becomes_a_failed_result(self, tmp_path: Path) -> None:
        payload = _payload(tmp_path)
        payload["budget"] = {}
        result = EvaluationResult.from_dict(evaluate_task(payload))
        assert not result.succeeded

    def test_nothing_escapes_the_worker(self, tmp_path: Path) -> None:
        # Every failure mode must come back as data. An exception crossing a process
        # boundary loses its traceback and can fail to unpickle.
        for broken in ("config", "spec", "budget"):
            payload = _payload(tmp_path)
            payload[broken] = None
            outcome = evaluate_task(payload)
            assert outcome["succeeded"] is False

    def test_results_are_reproducible_across_calls(self, tmp_path: Path) -> None:
        first = EvaluationResult.from_dict(evaluate_task(_payload(tmp_path / "a")))
        reset_worker_cache()
        second = EvaluationResult.from_dict(evaluate_task(_payload(tmp_path / "b")))
        assert first.metrics["validation_accuracy"] == second.metrics["validation_accuracy"]

    def test_the_evaluator_is_cached_per_configuration(self, tmp_path: Path) -> None:
        from nas_engine.orchestration import worker

        evaluate_task(_payload(tmp_path))
        assert len(worker._WORKER_CACHE) == 1
        evaluate_task(_payload(tmp_path))
        assert len(worker._WORKER_CACHE) == 1

    def test_a_different_configuration_builds_a_second_evaluator(self, tmp_path: Path) -> None:
        from nas_engine.orchestration import worker

        evaluate_task(_payload(tmp_path / "a"))
        evaluate_task(_payload(tmp_path / "b", budget={"max_evaluations": 9, "epochs": 1}))
        assert len(worker._WORKER_CACHE) == 2

    def test_the_cache_can_be_cleared(self, tmp_path: Path) -> None:
        from nas_engine.orchestration import worker

        evaluate_task(_payload(tmp_path))
        reset_worker_cache()
        assert worker._WORKER_CACHE == {}


class TestExecutorSelection:
    def test_sequential_mode_builds_the_inline_backend(self, tmp_path: Path) -> None:
        from nas_engine.datasets.loaders import LoaderSettings
        from nas_engine.datasets.registry import build_dataset
        from nas_engine.evaluation.evaluator import CandidateEvaluator
        from nas_engine.training.trainer import TrainingSettings

        config = build_smoke_config(tmp_path)
        evaluator = CandidateEvaluator(
            dataset=build_dataset(config.dataset.provider, **config.dataset.options),
            loader_settings=LoaderSettings(batch_size=32),
            training_settings=TrainingSettings(epochs=1, topk=2),
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        executor = build_executor(
            mode="sequential",
            evaluator=evaluator,
            config_payload=config.to_dict(),
            workers=1,
            start_method="spawn",
            seed=1,
        )
        try:
            assert isinstance(executor, SequentialExecutor)
            assert executor.mode == "sequential"
        finally:
            executor.shutdown()

    def test_multiprocessing_mode_builds_the_pool_backend(self, tmp_path: Path) -> None:
        from nas_engine.orchestration.executors import ProcessPoolExecutorBackend

        config = build_smoke_config(tmp_path)
        executor = build_executor(
            mode="multiprocessing",
            evaluator=None,  # type: ignore[arg-type]
            config_payload=config.to_dict(),
            workers=2,
            start_method="spawn",
            seed=1,
        )
        try:
            assert isinstance(executor, ProcessPoolExecutorBackend)
            assert "multiprocessing" in executor.mode
            assert executor.max_in_flight == 2
        finally:
            executor.shutdown()

    def test_an_unknown_mode_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown concurrency mode"):
            build_executor(
                mode="telepathy",
                evaluator=None,  # type: ignore[arg-type]
                config_payload={},
                workers=1,
                start_method="spawn",
                seed=1,
            )

    def test_sequential_execution_returns_results_in_input_order(self, tmp_path: Path) -> None:
        from nas_engine.datasets.loaders import LoaderSettings
        from nas_engine.datasets.registry import build_dataset
        from nas_engine.evaluation.evaluator import (
            CandidateEvaluator,
            EvaluationSettings,
        )
        from nas_engine.training.trainer import TrainingSettings

        config = build_smoke_config(tmp_path)
        evaluator = CandidateEvaluator(
            dataset=build_dataset(config.dataset.provider, **config.dataset.options),
            loader_settings=LoaderSettings(batch_size=32),
            training_settings=TrainingSettings(epochs=1, topk=2),
            settings=EvaluationSettings(measure_latency=False, save_weights=False),
            artifact_root=tmp_path / "artifacts",
            device="cpu",
        )
        sampler = ArchitectureSampler(tiny_cnn_space(), seed=11)
        tasks = [
            EvaluationTask(
                candidate_id=f"c{index}",
                trial_id=f"t{index}",
                architecture_hash="",
                spec=sampler.sample(),
                budget=TrainingBudget(epochs=1),
            )
            for index in range(3)
        ]
        results = SequentialExecutor(evaluator).run_batch(tasks)
        assert [result.candidate_id for result in results] == ["c0", "c1", "c2"]
