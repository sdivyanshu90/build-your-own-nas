"""Execution backends for candidate evaluation.

The engine dispatches *tasks* and receives *results*. How those tasks actually run —
inline, or across worker processes — is behind this interface, so the engine's logic is
identical in both modes and can be tested entirely sequentially.

Two backends
------------
:class:`SequentialExecutor`
    Runs each task inline. Fully deterministic: the same configuration and seed produce
    byte-identical results, in the same order. This is the mode every reproducibility
    guarantee in this project is stated for.

:class:`ProcessPoolExecutorBackend`
    Runs tasks across worker processes. Faster on a multi-core machine, at the cost of
    determinism guarantees that are weaker in one specific and *documented* way (below).

What concurrency does and does not change
------------------------------------------
It **does not** change any individual candidate's result. Each candidate's seed is derived
from its architecture hash, so its weights, data order, and measured accuracy are
independent of which worker ran it and of what ran before it.

It **does** change:

* **Completion order.** Results arrive as workers finish, not in dispatch order. A strategy
  that adapts to observations — evolution, successive halving — therefore sees a different
  observation *sequence*, and can make different subsequent proposals.
* **Measured latency and duration.** Workers contend for cores and memory bandwidth, so
  timing metrics are noisier and systematically slower than in sequential mode.

The honest summary: **sequential execution is reproducible; multiprocessing is
repeatable in distribution but not identical run to run.** This is stated in
``docs/architecture/concurrency.md`` and enforced by the determinism tests, which only
assert bit-identity for sequential runs.

Batch semantics
---------------
Both backends expose ``run_batch``: submit a list of tasks, get back every result. This is
a barrier — the engine waits for the slowest task in a batch before proceeding. A streaming
futures interface would keep workers marginally busier, but it also makes the engine's
checkpointing and state transitions much harder to reason about, and every strategy here
already has natural batch boundaries. The trade-off is recorded in
``docs/adr/0004-concurrency-model.md``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nas_engine.architectures.canonical import to_canonical_dict
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import CandidateEvaluator, EvaluationContext
from nas_engine.evaluation.result import EvaluationFailure, EvaluationResult
from nas_engine.exceptions import WorkerError
from nas_engine.observability.logging import get_logger
from nas_engine.utilities.timing import utc_now

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationTask:
    """One unit of work handed to an executor.

    Attributes:
        candidate_id: Candidate being evaluated.
        trial_id: Identifier of this attempt.
        architecture_hash: Canonical hash, carried so a failed task can still be attributed.
        spec: Architecture to evaluate.
        budget: Resources to spend.
        attempt: Zero-based attempt number.
    """

    candidate_id: str
    trial_id: str
    architecture_hash: str
    spec: ArchitectureSpec
    budget: TrainingBudget
    attempt: int = 0

    def to_payload(self, config_payload: dict[str, Any], *, seed: int) -> dict[str, Any]:
        """Render the task as a picklable payload for a worker process.

        Args:
            config_payload: Serialised configuration.
            seed: Master seed.

        Returns:
            A plain-data payload.
        """
        return {
            "config": config_payload,
            "spec": to_canonical_dict(self.spec),
            "budget": self.budget.to_dict(),
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "architecture_hash": self.architecture_hash,
            "attempt": self.attempt,
            "seed": seed,
        }


class EvaluationExecutor(ABC):
    """Interface every execution backend implements."""

    @property
    @abstractmethod
    def max_in_flight(self) -> int:
        """Maximum tasks that may be dispatched at once."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Short name of the backend, recorded in logs and reports."""

    @abstractmethod
    def run_batch(self, tasks: list[EvaluationTask]) -> list[EvaluationResult]:
        """Run every task and return every result.

        Results are returned in the *same order as the input tasks*, regardless of the
        order in which they completed. Ordering the output makes the engine's bookkeeping
        deterministic even when execution is not.

        Args:
            tasks: Tasks to run.

        Returns:
            One result per task, in input order.
        """

    def shutdown(self) -> None:  # noqa: B027 - optional hook with a no-op default
        """Release any resources the backend holds. Safe to call more than once."""

    def __enter__(self) -> EvaluationExecutor:
        """Enter a context that shuts the backend down on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Shut the backend down."""
        self.shutdown()


class SequentialExecutor(EvaluationExecutor):
    """Runs evaluations inline in the calling process.

    Args:
        evaluator: The evaluator to use.
    """

    def __init__(self, evaluator: CandidateEvaluator) -> None:
        self._evaluator = evaluator

    @property
    def max_in_flight(self) -> int:
        """Always ``1``: tasks run one after another."""
        return 1

    @property
    def mode(self) -> str:
        """Backend name."""
        return "sequential"

    def run_batch(self, tasks: list[EvaluationTask]) -> list[EvaluationResult]:
        """Run tasks one at a time.

        Args:
            tasks: Tasks to run.

        Returns:
            Results in input order.
        """
        return [
            self._evaluator.evaluate(
                task.spec,
                task.budget,
                EvaluationContext(
                    candidate_id=task.candidate_id,
                    trial_id=task.trial_id,
                    worker_id="main",
                    attempt=task.attempt,
                ),
            )
            for task in tasks
        ]


class ProcessPoolExecutorBackend(EvaluationExecutor):
    """Runs evaluations across worker processes.

    Args:
        config_payload: Serialised configuration, sent to each worker.
        workers: Number of worker processes.
        start_method: Multiprocessing start method.
        seed: Master seed.
        max_in_flight: Cap on simultaneously dispatched tasks; defaults to ``workers``.
        task_timeout_seconds: Per-task wall-clock limit enforced by the parent. This is a
            backstop for a worker that hangs without raising; the evaluator has its own,
            tighter, in-process limit.

    Raises:
        ValueError: If ``workers`` is not positive.
    """

    def __init__(
        self,
        *,
        config_payload: dict[str, Any],
        workers: int,
        start_method: str = "spawn",
        seed: int = 42,
        max_in_flight: int | None = None,
        task_timeout_seconds: float | None = None,
    ) -> None:
        if workers < 1:
            msg = f"workers must be >= 1, received {workers}"
            raise ValueError(msg)
        self._config_payload = config_payload
        self._workers = workers
        self._start_method = start_method
        self._seed = seed
        self._max_in_flight = max_in_flight if max_in_flight is not None else workers
        self._task_timeout_seconds = task_timeout_seconds
        self._pool: Any = None

    @property
    def max_in_flight(self) -> int:
        """Maximum simultaneously dispatched tasks."""
        return self._max_in_flight

    @property
    def mode(self) -> str:
        """Backend name."""
        return f"multiprocessing[{self._workers}x{self._start_method}]"

    def _ensure_pool(self) -> Any:
        """Create the process pool on first use.

        Deferring creation means a search that never dispatches anything — because its
        strategy is already finished — does not pay for spawning interpreters.

        Returns:
            The pool.

        Raises:
            WorkerError: If the pool cannot be created.
        """
        if self._pool is not None:
            return self._pool
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        try:
            context = multiprocessing.get_context(self._start_method)
            self._pool = ProcessPoolExecutor(max_workers=self._workers, mp_context=context)
        except (OSError, ValueError) as exc:
            msg = (
                f"could not start a process pool with {self._workers} workers using the "
                f"'{self._start_method}' start method: {exc}. Set "
                "concurrency.mode='sequential' to run in-process."
            )
            raise WorkerError(
                msg,
                details={
                    "workers": self._workers,
                    "start_method": self._start_method,
                    "error": str(exc),
                },
            ) from exc
        return self._pool

    def run_batch(self, tasks: list[EvaluationTask]) -> list[EvaluationResult]:
        """Dispatch tasks to workers and collect their results.

        A worker that dies takes its task's result with it. Rather than failing the whole
        batch, the dead task becomes a retriable ``WorkerError`` failure, so the engine can
        apply its retry policy and the rest of the batch is unaffected.

        Args:
            tasks: Tasks to run.

        Returns:
            Results in input order.
        """
        if not tasks:
            return []
        pool = self._ensure_pool()
        futures = {
            pool.submit(
                _worker_entrypoint, task.to_payload(self._config_payload, seed=self._seed)
            ): index
            for index, task in enumerate(tasks)
        }
        results: list[EvaluationResult | None] = [None] * len(tasks)

        for future, index in futures.items():
            task = tasks[index]
            try:
                payload = future.result(timeout=self._task_timeout_seconds)
                results[index] = EvaluationResult.from_dict(payload)
            except (KeyboardInterrupt, SystemExit):
                for pending in futures:
                    pending.cancel()
                raise
            except BaseException as error:
                _LOGGER.error(
                    "executor.worker_failed",
                    candidate_id=task.candidate_id,
                    trial_id=task.trial_id,
                    error=str(error),
                    error_type=type(error).__name__,
                )
                worker_error = WorkerError(
                    f"worker process failed while evaluating candidate {task.candidate_id}: "
                    f"{type(error).__name__}: {error}",
                    details={"candidate_id": task.candidate_id, "error": str(error)},
                )
                results[index] = EvaluationResult(
                    candidate_id=task.candidate_id,
                    architecture_hash=task.architecture_hash,
                    budget=task.budget,
                    succeeded=False,
                    failure=EvaluationFailure.from_exception(worker_error),
                    started_at=utc_now(),
                    completed_at=utc_now(),
                    worker_id="dead",
                )

        return [result for result in results if result is not None]

    def shutdown(self) -> None:
        """Shut down the process pool, waiting for in-flight tasks."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _worker_entrypoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Module-level indirection so the pool target is picklable.

    ``ProcessPoolExecutor`` pickles the callable by qualified name, so it must be a
    module-level function — a bound method or closure cannot be dispatched.

    Args:
        payload: Task payload.

    Returns:
        The serialised evaluation result.
    """
    from nas_engine.orchestration.worker import evaluate_task

    return evaluate_task(payload)


def build_executor(
    *,
    mode: str,
    evaluator: CandidateEvaluator,
    config_payload: dict[str, Any],
    workers: int,
    start_method: str,
    seed: int,
    max_in_flight: int | None = None,
    task_timeout_seconds: float | None = None,
) -> EvaluationExecutor:
    """Construct the executor for a concurrency mode.

    Args:
        mode: ``"sequential"`` or ``"multiprocessing"``.
        evaluator: Evaluator used by the sequential backend.
        config_payload: Serialised configuration used by the multiprocessing backend.
        workers: Worker count.
        start_method: Multiprocessing start method.
        seed: Master seed.
        max_in_flight: Cap on simultaneously dispatched tasks.
        task_timeout_seconds: Parent-side per-task timeout.

    Returns:
        The executor.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode == "sequential":
        return SequentialExecutor(evaluator)
    if mode == "multiprocessing":
        cpu_count = os.cpu_count() or 1
        if workers > cpu_count:
            _LOGGER.warning(
                "executor.oversubscribed",
                workers=workers,
                cpu_count=cpu_count,
                note="workers exceed available cores; latency metrics will be inflated",
            )
        return ProcessPoolExecutorBackend(
            config_payload=config_payload,
            workers=workers,
            start_method=start_method,
            seed=seed,
            max_in_flight=max_in_flight,
            task_timeout_seconds=task_timeout_seconds,
        )
    msg = f"unknown concurrency mode {mode!r}; expected 'sequential' or 'multiprocessing'"
    raise ValueError(msg)


__all__ = [
    "EvaluationExecutor",
    "EvaluationTask",
    "ProcessPoolExecutorBackend",
    "SequentialExecutor",
    "build_executor",
]
