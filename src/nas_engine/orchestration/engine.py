"""The search engine: the orchestrator that owns the candidate lifecycle.

Everything else in this project does one job well. The engine is what makes them a system.

The loop
--------
.. code-block:: text

    while not stopped:
        proposals = strategy.propose(free_slots)
        for proposal in proposals:
            validate → hash → deduplicate → persist → queue
        tasks   = build tasks from queued candidates
        results = executor.run_batch(tasks)
        for result in results:
            persist trial, metrics, artifacts
            transition candidate state
            apply retry policy on failure
            notify strategy
        checkpoint if due
        evaluate stopping conditions

Design decisions worth stating
-------------------------------
**The engine, not the strategy, owns identity.** Hashing, duplicate detection, and
persistence live here. A strategy that had to remember what it had already proposed across
a resume would need its own database, and every new strategy would reimplement it.

**Failure is data, not control flow.** The evaluator never raises for a candidate-level
problem; it returns a failed result. The engine therefore has exactly one path through the
loop, and a failing candidate cannot abort the search.

**Checkpoint after processing, not before.** A checkpoint is only useful if the state it
describes is already durable. Writing one before the results are persisted would produce a
checkpoint that claims work is done which the database does not record.

**Recovery is explicit.** On resume the engine sweeps for candidates left in ``RUNNING``
by a dead process and returns them to the queue (or fails them if retries are exhausted)
*before* asking the strategy for anything new.

Stopping conditions, checked in order
--------------------------------------
1. Evaluation budget exhausted.
2. Wall-clock limit reached.
3. Strategy reports it is finished, with nothing in flight.
4. Strategy can propose nothing new and nothing is in flight — the space is exhausted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.config.loader import check_config_compatibility
from nas_engine.config.models import SearchConfig
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.registry import build_dataset
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import CandidateEvaluator
from nas_engine.evaluation.result import EvaluationResult
from nas_engine.exceptions import (
    DuplicateRecordError,
    OrchestrationError,
    RecordNotFoundError,
)
from nas_engine.models.builder import ModelBuilder
from nas_engine.objectives.constraints import ConstraintSet
from nas_engine.objectives.objective import ObjectiveSet
from nas_engine.objectives.online import online_objective_value
from nas_engine.objectives.ranking import RankingResult, rank_candidates
from nas_engine.observability.context import candidate_context, search_context
from nas_engine.observability.events import Event, emit
from nas_engine.observability.logging import configure_logging, get_logger
from nas_engine.observability.metrics import CounterRegistry
from nas_engine.orchestration.checkpoint import EngineState, SearchCheckpoint
from nas_engine.orchestration.executors import EvaluationExecutor, EvaluationTask, build_executor
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.orchestration.result import SearchResult, StopReason
from nas_engine.orchestration.retry import RetryPolicy
from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import ensure_schema
from nas_engine.persistence.models import SearchStatus
from nas_engine.persistence.repository import SearchRepository
from nas_engine.search.registry import build_strategy
from nas_engine.search.strategy import Observation, Proposal, SearchStrategy
from nas_engine.search_space.space import SearchSpace
from nas_engine.search_space.validation import check_architecture
from nas_engine.utilities.determinism import configure_determinism
from nas_engine.utilities.environment import collect_environment
from nas_engine.utilities.paths import ensure_directory
from nas_engine.utilities.seeding import SeedBundle, seed_everything
from nas_engine.utilities.timing import Stopwatch

_LOGGER = get_logger(__name__)


class SearchEngine:
    """Runs a neural architecture search end to end.

    The advertised usage is three lines:

    .. code-block:: python

        config = SearchConfig.from_yaml("configs/random_search.yaml")
        engine = SearchEngine(config)
        result = engine.run()

    Every collaborator can also be injected, which is what makes the engine testable
    without a real dataset or a real database:

    .. code-block:: python

        engine = SearchEngine(config, dataset=tiny_bundle, database=Database.in_memory())

    Args:
        config: Validated configuration.
        dataset: Pre-built dataset bundle; built from configuration when omitted.
        database: Database handle; built from configuration when omitted.
        strategy: Search strategy; built from configuration when omitted.
        evaluator: Candidate evaluator; built from configuration when omitted.
        search_space: Search space; built from configuration when omitted.
        configure_process: Whether to configure process-wide logging, seeding, determinism,
            and thread counts. Tests disable this so the engine cannot reconfigure the test
            runner's logging.

    Raises:
        ConfigurationError: If the configuration cannot be realised, for example because a
            requested accelerator is unavailable.
        DatasetError: If the dataset cannot be built.
    """

    def __init__(
        self,
        config: SearchConfig,
        *,
        dataset: DatasetBundle | None = None,
        database: Database | None = None,
        strategy: SearchStrategy | None = None,
        evaluator: CandidateEvaluator | None = None,
        search_space: SearchSpace | None = None,
        configure_process: bool = True,
    ) -> None:
        self.config = config
        self._owns_database = database is None
        self._configure_process = configure_process

        if configure_process:
            configure_logging(
                level=config.logging.level,
                log_format=config.logging.format,
                log_file=config.log_file,
                force=True,
            )
            self._determinism = configure_determinism(
                enabled=config.reproducibility.deterministic,
                warn_only=config.reproducibility.warn_only,
            )
            if config.hardware.torch_threads is not None:
                torch.set_num_threads(config.hardware.torch_threads)
        else:
            self._determinism = configure_determinism(
                enabled=False, warn_only=config.reproducibility.warn_only
            )

        self._seeds: SeedBundle = seed_everything(config.reproducibility.seed)
        self._device = config.hardware.resolve_device()

        ensure_directory(config.output_dir)
        self._artifact_root = ensure_directory(config.artifact_dir)
        ensure_directory(config.report_dir)

        self._dataset = dataset if dataset is not None else self._build_dataset()
        self._space = (
            search_space
            if search_space is not None
            else config.search_space.build(
                input_size=self._dataset.input_size,
                num_classes=self._dataset.num_classes,
                input_channels=self._dataset.input_channels,
            )
        )
        self._space.require_non_empty()

        self._database = (
            database
            if database is not None
            else Database(config.database_url, echo=config.persistence.echo_sql)
        )
        ensure_schema(self._database)
        self._repository = SearchRepository(self._database)

        self._objectives: ObjectiveSet = config.objectives.build_objectives()
        self._constraints: ConstraintSet = config.objectives.build_constraints()
        self._retry_policy = RetryPolicy(
            max_retries=config.retry.max_retries,
            retry_on_timeout=config.retry.retry_on_timeout,
            retry_on_resource_error=config.retry.retry_on_resource_error,
            backoff_seconds=config.retry.backoff_seconds,
            backoff_multiplier=config.retry.backoff_multiplier,
            max_backoff_seconds=config.retry.max_backoff_seconds,
        )

        self._evaluator = evaluator if evaluator is not None else self._build_evaluator()
        self._strategy = strategy if strategy is not None else self._build_strategy()

        self._state = EngineState()
        self._metrics = CounterRegistry()
        self._search_id: str | None = None
        self._warnings: list[str] = []
        # Specifications for candidates that are queued or in flight. Kept in memory so a
        # retry does not have to re-read and re-validate the specification from the
        # database on every attempt.
        self._pending_specs: dict[str, ArchitectureSpec] = {}
        self._pending_parents: dict[str, str | None] = {}
        self._attempts: dict[str, int] = {}

    # ------------------------------------------------------------------ builders --
    def _build_dataset(self) -> DatasetBundle:
        """Construct the dataset bundle from configuration."""
        bundle = build_dataset(self.config.dataset.provider, **self.config.dataset.options)
        _LOGGER.info("engine.dataset_ready", dataset=bundle.summary())
        return bundle

    def _build_evaluator(self) -> CandidateEvaluator:
        """Construct the candidate evaluator from configuration."""
        return CandidateEvaluator(
            dataset=self._dataset,
            loader_settings=self.config.dataset.build_loader_settings(),
            training_settings=self.config.training.build(epochs=self.config.budget.epochs),
            settings=self.config.evaluation.build(
                max_seconds=self.config.budget.max_seconds_per_evaluation
            ),
            artifact_root=self._artifact_root,
            device=self._device,
            seed=self.config.reproducibility.seed,
            model_builder=ModelBuilder(zero_init_residual=self.config.training.zero_init_residual),
        )

    def _build_strategy(self) -> SearchStrategy:
        """Construct the search strategy from configuration."""
        return build_strategy(
            self.config.algorithm.name,
            space=self._space,
            seed=self._seeds.strategy,
            budget=self.config.budget.build_budget(),
            max_evaluations=self.config.budget.max_evaluations,
            native_resolution=self._dataset.input_size,
            params=self.config.algorithm.params,
        )

    def _build_executor(self) -> EvaluationExecutor:
        """Construct the execution backend from configuration."""
        return build_executor(
            mode=self.config.concurrency.mode,
            evaluator=self._evaluator,
            config_payload=self.config.to_dict(),
            workers=self.config.concurrency.workers,
            start_method=self.config.concurrency.start_method,
            seed=self.config.reproducibility.seed,
            max_in_flight=self.config.concurrency.effective_in_flight,
            task_timeout_seconds=self.config.budget.max_seconds_per_evaluation,
        )

    # ---------------------------------------------------------------- properties --
    @property
    def repository(self) -> SearchRepository:
        """The repository this engine writes through."""
        return self._repository

    @property
    def strategy(self) -> SearchStrategy:
        """The active search strategy."""
        return self._strategy

    @property
    def evaluator(self) -> CandidateEvaluator:
        """The candidate evaluator, exposed so the CLI can run a held-out test pass."""
        return self._evaluator

    @property
    def objectives(self) -> ObjectiveSet:
        """The objective set used for ranking."""
        return self._objectives

    @property
    def search_space(self) -> SearchSpace:
        """The space being searched."""
        return self._space

    @property
    def dataset(self) -> DatasetBundle:
        """The dataset in use."""
        return self._dataset

    @property
    def device(self) -> torch.device:
        """The device evaluations run on."""
        return self._device

    @property
    def search_id(self) -> str | None:
        """The identifier of the current or most recent run."""
        return self._search_id

    @property
    def artifact_root(self) -> Path:
        """The validated artifact root."""
        return self._artifact_root

    # ------------------------------------------------------------------- session --
    def _create_search(self) -> str:
        """Insert a new search record and return its identifier."""
        environment = collect_environment()
        return self._repository.create_search(
            name=self.config.project.name,
            strategy=self.config.algorithm.name,
            config=self.config.to_dict(),
            config_hash=self.config.config_hash(),
            config_version=self.config.version,
            search_space=self._space.model_dump(mode="json"),
            seed=self.config.reproducibility.seed,
            seeds=self._seeds.to_dict(),
            environment={
                **environment.to_dict(),
                "determinism": self._determinism.to_dict(),
            },
            planned_evaluations=self.config.budget.max_evaluations,
        )

    def _restore(self, search_id: str) -> bool:
        """Restore strategy and engine state from the latest checkpoint.

        Args:
            search_id: Search to restore.

        Returns:
            ``True`` when a checkpoint was found and applied.

        Raises:
            CheckpointError: If the checkpoint is corrupt or belongs to another strategy.
        """
        payload = self._repository.latest_checkpoint(search_id)
        if payload is None:
            _LOGGER.warning(
                "engine.no_checkpoint",
                search_id=search_id,
                note="resuming from database state only; the strategy restarts its plan",
            )
            self._warnings.append(
                "no strategy checkpoint was found for this search; the strategy restarted "
                "from its initial state and may re-propose architectures that are already "
                "recorded (they will be skipped as duplicates)"
            )
            return False

        checkpoint = SearchCheckpoint.from_payload(payload)
        self._warnings.extend(
            checkpoint.validate_for(
                strategy_name=self.config.algorithm.name,
                config_hash=self.config.config_hash(),
            )
        )
        self._strategy.load_state_dict(checkpoint.strategy_state)
        self._state = checkpoint.engine_state
        emit(
            Event.CHECKPOINT_RESTORED,
            search_id=search_id,
            created_at=checkpoint.created_at,
            proposed=self._state.proposed,
            completed=self._state.completed,
        )
        return True

    def _checkpoint(self, search_id: str) -> None:
        """Persist strategy and engine state."""
        checkpoint = SearchCheckpoint(
            search_id=search_id,
            strategy_name=self._strategy.name,
            strategy_state=self._strategy.state_dict(),
            engine_state=self._state,
            config_hash=self.config.config_hash(),
        )
        sequence = self._repository.save_checkpoint(search_id, checkpoint.to_payload())
        self._repository.prune_checkpoints(search_id, keep=self.config.persistence.keep_checkpoints)
        emit(Event.CHECKPOINT_SAVED, search_id=search_id, sequence=sequence)

    # ------------------------------------------------------------------ proposals --
    def _accept_proposal(self, search_id: str, proposal: Proposal) -> str | None:
        """Validate, deduplicate, and persist one proposal.

        Args:
            search_id: Owning search.
            proposal: The proposal to process.

        Returns:
            The candidate id when the proposal was accepted and queued, otherwise ``None``.
        """
        self._state.proposed += 1
        hash_value = architecture_hash(proposal.spec)
        rung = proposal.budget.rung

        emit(
            Event.CANDIDATE_PROPOSED,
            search_id=search_id,
            architecture_hash=hash_value,
            rung=rung,
            origin=proposal.origin,
            mutation=proposal.mutation,
        )

        existing = self._repository.find_candidate(search_id, hash_value, rung=rung)
        if existing is not None:
            self._state.duplicates += 1
            self._metrics.increment("candidates.duplicate")
            self._strategy.on_duplicate(hash_value)
            emit(
                Event.CANDIDATE_DUPLICATE,
                search_id=search_id,
                architecture_hash=hash_value,
                rung=rung,
                existing_candidate_id=existing.id,
                existing_status=existing.status,
            )
            return None

        report = check_architecture(proposal.spec, self._space)
        try:
            candidate_id = self._repository.add_candidate(
                search_id=search_id,
                architecture_hash=hash_value,
                spec=proposal.spec,
                rung=rung,
                status=CandidateState.PROPOSED,
                parent_id=proposal.parent_id,
                mutation=proposal.mutation,
                origin=proposal.origin,
                generation=proposal.metadata.get("generation"),
                metadata={**proposal.metadata, "budget": proposal.budget.to_dict()},
            )
        except DuplicateRecordError:
            # Lost a race with another writer between the lookup above and this insert.
            # The unique constraint is the authority; treat it as a duplicate.
            self._state.duplicates += 1
            self._strategy.on_duplicate(hash_value)
            emit(
                Event.CANDIDATE_DUPLICATE,
                search_id=search_id,
                architecture_hash=hash_value,
                rung=rung,
                note="lost an insert race",
            )
            return None

        if not report.is_valid:
            terminal = (
                CandidateState.PRUNED
                if report.only_constraint_violations
                else CandidateState.FAILED
            )
            if terminal is CandidateState.PRUNED:
                self._state.pruned += 1
                self._metrics.increment("candidates.pruned")
            else:
                self._state.invalid += 1
                self._metrics.increment("candidates.invalid")
            self._repository.update_candidate_state(
                candidate_id,
                terminal,
                reason=report.summary(),
                error={"issues": [issue.to_dict() for issue in report.issues]},
            )
            self._strategy.on_rejected(proposal.spec, report.summary())
            emit(
                Event.CANDIDATE_REJECTED
                if terminal is CandidateState.FAILED
                else Event.CANDIDATE_PRUNED,
                search_id=search_id,
                candidate_id=candidate_id,
                architecture_hash=hash_value,
                reason=report.summary(),
            )
            return None

        self._repository.update_candidate_state(candidate_id, CandidateState.VALIDATED)
        self._repository.update_candidate_state(candidate_id, CandidateState.QUEUED)
        self._state.accepted += 1
        self._metrics.increment("candidates.accepted")
        self._pending_specs[candidate_id] = proposal.spec
        self._pending_parents[candidate_id] = proposal.parent_id
        self._attempts.setdefault(candidate_id, 0)
        emit(
            Event.CANDIDATE_QUEUED,
            search_id=search_id,
            candidate_id=candidate_id,
            architecture_hash=hash_value,
            budget=proposal.budget.describe(),
        )
        return candidate_id

    # -------------------------------------------------------------------- results --
    def _process_result(
        self, search_id: str, task: EvaluationTask, result: EvaluationResult
    ) -> None:
        """Persist one evaluation result and notify the strategy.

        Args:
            search_id: Owning search.
            task: The task that produced the result.
            result: The evaluation outcome.
        """
        candidate_id = task.candidate_id
        spec = self._pending_specs.get(candidate_id, task.spec)

        with candidate_context(
            candidate_id, architecture_hash=task.architecture_hash, trial_id=task.trial_id
        ):
            if result.succeeded:
                self._finish_successful(search_id, task, result, spec)
            else:
                self._finish_failed(search_id, task, result, spec)

    def _finish_successful(
        self,
        search_id: str,
        task: EvaluationTask,
        result: EvaluationResult,
        spec: ArchitectureSpec,
    ) -> None:
        """Record a successful evaluation."""
        objective_value = online_objective_value(result.metrics, self._objectives)
        self._repository.complete_trial(
            task.trial_id,
            metrics=result.metrics,
            duration_seconds=result.duration_seconds,
            training=result.training,
            artifacts=result.artifacts,
            artifact_sizes=result.artifact_bytes,
        )
        self._repository.update_candidate_state(
            task.candidate_id,
            CandidateState.COMPLETED,
            objective_value=objective_value,
        )
        self._state.completed += 1
        self._metrics.increment("evaluations.completed")
        self._metrics.observe_duration("evaluation", result.duration_seconds)
        self._pending_specs.pop(task.candidate_id, None)

        emit(
            Event.EVALUATION_COMPLETED,
            search_id=search_id,
            candidate_id=task.candidate_id,
            architecture_hash=task.architecture_hash,
            trial_id=task.trial_id,
            validation_accuracy=result.metrics.get("validation_accuracy"),
            objective_value=objective_value,
            duration_seconds=result.duration_seconds,
            worker_id=result.worker_id,
        )
        self._strategy.observe(
            Observation(
                candidate_id=task.candidate_id,
                architecture_hash=task.architecture_hash,
                spec=spec,
                result=result,
                objective_value=objective_value,
                parent_id=self._pending_parents.get(task.candidate_id),
            )
        )

    def _finish_failed(
        self,
        search_id: str,
        task: EvaluationTask,
        result: EvaluationResult,
        spec: ArchitectureSpec,
    ) -> None:
        """Record a failed evaluation and apply the retry policy."""
        failure = result.failure
        assert failure is not None  # a failed result always carries a failure record

        from nas_engine.evaluation.result import FailureKind

        self._repository.fail_trial(
            task.trial_id,
            error=failure.to_dict(),
            duration_seconds=result.duration_seconds,
            timeout=failure.kind is FailureKind.TIMEOUT,
        )
        emit(
            Event.EVALUATION_TIMEOUT
            if failure.kind is FailureKind.TIMEOUT
            else Event.EVALUATION_FAILED,
            search_id=search_id,
            candidate_id=task.candidate_id,
            architecture_hash=task.architecture_hash,
            trial_id=task.trial_id,
            failure_kind=failure.kind.value,
            error=failure.message,
        )

        decision = self._retry_policy.decide(failure, attempt=task.attempt)
        if decision.should_retry:
            self._repository.increment_retry(task.candidate_id)
            self._repository.update_candidate_state(
                task.candidate_id, CandidateState.QUEUED, reason=decision.reason
            )
            self._attempts[task.candidate_id] = task.attempt + 1
            self._state.retried += 1
            self._metrics.increment("evaluations.retried")
            emit(
                Event.RETRY_SCHEDULED,
                search_id=search_id,
                candidate_id=task.candidate_id,
                attempt=task.attempt + 1,
                delay_seconds=decision.delay_seconds,
                reason=decision.reason,
            )
            if decision.delay_seconds > 0:
                time.sleep(decision.delay_seconds)
            return

        self._repository.update_candidate_state(
            task.candidate_id,
            CandidateState.FAILED,
            reason=decision.reason,
            error=failure.to_dict(),
        )
        self._state.failed += 1
        self._metrics.increment("evaluations.failed")
        self._pending_specs.pop(task.candidate_id, None)
        if task.attempt >= self._retry_policy.max_retries and failure.retriable:
            emit(
                Event.RETRY_EXHAUSTED,
                search_id=search_id,
                candidate_id=task.candidate_id,
                attempts=task.attempt + 1,
            )
        self._strategy.observe(
            Observation(
                candidate_id=task.candidate_id,
                architecture_hash=task.architecture_hash,
                spec=spec,
                result=result,
                objective_value=None,
                parent_id=self._pending_parents.get(task.candidate_id),
            )
        )

    # ---------------------------------------------------------------- dispatching --
    def _build_tasks(self, search_id: str, limit: int) -> list[EvaluationTask]:
        """Claim up to ``limit`` queued candidates and turn them into tasks.

        Args:
            search_id: Owning search.
            limit: Maximum tasks to build.

        Returns:
            The tasks, each with a freshly recorded trial.
        """
        tasks: list[EvaluationTask] = []
        for _ in range(limit):
            claimed = self._repository.claim_next_queued(search_id, worker_id="engine")
            if claimed is None:
                break
            spec = self._pending_specs.get(claimed.id)
            if spec is None:
                # Reached after a resume, where the candidate was queued by a previous
                # process and is not in this process's memory.
                spec = self._repository.get_candidate_spec(claimed.id)
                self._pending_specs[claimed.id] = spec
                self._pending_parents.setdefault(claimed.id, claimed.parent_id)
            attempt = self._attempts.get(claimed.id, claimed.retry_count)
            self._attempts[claimed.id] = attempt
            budget = self._budget_for(claimed.id)

            trial_id = self._repository.start_trial(
                candidate_id=claimed.id,
                attempt=attempt,
                budget=budget,
                worker_id="engine",
                device=str(self._device),
            )
            tasks.append(
                EvaluationTask(
                    candidate_id=claimed.id,
                    trial_id=trial_id,
                    architecture_hash=claimed.architecture_hash,
                    spec=spec,
                    budget=budget,
                    attempt=attempt,
                )
            )
            emit(
                Event.EVALUATION_STARTED,
                search_id=search_id,
                candidate_id=claimed.id,
                architecture_hash=claimed.architecture_hash,
                trial_id=trial_id,
                attempt=attempt,
                budget=budget.describe(),
            )
        return tasks

    def _budget_for(self, candidate_id: str) -> TrainingBudget:
        """Return the budget a candidate was proposed with.

        The budget travels in the candidate's stored metadata so that a resumed run
        reconstructs the exact fidelity the strategy asked for. Falling back to the
        configuration default would silently promote a rung-0 candidate to full fidelity
        after a restart.

        Args:
            candidate_id: Candidate to read.

        Returns:
            The stored budget, or the configuration default when none was recorded.
        """
        metadata = self._candidate_metadata(candidate_id)
        stored = metadata.get("budget")
        if isinstance(stored, dict):
            return TrainingBudget.from_dict(stored)
        return self.config.budget.build_budget()

    def _candidate_metadata(self, candidate_id: str) -> dict[str, Any]:
        """Read a candidate's stored metadata mapping.

        Args:
            candidate_id: Candidate to read.

        Returns:
            The metadata mapping; empty when the candidate is missing.
        """
        from sqlalchemy import select

        from nas_engine.persistence.models import CandidateRecord

        with self._database.session() as session:
            record = session.scalars(
                select(CandidateRecord).where(CandidateRecord.id == candidate_id)
            ).one_or_none()
            return dict(record.metadata_json) if record else {}

    # ---------------------------------------------------------------------- run ---
    def run(self, *, resume: bool = False, search_id: str | None = None) -> SearchResult:
        """Run the search to completion.

        Args:
            resume: Whether to continue an existing search rather than create a new one.
            search_id: Search to resume; the most recent one when omitted.

        Returns:
            A :class:`~nas_engine.orchestration.result.SearchResult`.

        Raises:
            OrchestrationError: If a resume is requested but no matching search exists.
            RecordNotFoundError: If an explicit ``search_id`` does not exist.
        """
        watch = Stopwatch().start()
        resumed = False

        if resume:
            target = search_id or self._find_resumable()
            self._search_id = target
            summary = self._repository.get_search(target)
            stored_config = self._repository.get_search_config(target)
            self._warnings.extend(check_config_compatibility(stored_config, self.config))
            recovery = self._repository.recover_interrupted(
                target, max_retries=self.config.retry.max_retries
            )
            if recovery.interrupted_running:
                self._warnings.append(
                    f"recovered {recovery.interrupted_running} interrupted evaluation(s): "
                    f"{len(recovery.requeued)} requeued, {len(recovery.abandoned)} abandoned"
                )
            self._restore(target)
            self._reconcile_completed(target)
            resumed = True
            emit(
                Event.SEARCH_RESUMED,
                search_id=target,
                strategy=summary.strategy,
                completed=recovery.completed,
                requeued=len(recovery.requeued),
            )
        else:
            self._search_id = self._create_search()
            emit(
                Event.SEARCH_STARTED,
                search_id=self._search_id,
                strategy=self.config.algorithm.name,
                max_evaluations=self.config.budget.max_evaluations,
                device=str(self._device),
                concurrency=self.config.concurrency.mode,
                seed=self.config.reproducibility.seed,
            )

        assert self._search_id is not None
        current_id = self._search_id
        self._repository.update_search_status(current_id, SearchStatus.RUNNING, started=True)

        stop_reason = StopReason.STRATEGY_FINISHED
        executor = self._build_executor()

        try:
            with search_context(current_id, strategy=self._strategy.name):
                stop_reason = self._loop(current_id, executor, watch)
        except KeyboardInterrupt:
            stop_reason = StopReason.INTERRUPTED
            self._warnings.append(
                "the run was interrupted; resume it with "
                f"'nas-engine resume --search-id {current_id}'"
            )
            emit(Event.SEARCH_INTERRUPTED, search_id=current_id)
        except Exception as error:
            stop_reason = StopReason.ERROR
            emit(
                Event.SEARCH_FAILED,
                search_id=current_id,
                error=str(error),
                error_type=type(error).__name__,
            )
            self._finalise(current_id, stop_reason, watch, resumed)
            raise
        finally:
            executor.shutdown()

        return self._finalise(current_id, stop_reason, watch, resumed)

    def resume(self, search_id: str | None = None) -> SearchResult:
        """Resume an interrupted search.

        Args:
            search_id: Search to resume; the most recent resumable one when omitted.

        Returns:
            A :class:`~nas_engine.orchestration.result.SearchResult`.

        Raises:
            OrchestrationError: If no resumable search exists.
        """
        return self.run(resume=True, search_id=search_id)

    def _reconcile_completed(self, search_id: str) -> None:
        """Trust the database over the checkpoint for the completed-evaluation count.

        Recovery can *undo* a completion: a candidate that a crashed process had already
        finished, but whose result was never persisted, goes back to the queue. The
        checkpoint's counter still includes it, so without reconciliation the engine would
        believe its budget was spent and would leave the recovered candidate queued
        forever — silently returning fewer results than requested.

        The database is authoritative because it is what the recovery sweep just updated.

        Args:
            search_id: Search being resumed.
        """
        counts = self._repository.count_candidates_by_status(search_id)
        actual = counts.get(CandidateState.COMPLETED.value, 0)
        if actual != self._state.completed:
            _LOGGER.info(
                "engine.completed_count_reconciled",
                search_id=search_id,
                checkpoint=self._state.completed,
                database=actual,
            )
            self._state.completed = actual

    def _find_resumable(self) -> str:
        """Locate the most recent search that can be resumed.

        Returns:
            The search identifier.

        Raises:
            OrchestrationError: If no matching search exists.
        """
        summary = self._repository.find_latest_search(name=self.config.project.name)
        if summary is None:
            summary = self._repository.find_latest_search()
        if summary is None:
            msg = (
                "no search was found to resume in "
                f"{self.config.database_url}. Start one with 'nas-engine search'."
            )
            raise OrchestrationError(msg, details={"database": self.config.database_url})
        if summary.status == SearchStatus.COMPLETED.value:
            self._warnings.append(
                f"search {summary.id} is already marked completed; resuming will only "
                "re-check its stopping conditions"
            )
        return summary.id

    def _loop(self, search_id: str, executor: EvaluationExecutor, watch: Stopwatch) -> StopReason:
        """Run the propose-evaluate-observe loop until a stopping condition fires.

        Args:
            search_id: Owning search.
            executor: Execution backend.
            watch: Stopwatch measuring this run segment.

        Returns:
            The reason the loop stopped.
        """
        in_flight_limit = max(1, executor.max_in_flight)
        if self._strategy.requires_synchronous_observations:
            in_flight_limit = 1
        idle_rounds = 0

        while True:
            if self._state.completed >= self.config.budget.max_evaluations:
                return StopReason.BUDGET_EXHAUSTED
            if self._exceeded_time_limit(watch):
                return StopReason.TIME_LIMIT

            queued = self._queued_count(search_id)
            free_slots = max(0, in_flight_limit - queued)
            if free_slots > 0 and not self._strategy.is_finished():
                remaining_budget = self.config.budget.max_evaluations - self._state.completed
                proposals = self._strategy.propose(min(free_slots, max(1, remaining_budget)))
                for proposal in proposals:
                    self._accept_proposal(search_id, proposal)

            tasks = self._build_tasks(search_id, in_flight_limit)
            if not tasks:
                if self._strategy.is_finished():
                    return StopReason.STRATEGY_FINISHED
                idle_rounds += 1
                if idle_rounds >= 2:
                    # Two consecutive rounds with nothing to run and nothing proposed means
                    # the strategy cannot make progress. Spinning would burn CPU forever.
                    return StopReason.SPACE_EXHAUSTED
                continue
            idle_rounds = 0

            results = executor.run_batch(tasks)
            for task, result in zip(tasks, results, strict=True):
                self._process_result(search_id, task, result)

            if self._state.completed % self.config.persistence.checkpoint_every == 0:
                self._checkpoint(search_id)

    def _queued_count(self, search_id: str) -> int:
        """Return how many candidates are waiting to run."""
        counts = self._repository.count_candidates_by_status(search_id)
        return counts.get(CandidateState.QUEUED.value, 0)

    def _exceeded_time_limit(self, watch: Stopwatch) -> bool:
        """Report whether the configured wall-clock limit has been reached."""
        limit = self.config.budget.max_seconds
        if limit is None:
            return False
        return self._state.elapsed_seconds + watch.elapsed_seconds >= limit

    # ----------------------------------------------------------------- finalising --
    def _finalise(
        self, search_id: str, stop_reason: StopReason, watch: Stopwatch, resumed: bool
    ) -> SearchResult:
        """Write final state and build the result object.

        Args:
            search_id: Owning search.
            stop_reason: Why the run stopped.
            watch: Stopwatch measuring this segment.
            resumed: Whether this segment resumed an existing search.

        Returns:
            The search result.
        """
        duration = watch.elapsed_seconds
        self._state.elapsed_seconds += duration
        self._checkpoint(search_id)

        status = {
            StopReason.INTERRUPTED: SearchStatus.PAUSED,
            StopReason.ERROR: SearchStatus.FAILED,
        }.get(stop_reason, SearchStatus.COMPLETED)
        self._repository.update_search_status(
            search_id,
            status,
            completed=status is SearchStatus.COMPLETED,
        )

        ranking = self.ranking(search_id)
        emit(
            Event.SEARCH_COMPLETED,
            search_id=search_id,
            stop_reason=stop_reason.value,
            completed=self._state.completed,
            failed=self._state.failed,
            duration_seconds=duration,
            pareto_size=len(ranking.pareto_front),
        )
        if ranking.pareto_front:
            emit(
                Event.PARETO_UPDATED,
                search_id=search_id,
                size=len(ranking.pareto_front),
                members=[candidate.architecture_hash for candidate in ranking.pareto_front],
            )

        return SearchResult(
            search_id=search_id,
            status=status.value,
            stop_reason=stop_reason,
            best=ranking.best,
            pareto_front=ranking.pareto_front,
            ranked=ranking.ranked,
            engine_state=self._state,
            strategy_statistics=self._strategy.statistics().to_dict(),
            duration_seconds=duration,
            total_evaluations=self._state.completed,
            warnings=tuple(self._warnings),
            resumed=resumed,
        )

    def ranking(self, search_id: str | None = None) -> RankingResult:
        """Recompute the ranking and Pareto front from persisted metrics.

        Always recomputed, never cached: a cached front goes stale the moment another
        candidate completes, and a stale front is worse than a slow one.

        Args:
            search_id: Search to rank; the current one when omitted.

        Returns:
            The ranking.

        Raises:
            OrchestrationError: If no search is available to rank.
        """
        target = search_id or self._search_id
        if target is None:
            msg = "no search has been run yet, so there is nothing to rank"
            raise OrchestrationError(msg)
        population = self._repository.completed_metrics(target)
        return rank_candidates(population, self._objectives, constraints=self._constraints)

    def load_best_model(self, search_id: str | None = None) -> tuple[ArchitectureSpec, Any]:
        """Rebuild the best candidate's model with its trained weights.

        Args:
            search_id: Search to load from; the current one when omitted.

        Returns:
            The architecture and the loaded :class:`torch.nn.Module`.

        Raises:
            OrchestrationError: If no search or no completed candidate is available.
            RecordNotFoundError: If the candidate's artifacts are missing.
        """
        target = search_id or self._search_id
        if target is None:
            msg = "no search has been run yet, so there is no best model to load"
            raise OrchestrationError(msg)
        ranking = self.ranking(target)
        if ranking.best is None:
            msg = f"search {target} has no completed candidate to load"
            raise OrchestrationError(msg, details={"search_id": target})

        candidate = self._repository.get_candidate(ranking.best.candidate_id)
        spec = self._repository.get_candidate_spec(candidate.id)
        model = ModelBuilder(initialize=False).build(spec, device=self._device)

        weights_path = candidate.artifacts.get("weights")
        if weights_path is None:
            msg = (
                f"candidate {candidate.id} has no stored weights; enable "
                "evaluation.save_weights before running the search to be able to reload "
                "the trained model"
            )
            raise RecordNotFoundError(msg, details={"candidate_id": candidate.id})
        full_path = self._artifact_root / weights_path
        if not full_path.is_file():
            msg = (
                f"the weights file for candidate {candidate.id} is missing from disk: "
                f"{full_path}. The database record survived but the artifact did not."
            )
            raise RecordNotFoundError(
                msg, details={"candidate_id": candidate.id, "path": str(full_path)}
            )
        state = torch.load(full_path, map_location=self._device, weights_only=True)
        model.load_state_dict(state)
        return spec, model

    def close(self) -> None:
        """Release the database connection when this engine owns it."""
        if self._owns_database:
            self._database.dispose()

    def __enter__(self) -> SearchEngine:
        """Enter a context that closes the engine on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the engine."""
        self.close()


__all__ = ["SearchEngine"]
