"""Candidate evaluation: the bridge between a genotype and a set of measured metrics.

Responsibilities
----------------
The evaluator owns the full lifecycle of measuring one candidate at one budget:

1. Check resource constraints analytically, before allocating anything.
2. Seed every RNG from a value derived from the *architecture hash*, not from the
   evaluation order.
3. Build the data loaders for the requested fidelity.
4. Build the model.
5. Train it, with checkpointing so an interrupted evaluation can resume.
6. Measure validation accuracy, parameter count, serialised size, and latency.
7. Persist weights as an artifact.
8. Return a single :class:`~nas_engine.evaluation.result.EvaluationResult`, whether it
   succeeded or failed.

Why seeding by architecture hash matters
-----------------------------------------
If every candidate drew from one shared stream, the weights a candidate received would
depend on how many candidates were evaluated before it. Under multiprocessing that order
is nondeterministic, so the same search would produce different results on every run —
and worse, a candidate's measured accuracy would depend on its position in the queue.
Deriving the seed from ``(master_seed, architecture_hash, rung)`` makes each candidate's
initial weights a pure function of *what it is*, not *when it ran*. Two searches that
propose the same architecture will train it identically.

Failure handling
----------------
The evaluator never raises for a candidate-specific problem. Every exception is caught,
classified by :func:`~nas_engine.evaluation.result.classify_failure`, and returned as a
failed result. The orchestration engine decides what to do about it. Exceptions that
indicate the *engine* is broken (rather than the candidate) propagate — see
:class:`KeyboardInterrupt` handling.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nas_engine.architectures.cost import compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.loaders import (
    DataLoaders,
    FidelityView,
    LoaderSettings,
    build_dataloaders,
)
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.latency import LatencyMeasurement, measure_latency
from nas_engine.evaluation.model_size import measure_model_size, save_model_weights
from nas_engine.evaluation.result import EvaluationFailure, EvaluationResult
from nas_engine.exceptions import ResourceLimitError
from nas_engine.models.builder import ModelBuilder, NasNetwork
from nas_engine.observability.logging import get_logger
from nas_engine.training.trainer import Trainer, TrainingSettings
from nas_engine.utilities.paths import resolve_under_root, safe_filename
from nas_engine.utilities.seeding import derive_seed, seed_everything
from nas_engine.utilities.timing import Stopwatch, utc_now

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationSettings:
    """Configuration for what the evaluator measures and stores.

    Attributes:
        measure_latency: Whether to benchmark inference latency.
        latency_batch_size: Batch size for the latency benchmark.
        latency_warmup_iterations: Untimed warm-up passes.
        latency_timed_iterations: Passes per timed block.
        latency_repeats: Number of timed blocks.
        measure_model_size: Whether to serialise the model to measure its size.
        save_weights: Whether to persist trained weights as an artifact.
        save_training_checkpoints: Whether to keep resumable training checkpoints. Costs
            disk but makes an interrupted evaluation resumable rather than restarted.
        max_parameters: Hard ceiling enforced before building; ``None`` disables.
        max_evaluation_seconds: Wall-clock ceiling applied when the budget does not set
            one; ``None`` disables.
    """

    measure_latency: bool = True
    latency_batch_size: int = 1
    latency_warmup_iterations: int = 3
    latency_timed_iterations: int = 5
    latency_repeats: int = 3
    measure_model_size: bool = True
    save_weights: bool = True
    save_training_checkpoints: bool = False
    max_parameters: int | None = None
    max_evaluation_seconds: float | None = None


@dataclass(frozen=True)
class EvaluationContext:
    """Identifiers attached to one evaluation.

    Attributes:
        candidate_id: Candidate identifier.
        trial_id: Identifier of this specific attempt; distinguishes retries and rungs.
        worker_id: Worker identifier when running under multiprocessing.
        attempt: Zero-based retry attempt number.
    """

    candidate_id: str
    trial_id: str
    worker_id: str | None = None
    attempt: int = 0


@dataclass
class _Measurements:
    """Internal accumulator for the metrics gathered during one evaluation."""

    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_bytes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    latency: LatencyMeasurement | None = None


class CandidateEvaluator:
    """Trains and measures one candidate at a time.

    Dependencies are injected rather than constructed internally so that tests can supply
    a stub model builder or a tiny dataset without patching module globals.

    Args:
        dataset: The dataset bundle to train and validate on.
        loader_settings: DataLoader configuration.
        training_settings: Training hyperparameters. The epoch count is overridden per
            budget.
        settings: What to measure and store.
        artifact_root: Directory that all artifacts are written under. Every path is
            validated against this root, so a hostile architecture hash cannot escape it.
        device: Device to evaluate on.
        seed: Master seed; per-candidate seeds are derived from it.
        model_builder: Builder used to construct networks.

    Raises:
        UnsafePathError: If ``artifact_root`` cannot be created or written to.
    """

    def __init__(
        self,
        *,
        dataset: DatasetBundle,
        loader_settings: LoaderSettings,
        training_settings: TrainingSettings,
        settings: EvaluationSettings | None = None,
        artifact_root: Path,
        device: torch.device | str = "cpu",
        seed: int = 42,
        model_builder: ModelBuilder | None = None,
    ) -> None:
        from nas_engine.utilities.paths import ensure_directory

        self._dataset = dataset
        self._loader_settings = loader_settings
        self._training_settings = training_settings
        self._settings = settings if settings is not None else EvaluationSettings()
        self._artifact_root = ensure_directory(Path(artifact_root))
        self._device = torch.device(device)
        self._seed = seed
        self._model_builder = model_builder if model_builder is not None else ModelBuilder()
        self._loader_cache: dict[tuple[float, int | None], DataLoaders] = {}

    # -- properties ----------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """The device evaluations run on."""
        return self._device

    @property
    def dataset(self) -> DatasetBundle:
        """The dataset bundle in use."""
        return self._dataset

    @property
    def artifact_root(self) -> Path:
        """The validated artifact root."""
        return self._artifact_root

    # -- helpers -------------------------------------------------------------------
    def candidate_seed(self, spec: ArchitectureSpec, budget: TrainingBudget) -> int:
        """Derive the seed used to initialise and train one candidate.

        Args:
            spec: The architecture.
            budget: The budget it will be trained at.

        Returns:
            A seed that depends only on the master seed, the architecture, and the rung.
        """
        return derive_seed(self._seed, f"eval:{architecture_hash(spec)}:{budget.rung}")

    def _loaders_for(self, budget: TrainingBudget) -> DataLoaders:
        """Return (and cache) the loaders for a budget's fidelity.

        Loaders are cached per fidelity because constructing them re-derives a random
        permutation over the training split; caching keeps that cost off the per-candidate
        path without changing behaviour, since the permutation is seeded and therefore
        identical each time.

        Args:
            budget: Budget whose fidelity determines the loaders.

        Returns:
            The loaders.
        """
        key = (budget.train_fraction, budget.resolution)
        cached = self._loader_cache.get(key)
        if cached is not None:
            return cached
        loaders = build_dataloaders(
            self._dataset,
            self._loader_settings,
            seed=derive_seed(self._seed, "loaders"),
            fidelity=FidelityView(
                train_fraction=budget.train_fraction, resolution=budget.resolution
            ),
        )
        self._loader_cache[key] = loaders
        return loaders

    def _candidate_directory(self, spec: ArchitectureSpec) -> Path:
        """Return the artifact directory for one architecture, creating it if needed.

        Args:
            spec: The architecture.

        Returns:
            A directory guaranteed to be inside the artifact root.

        Raises:
            UnsafePathError: If the directory escapes the artifact root.
        """
        name = safe_filename(architecture_hash(spec))
        directory = resolve_under_root(self._artifact_root, name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _enforce_parameter_limit(self, spec: ArchitectureSpec) -> None:
        """Reject a candidate that exceeds the configured parameter ceiling.

        Args:
            spec: The architecture.

        Raises:
            ResourceLimitError: If the analytic parameter count exceeds the limit.
        """
        limit = self._settings.max_parameters
        if limit is None:
            return
        cost = compute_cost(spec)
        if cost.trainable_parameters > limit:
            msg = (
                f"architecture has {cost.trainable_parameters:,} trainable parameters "
                f"which exceeds the evaluation limit of {limit:,}; raise "
                "evaluation.max_parameters or tighten the search space"
            )
            raise ResourceLimitError(
                msg,
                details={
                    "trainable_parameters": cost.trainable_parameters,
                    "limit": limit,
                },
            )

    def _measure_static(self, spec: ArchitectureSpec, model: NasNetwork) -> _Measurements:
        """Collect metrics that do not require training.

        Args:
            spec: The architecture.
            model: The built model.

        Returns:
            A populated :class:`_Measurements`.
        """
        measurements = _Measurements()
        cost = compute_cost(spec, model.trace)
        measurements.metrics.update(
            {
                "trainable_parameters": float(cost.trainable_parameters),
                "non_trainable_parameters": float(cost.non_trainable_parameters),
                "total_parameters": float(cost.total_parameters),
                "multiply_accumulates": float(cost.multiply_accumulates),
                "depth": float(cost.depth),
                "total_stride": float(spec.total_stride),
            }
        )
        if self._settings.measure_model_size:
            size = measure_model_size(model)
            measurements.metrics.update(size.to_metrics())
        return measurements

    def _measure_latency(
        self, spec: ArchitectureSpec, model: NasNetwork, measurements: _Measurements
    ) -> None:
        """Benchmark latency and fold the results into ``measurements``.

        Args:
            spec: The architecture, used for the input shape.
            model: The trained model.
            measurements: Accumulator updated in place.
        """
        if not self._settings.measure_latency:
            return
        latency = measure_latency(
            model,
            input_shape=(spec.input_channels, spec.input_size, spec.input_size),
            device=self._device,
            batch_size=self._settings.latency_batch_size,
            warmup_iterations=self._settings.latency_warmup_iterations,
            timed_iterations=self._settings.latency_timed_iterations,
            repeats=self._settings.latency_repeats,
        )
        measurements.latency = latency
        measurements.metrics.update(latency.to_metrics())
        measurements.notes.append(latency.warning)

    # -- main entry point ----------------------------------------------------------
    def evaluate(
        self,
        spec: ArchitectureSpec,
        budget: TrainingBudget,
        context: EvaluationContext,
        *,
        resume: bool = True,
    ) -> EvaluationResult:
        """Evaluate one candidate and return its result.

        This method does not raise for candidate-level problems: failures are captured,
        classified, and returned inside the result. :class:`KeyboardInterrupt` and
        :class:`SystemExit` propagate, because they mean the operator wants the process to
        stop, not that the candidate is bad.

        Args:
            spec: Architecture to evaluate.
            budget: Resources to spend.
            context: Identifiers for logging and artifact naming.
            resume: Whether to resume from an existing training checkpoint.

        Returns:
            An :class:`~nas_engine.evaluation.result.EvaluationResult`.
        """
        hash_value = architecture_hash(spec)
        started_at = utc_now()
        watch = Stopwatch().start()

        try:
            return self._evaluate_inner(
                spec,
                budget,
                context,
                hash_value=hash_value,
                started_at=started_at,
                watch=watch,
                resume=resume,
            )
        except (KeyboardInterrupt, SystemExit):
            # Operator intent, not a candidate failure. Let it unwind.
            raise
        except BaseException as error:
            failure = EvaluationFailure.from_exception(
                error, traceback_text=traceback.format_exc(limit=20)
            )
            _LOGGER.warning(
                "evaluator.failed",
                candidate_id=context.candidate_id,
                trial_id=context.trial_id,
                architecture_hash=hash_value,
                failure_kind=failure.kind.value,
                retriable=failure.retriable,
                error=failure.message,
            )
            return EvaluationResult(
                candidate_id=context.candidate_id,
                architecture_hash=hash_value,
                budget=budget,
                succeeded=False,
                failure=failure,
                started_at=started_at,
                completed_at=utc_now(),
                duration_seconds=watch.stop(),
                device=str(self._device),
                worker_id=context.worker_id,
            )

    def _evaluate_inner(
        self,
        spec: ArchitectureSpec,
        budget: TrainingBudget,
        context: EvaluationContext,
        *,
        hash_value: str,
        started_at: object,
        watch: Stopwatch,
        resume: bool,
    ) -> EvaluationResult:
        """Run the successful path of an evaluation.

        Split from :meth:`evaluate` so the exception handler there stays a single, obvious
        boundary rather than wrapping a hundred lines of logic.

        Args:
            spec: Architecture to evaluate.
            budget: Resources to spend.
            context: Identifiers.
            hash_value: Pre-computed architecture hash.
            started_at: Start timestamp.
            watch: Running stopwatch.
            resume: Whether to resume from a training checkpoint.

        Returns:
            A successful :class:`~nas_engine.evaluation.result.EvaluationResult`.
        """
        from datetime import datetime

        assert isinstance(started_at, datetime)

        self._enforce_parameter_limit(spec)

        seed = self.candidate_seed(spec, budget)
        seed_everything(seed)

        loaders = self._loaders_for(budget)
        model = self._model_builder.build(spec, device=self._device)
        measurements = self._measure_static(spec, model)

        directory = self._candidate_directory(spec)
        checkpoint_path = (
            directory / f"training_{budget.key}.pt"
            if self._settings.save_training_checkpoints
            else None
        )

        max_seconds = budget.max_seconds or self._settings.max_evaluation_seconds
        training_settings = self._training_settings.with_epochs(budget.epochs)
        if max_seconds is not None:
            training_settings = TrainingSettings(
                epochs=budget.epochs,
                optimizer=training_settings.optimizer,
                scheduler=training_settings.scheduler,
                gradient_clip_norm=training_settings.gradient_clip_norm,
                label_smoothing=training_settings.label_smoothing,
                early_stopping_patience=training_settings.early_stopping_patience,
                early_stopping_min_delta=training_settings.early_stopping_min_delta,
                mixed_precision=training_settings.mixed_precision,
                topk=training_settings.topk,
                max_seconds=max_seconds,
                restore_best_weights=training_settings.restore_best_weights,
                checkpoint_every_epochs=training_settings.checkpoint_every_epochs,
                log_every_n_steps=training_settings.log_every_n_steps,
            )

        # These lines use an ``evaluator.*`` namespace rather than the ``evaluation.*``
        # names in the Event enum. The engine already emits one Event per attempt; if this
        # inner log reused those names, every attempt would appear twice and anything
        # counting evaluations would double it. The evaluator's value is that it runs
        # inside the worker process and reports what the attempt actually did.
        _LOGGER.info(
            "evaluator.started",
            candidate_id=context.candidate_id,
            trial_id=context.trial_id,
            architecture_hash=hash_value,
            budget=budget.describe(),
            worker_id=context.worker_id,
            device=str(self._device),
            seed=seed,
        )

        trainer = Trainer(training_settings, device=self._device, seed=seed)
        outcome = trainer.fit(
            model,
            loaders,
            architecture_hash=hash_value,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

        measurements.metrics.update(
            {
                "validation_accuracy": outcome.best_validation_accuracy,
                "validation_loss": outcome.best_validation_loss,
                "validation_topk_accuracy": outcome.best_validation_topk,
                "train_loss": outcome.final_train_loss,
                "epochs_completed": float(outcome.epochs_completed),
                "training_seconds": outcome.duration_seconds,
                "train_examples": float(loaders.train_examples),
                "effective_resolution": float(loaders.input_size),
            }
        )
        self._measure_latency(spec, model, measurements)

        if self._settings.save_weights:
            weights_path = directory / f"weights_{budget.key}.pt"
            size = save_model_weights(model, weights_path)
            relative = str(weights_path.relative_to(self._artifact_root))
            measurements.artifacts["weights"] = relative
            measurements.artifact_bytes["weights"] = size
            measurements.metrics.setdefault("model_size_bytes", float(size))
        if checkpoint_path is not None and checkpoint_path.exists():
            relative_checkpoint = str(checkpoint_path.relative_to(self._artifact_root))
            measurements.artifacts["training_checkpoint"] = relative_checkpoint
            measurements.artifact_bytes["training_checkpoint"] = checkpoint_path.stat().st_size

        duration = watch.stop()
        measurements.metrics["evaluation_seconds"] = duration

        _LOGGER.info(
            "evaluator.completed",
            candidate_id=context.candidate_id,
            trial_id=context.trial_id,
            architecture_hash=hash_value,
            validation_accuracy=outcome.best_validation_accuracy,
            trainable_parameters=measurements.metrics.get("trainable_parameters"),
            duration_seconds=duration,
            worker_id=context.worker_id,
        )

        return EvaluationResult(
            candidate_id=context.candidate_id,
            architecture_hash=hash_value,
            budget=budget,
            metrics=measurements.metrics,
            succeeded=True,
            artifacts=measurements.artifacts,
            artifact_bytes=measurements.artifact_bytes,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=duration,
            device=str(self._device),
            worker_id=context.worker_id,
            training=outcome.to_dict(),
            notes=tuple(measurements.notes),
        )

    def evaluate_on_test(
        self, spec: ArchitectureSpec, *, weights_path: Path | None = None
    ) -> dict[str, float]:
        """Measure a trained architecture on the held-out test split.

        This is the *only* code path that touches the test split, and it is never called
        during a search. Doing so would leak the test set into the selection process and
        invalidate the headline number.

        Args:
            spec: Architecture to evaluate.
            weights_path: Trained weights to load. When ``None``, a freshly initialised
                model is measured, which is only useful as a sanity check.

        Returns:
            Test metrics keyed ``test_accuracy``, ``test_loss``, ``test_topk_accuracy``.

        Raises:
            CheckpointError: If the weights file cannot be read.
        """
        model = self._model_builder.build(spec, device=self._device)
        if weights_path is not None:
            from nas_engine.exceptions import CheckpointError

            if not weights_path.is_file():
                msg = f"weights file not found: {weights_path}"
                raise CheckpointError(msg, details={"path": str(weights_path)})
            try:
                state = torch.load(weights_path, map_location=self._device, weights_only=True)
            except Exception as exc:
                msg = f"weights at {weights_path} could not be read: {exc}"
                raise CheckpointError(
                    msg, details={"path": str(weights_path), "error": str(exc)}
                ) from exc
            model.load_state_dict(state)

        loaders = build_dataloaders(
            self._dataset,
            self._loader_settings,
            seed=derive_seed(self._seed, "loaders"),
            include_test=True,
        )
        assert loaders.test is not None
        trainer = Trainer(self._training_settings, device=self._device, seed=self._seed)
        metrics = trainer.evaluate(model, loaders.test, phase="test")
        return {
            "test_accuracy": metrics.accuracy,
            "test_loss": metrics.loss,
            "test_topk_accuracy": metrics.topk_accuracy,
            "test_examples": float(metrics.examples),
        }


__all__ = ["CandidateEvaluator", "EvaluationContext", "EvaluationSettings"]
