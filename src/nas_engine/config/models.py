"""Validated configuration models.

Everything a search needs is described by one :class:`SearchConfig` tree. Pydantic
validates it once, at the edge, and every component downstream receives already-checked
values — so no module needs defensive checks for "what if epochs is negative".

Principles
----------
**Fail early, fail loudly.** ``extra="forbid"`` means a typo like ``epocs: 5`` is an error
rather than a silently ignored key that leaves the default in place. That single setting
prevents an entire class of "why did my configuration have no effect" confusion.

**Every error is actionable.** Validators state the field, the received value, the
expected range, and how to fix it. A message that says only "invalid value" costs the
reader a trip to the source.

**Configuration is data, never code.** YAML is parsed with ``yaml.safe_load``; no field
names a Python object to import or a callable to evaluate. A configuration file from an
untrusted source can misconfigure a run but cannot execute anything. See
``docs/architecture/security.md``.

**Versioned.** Every saved configuration carries :data:`CONFIG_VERSION`. Resuming a search
whose stored configuration came from an incompatible version fails with an explanation
rather than a subtly different run.

**The config layer converts, the domain does not.** Domain packages (``training``,
``evaluation``, ``search``) take plain dataclasses. The ``build_*`` methods here do the
conversion, which keeps Pydantic out of the domain and lets the domain be used without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nas_engine.architectures.types import ActivationType, NormalizationType
from nas_engine.datasets.loaders import LoaderSettings
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import EvaluationSettings
from nas_engine.exceptions import ConfigurationError
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
)
from nas_engine.search_space.presets import PRESETS, get_preset
from nas_engine.search_space.space import SearchSpace
from nas_engine.training.optimizers import OptimizerSettings, OptimizerType
from nas_engine.training.schedulers import SchedulerSettings, SchedulerType
from nas_engine.training.trainer import TrainingSettings
from nas_engine.utilities.hashing import stable_json_hash

#: Version of the configuration schema. Bump when a change would make an older stored
#: configuration reproduce a different run.
CONFIG_VERSION: int = 1


class _ConfigModel(BaseModel):
    """Base for configuration sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class ProjectConfig(_ConfigModel):
    """Identity and output location for a run.

    Attributes:
        name: Human-readable run name, persisted with the search.
        description: Free-text description included in reports.
        output_dir: Root directory for the database, artifacts, and reports.
    """

    name: str = Field(default="nas-run", min_length=1, max_length=200)
    description: str = ""
    output_dir: Path = Path("artifacts")


class DatasetConfig(_ConfigModel):
    """Which dataset to use and how to load it.

    Attributes:
        provider: Registered provider name, e.g. ``"synthetic"`` or ``"cifar10"``.
        batch_size: Examples per batch.
        num_workers: DataLoader worker processes. ``0`` keeps loading in the main process,
            which is faster for the small datasets used here and fully deterministic.
        pin_memory: Whether to pin host memory; only useful with CUDA.
        drop_last: Whether to drop a short final training batch.
        persistent_workers: Whether to keep DataLoader workers alive between epochs.
        options: Provider-specific keyword arguments passed straight to the provider.
    """

    provider: str = "synthetic"
    batch_size: Annotated[int, Field(ge=1, le=4096)] = 64
    num_workers: Annotated[int, Field(ge=0, le=32)] = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> DatasetConfig:
        """Reject settings that cannot work together.

        Returns:
            ``self``.

        Raises:
            ValueError: If persistent workers are requested without workers.
        """
        if self.persistent_workers and self.num_workers == 0:
            msg = (
                "dataset.persistent_workers requires dataset.num_workers > 0; with 0 "
                "workers there are no worker processes to keep alive"
            )
            raise ValueError(msg)
        return self

    def build_loader_settings(self) -> LoaderSettings:
        """Convert to the plain dataclass the datasets package consumes."""
        return LoaderSettings(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


class SearchSpaceConfig(_ConfigModel):
    """Which search space to search.

    A space is selected by preset name and refined by ``overrides``, a partial mapping
    merged over the preset's own fields. Inline definition of a whole space is supported by
    overriding every field, but presets keep the common case short and reviewable.

    Attributes:
        preset: Preset name from :data:`~nas_engine.search_space.presets.PRESETS`.
        overrides: Partial mapping merged over the preset before validation.
    """

    preset: str = "default_cnn"
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, value: str) -> str:
        """Reject unknown preset names.

        Args:
            value: Preset name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the preset is not registered.
        """
        if value not in PRESETS:
            msg = (
                f"search_space.preset={value!r} is not a known preset; available presets "
                f"are {sorted(PRESETS)}"
            )
            raise ValueError(msg)
        return value

    def build(
        self,
        *,
        input_size: int | None = None,
        num_classes: int | None = None,
        input_channels: int | None = None,
    ) -> SearchSpace:
        """Build the configured search space.

        Args:
            input_size: Override the preset's input extent, normally taken from the dataset.
            num_classes: Override the preset's class count.
            input_channels: Override the preset's channel count.

        Returns:
            The validated space.

        Raises:
            ConfigurationError: If the overrides do not produce a valid space.
        """
        base = get_preset(
            self.preset,
            input_size=input_size,
            num_classes=num_classes,
            input_channels=input_channels,
        )
        if not self.overrides:
            return base
        merged = {**base.model_dump(mode="python"), **self.overrides}
        try:
            return SearchSpace.model_validate(merged)
        except Exception as exc:
            msg = (
                f"search_space.overrides produced an invalid search space: {exc}. Check the "
                "field names against nas_engine.search_space.space.SearchSpace."
            )
            raise ConfigurationError(
                msg, details={"overrides": sorted(self.overrides), "error": str(exc)}
            ) from exc


class AlgorithmConfig(_ConfigModel):
    """Which search strategy to run.

    Attributes:
        name: Registered strategy name.
        params: Strategy-specific parameters, validated by the strategy's factory.
    """

    name: str = "random_search"
    params: dict[str, Any] = Field(default_factory=dict)


class BudgetConfig(_ConfigModel):
    """How much compute the search may spend.

    Attributes:
        max_evaluations: Total candidate evaluations.
        max_seconds: Wall-clock limit for the whole search; ``None`` for no limit.
        epochs: Training epochs per candidate at full fidelity.
        train_fraction: Fraction of the training split used at base fidelity.
        resolution: Input resolution at base fidelity; ``None`` for native.
        max_seconds_per_evaluation: Wall-clock limit for one evaluation.
    """

    max_evaluations: Annotated[int, Field(ge=1, le=1_000_000)] = 12
    max_seconds: float | None = Field(default=None, gt=0)
    epochs: Annotated[int, Field(ge=1, le=10_000)] = 3
    train_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    resolution: int | None = Field(default=None, ge=4)
    max_seconds_per_evaluation: float | None = Field(default=None, gt=0)

    def build_budget(self) -> TrainingBudget:
        """Convert to the base :class:`~nas_engine.evaluation.budget.TrainingBudget`."""
        return TrainingBudget(
            epochs=self.epochs,
            train_fraction=self.train_fraction,
            resolution=self.resolution,
            max_seconds=self.max_seconds_per_evaluation,
            rung=0,
        )


class OptimizerConfig(_ConfigModel):
    """Optimiser hyperparameters.

    Attributes:
        name: ``"sgd"`` or ``"adamw"``.
        learning_rate: Base learning rate.
        weight_decay: Decay applied to weight matrices only.
        momentum: SGD momentum.
        nesterov: Whether SGD uses Nesterov momentum.
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay.
        eps: AdamW numerical stability term.
        decay_normalization: Whether normalisation and bias parameters are decayed.
    """

    name: OptimizerType = OptimizerType.ADAMW
    learning_rate: Annotated[float, Field(gt=0, le=10.0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0, le=1.0)] = 1e-4
    momentum: Annotated[float, Field(ge=0, lt=1.0)] = 0.9
    nesterov: bool = True
    beta1: Annotated[float, Field(ge=0, lt=1.0)] = 0.9
    beta2: Annotated[float, Field(ge=0, lt=1.0)] = 0.999
    eps: Annotated[float, Field(gt=0, le=1.0)] = 1e-8
    decay_normalization: bool = False

    def build(self) -> OptimizerSettings:
        """Convert to the plain dataclass the training package consumes."""
        return OptimizerSettings(
            name=self.name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            decay_normalization=self.decay_normalization,
        )


class SchedulerConfig(_ConfigModel):
    """Learning-rate schedule hyperparameters.

    Attributes:
        name: ``"constant"``, ``"cosine"``, or ``"step"``.
        warmup_steps: Linear warm-up length in optimiser steps.
        min_lr_factor: Cosine floor as a fraction of the base rate.
        step_size_epochs: Step-schedule decay interval.
        gamma: Step-schedule decay factor.
    """

    name: SchedulerType = SchedulerType.COSINE
    warmup_steps: Annotated[int, Field(ge=0, le=100_000)] = 0
    min_lr_factor: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    step_size_epochs: Annotated[int, Field(ge=1)] = 10
    gamma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1

    def build(self) -> SchedulerSettings:
        """Convert to the plain dataclass the training package consumes."""
        return SchedulerSettings(
            name=self.name,
            warmup_steps=self.warmup_steps,
            min_lr_factor=self.min_lr_factor,
            step_size_epochs=self.step_size_epochs,
            gamma=self.gamma,
        )


class TrainingConfig(_ConfigModel):
    """How each candidate is trained.

    Attributes:
        optimizer: Optimiser settings.
        scheduler: Schedule settings.
        gradient_clip_norm: Global gradient-norm clip; ``None`` disables.
        label_smoothing: Cross-entropy label smoothing.
        early_stopping_patience: Epochs without improvement tolerated; ``0`` disables.
        early_stopping_min_delta: Minimum improvement that resets patience.
        mixed_precision: Whether to use autocast on CUDA.
        topk: ``k`` for recorded top-k accuracy.
        restore_best_weights: Whether to restore the best epoch's weights.
        checkpoint_every_epochs: Training-checkpoint interval; ``0`` writes only at the end.
        log_every_n_steps: Debug logging interval; ``0`` disables.
        zero_init_residual: Whether residual branches start as identities.
    """

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gradient_clip_norm: float | None = Field(default=5.0, gt=0)
    label_smoothing: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    early_stopping_patience: Annotated[int, Field(ge=0, le=1000)] = 0
    early_stopping_min_delta: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    mixed_precision: bool = False
    topk: Annotated[int, Field(ge=1, le=100)] = 5
    restore_best_weights: bool = True
    checkpoint_every_epochs: Annotated[int, Field(ge=0, le=1000)] = 0
    log_every_n_steps: Annotated[int, Field(ge=0, le=100_000)] = 0
    zero_init_residual: bool = True

    def build(self, *, epochs: int) -> TrainingSettings:
        """Convert to the plain dataclass the training package consumes.

        Args:
            epochs: Epoch budget for this evaluation.

        Returns:
            The training settings.
        """
        return TrainingSettings(
            epochs=epochs,
            optimizer=self.optimizer.build(),
            scheduler=self.scheduler.build(),
            gradient_clip_norm=self.gradient_clip_norm,
            label_smoothing=self.label_smoothing,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            mixed_precision=self.mixed_precision,
            topk=self.topk,
            restore_best_weights=self.restore_best_weights,
            checkpoint_every_epochs=self.checkpoint_every_epochs,
            log_every_n_steps=self.log_every_n_steps,
        )


class EvaluationConfig(_ConfigModel):
    """What the evaluator measures and stores.

    Attributes:
        measure_latency: Whether to benchmark inference latency.
        latency_batch_size: Batch size for the benchmark.
        latency_warmup_iterations: Untimed warm-up passes.
        latency_timed_iterations: Passes per timed block.
        latency_repeats: Number of timed blocks.
        measure_model_size: Whether to serialise the model to measure its size.
        save_weights: Whether to persist trained weights.
        save_training_checkpoints: Whether to keep resumable training checkpoints.
        max_parameters: Hard parameter ceiling enforced before building.
    """

    measure_latency: bool = True
    latency_batch_size: Annotated[int, Field(ge=1, le=1024)] = 1
    latency_warmup_iterations: Annotated[int, Field(ge=0, le=1000)] = 3
    latency_timed_iterations: Annotated[int, Field(ge=1, le=10_000)] = 5
    latency_repeats: Annotated[int, Field(ge=1, le=1000)] = 3
    measure_model_size: bool = True
    save_weights: bool = True
    save_training_checkpoints: bool = False
    max_parameters: int | None = Field(default=None, ge=1)

    def build(self, *, max_seconds: float | None) -> EvaluationSettings:
        """Convert to the plain dataclass the evaluation package consumes.

        Args:
            max_seconds: Per-evaluation wall-clock limit.

        Returns:
            The evaluation settings.
        """
        return EvaluationSettings(
            measure_latency=self.measure_latency,
            latency_batch_size=self.latency_batch_size,
            latency_warmup_iterations=self.latency_warmup_iterations,
            latency_timed_iterations=self.latency_timed_iterations,
            latency_repeats=self.latency_repeats,
            measure_model_size=self.measure_model_size,
            save_weights=self.save_weights,
            save_training_checkpoints=self.save_training_checkpoints,
            max_parameters=self.max_parameters,
            max_evaluation_seconds=max_seconds,
        )


class ObjectiveEntry(_ConfigModel):
    """One objective in configuration form.

    Attributes:
        metric: Metric key.
        direction: ``"maximize"`` or ``"minimize"``.
        weight: Relative importance.
        normalization: Rescaling strategy.
        reference: Divisor for reference normalisation.
        required: Whether a candidate missing this metric can be scored.
        missing_value: Value substituted for an absent optional metric.
    """

    metric: str = Field(min_length=1)
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    weight: Annotated[float, Field(ge=0.0)] = 1.0
    normalization: NormalizationStrategy = NormalizationStrategy.MINMAX
    reference: float | None = None
    required: bool = True
    missing_value: float | None = None

    def build(self) -> Objective:
        """Convert to the domain objective."""
        return Objective(
            metric=self.metric,
            direction=self.direction,
            weight=self.weight,
            normalization=self.normalization,
            reference=self.reference,
            required=self.required,
            missing_value=self.missing_value,
        )


class ConstraintEntry(_ConfigModel):
    """One hard constraint in configuration form.

    Attributes:
        metric: Metric key.
        operator: Comparison to apply.
        threshold: Value compared against.
        required: Whether a missing metric makes the candidate infeasible.
    """

    metric: str = Field(min_length=1)
    operator: ComparisonOperator = ComparisonOperator.LE
    threshold: float
    required: bool = True

    def build(self) -> MetricConstraint:
        """Convert to the domain constraint."""
        return MetricConstraint(
            metric=self.metric,
            operator=self.operator,
            threshold=self.threshold,
            required=self.required,
        )


class ObjectivesConfig(_ConfigModel):
    """The objectives and constraints a search optimises.

    Attributes:
        objectives: Objectives, most important first.
        constraints: Hard constraints on measured metrics.
    """

    objectives: list[ObjectiveEntry] = Field(
        default_factory=lambda: [
            ObjectiveEntry(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE),
            ObjectiveEntry(
                metric="trainable_parameters",
                direction=ObjectiveDirection.MINIMIZE,
                weight=0.2,
                normalization=NormalizationStrategy.LOG,
            ),
        ],
        min_length=1,
    )
    constraints: list[ConstraintEntry] = Field(default_factory=list)

    def build_objectives(self) -> ObjectiveSet:
        """Convert to the domain objective set.

        Returns:
            The objective set.

        Raises:
            ConfigurationError: If the objectives are inconsistent.
        """
        try:
            return ObjectiveSet(tuple(entry.build() for entry in self.objectives))
        except Exception as exc:
            msg = f"objectives configuration is invalid: {exc}"
            raise ConfigurationError(msg, details={"error": str(exc)}) from exc

    def build_constraints(self) -> ConstraintSet:
        """Convert to the domain constraint set."""
        return ConstraintSet(tuple(entry.build() for entry in self.constraints))


class PersistenceConfig(_ConfigModel):
    """Where results are stored.

    Attributes:
        database_path: SQLite file, relative to ``project.output_dir`` unless absolute.
        database_url: Explicit SQLAlchemy URL, overriding ``database_path``.
        artifact_dir: Directory for weights and checkpoints, relative to the output dir.
        report_dir: Directory for reports and exports, relative to the output dir.
        checkpoint_every: Checkpoint the search every N completed evaluations.
        keep_checkpoints: Number of recent checkpoints to retain.
        echo_sql: Whether to log every SQL statement.
    """

    database_path: Path = Path("nas.db")
    database_url: str | None = None
    artifact_dir: Path = Path("candidates")
    report_dir: Path = Path("reports")
    checkpoint_every: Annotated[int, Field(ge=1, le=10_000)] = 1
    keep_checkpoints: Annotated[int, Field(ge=1, le=1000)] = 5
    echo_sql: bool = False


class LoggingConfig(_ConfigModel):
    """Log level, format, and destination.

    Attributes:
        level: Minimum level name.
        format: ``"console"`` for human output, ``"json"`` for machine ingestion.
        file: Optional log file, relative to the output dir.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    file: Path | None = None


class HardwareConfig(_ConfigModel):
    """Device selection and thread limits.

    Attributes:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.
        torch_threads: Intra-op thread count; ``None`` leaves the PyTorch default.
    """

    device: str = "auto"
    torch_threads: int | None = Field(default=None, ge=1, le=256)

    def resolve_device(self) -> torch.device:
        """Resolve the configured device to a concrete :class:`torch.device`.

        ``"auto"`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicitly requested
        accelerator that is unavailable is an error rather than a silent CPU fallback: a
        run that silently takes a hundred times longer than expected is worse than one that
        refuses to start.

        Returns:
            The resolved device.

        Raises:
            ConfigurationError: If a requested accelerator is unavailable or the string is
                not a valid device specification.
        """
        requested = self.device.strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():  # pragma: no cover - depends on the host
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():  # pragma: no cover - Apple only
                return torch.device("mps")
            return torch.device("cpu")

        if requested.startswith("cuda") and not torch.cuda.is_available():
            msg = (
                f"hardware.device={self.device!r} was requested but CUDA is not available "
                "in this environment. Use 'cpu', or 'auto' to select the best available "
                "device."
            )
            raise ConfigurationError(msg, details={"device": self.device})
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                msg = (
                    "hardware.device='mps' was requested but Apple Silicon MPS is not "
                    "available. Use 'cpu', or 'auto'."
                )
                raise ConfigurationError(msg, details={"device": self.device})

        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            msg = (
                f"hardware.device={self.device!r} is not a valid device specification; "
                "expected 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'"
            )
            raise ConfigurationError(msg, details={"device": self.device}) from exc


class ConcurrencyConfig(_ConfigModel):
    """How evaluations are executed.

    Attributes:
        mode: ``"sequential"`` or ``"multiprocessing"``.
        workers: Worker processes when multiprocessing.
        start_method: Process start method. ``"spawn"`` is the default because ``"fork"``
            is unsafe once CUDA or threaded BLAS has been initialised in the parent.
        max_in_flight: Cap on simultaneously dispatched evaluations; defaults to ``workers``.
    """

    mode: Literal["sequential", "multiprocessing"] = "sequential"
    workers: Annotated[int, Field(ge=1, le=64)] = 1
    start_method: Literal["spawn", "forkserver", "fork"] = "spawn"
    max_in_flight: int | None = Field(default=None, ge=1, le=256)

    @model_validator(mode="after")
    def _validate(self) -> ConcurrencyConfig:
        """Warn-by-erroring on contradictory concurrency settings.

        Returns:
            ``self``.

        Raises:
            ValueError: If multiple workers are requested in sequential mode.
        """
        if self.mode == "sequential" and self.workers != 1:
            msg = (
                f"concurrency.workers={self.workers} has no effect with "
                "concurrency.mode='sequential'; set mode='multiprocessing' or leave "
                "workers at 1"
            )
            raise ValueError(msg)
        return self

    @property
    def effective_in_flight(self) -> int:
        """Maximum simultaneously dispatched evaluations."""
        if self.mode == "sequential":
            return 1
        return self.max_in_flight if self.max_in_flight is not None else self.workers


class ReproducibilityConfig(_ConfigModel):
    """Seeding and determinism.

    Attributes:
        seed: Master seed; every component seed derives from it.
        deterministic: Whether to request deterministic PyTorch kernels.
        warn_only: Whether operations without a deterministic implementation warn instead
            of raising.
    """

    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 42
    deterministic: bool = True
    warn_only: bool = True


class RetryConfig(_ConfigModel):
    """Retry policy for failed evaluations.

    Attributes:
        max_retries: Retries allowed per candidate, beyond the first attempt.
        retry_on_timeout: Whether a timeout is retriable.
        retry_on_resource_error: Whether an out-of-memory failure is retriable.
        backoff_seconds: Delay before the first retry.
        backoff_multiplier: Multiplier applied per subsequent retry.
        max_backoff_seconds: Cap on the delay.
    """

    max_retries: Annotated[int, Field(ge=0, le=20)] = 1
    retry_on_timeout: bool = True
    retry_on_resource_error: bool = True
    backoff_seconds: Annotated[float, Field(ge=0.0, le=3600.0)] = 0.0
    backoff_multiplier: Annotated[float, Field(ge=1.0, le=10.0)] = 2.0
    max_backoff_seconds: Annotated[float, Field(ge=0.0, le=86400.0)] = 60.0


class SearchConfig(_ConfigModel):
    """The complete configuration for one search run.

    Attributes:
        version: Configuration schema version.
        project: Identity and output location.
        dataset: Dataset selection and loading.
        search_space: Which space to search.
        algorithm: Which strategy to run.
        budget: Compute allowance.
        training: Per-candidate training recipe.
        evaluation: What to measure and store.
        objectives: Objectives and hard constraints.
        persistence: Storage locations.
        logging: Log configuration.
        hardware: Device selection.
        concurrency: Execution mode.
        reproducibility: Seeding and determinism.
        retry: Retry policy.
    """

    version: int = CONFIG_VERSION
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    objectives: ObjectivesConfig = Field(default_factory=ObjectivesConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def _validate(self) -> SearchConfig:
        """Cross-section validation.

        Returns:
            ``self``.

        Raises:
            ValueError: If the version is unsupported or sections contradict each other.
        """
        if self.version > CONFIG_VERSION:
            msg = (
                f"configuration version {self.version} is newer than the supported version "
                f"{CONFIG_VERSION}; upgrade nas-engine to read this file"
            )
            raise ValueError(msg)
        if self.version < 1:
            msg = f"configuration version must be >= 1, received {self.version}"
            raise ValueError(msg)
        if self.training.mixed_precision and self.hardware.device == "cpu":
            msg = (
                "training.mixed_precision requires a CUDA device but hardware.device is "
                "'cpu'. Set mixed_precision=false, or use device='auto'."
            )
            raise ValueError(msg)
        if self.evaluation.save_training_checkpoints and not self.evaluation.save_weights:
            # Not fatal, but almost certainly a mistake: checkpoints exist to resume
            # training, and discarding the final weights makes that pointless.
            msg = (
                "evaluation.save_training_checkpoints=true with save_weights=false stores "
                "resumable checkpoints but discards the trained model; enable save_weights"
            )
            raise ValueError(msg)
        return self

    # -- derived paths -------------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        """Resolved root output directory."""
        return self.project.output_dir.expanduser().resolve()

    @property
    def artifact_dir(self) -> Path:
        """Resolved artifact directory."""
        directory = self.persistence.artifact_dir
        return directory if directory.is_absolute() else self.output_dir / directory

    @property
    def report_dir(self) -> Path:
        """Resolved report directory."""
        directory = self.persistence.report_dir
        return directory if directory.is_absolute() else self.output_dir / directory

    @property
    def database_url(self) -> str:
        """Resolved SQLAlchemy URL for the results database."""
        if self.persistence.database_url:
            return self.persistence.database_url
        path = self.persistence.database_path
        resolved = path if path.is_absolute() else self.output_dir / path
        return f"sqlite+pysqlite:///{resolved}"

    @property
    def log_file(self) -> Path | None:
        """Resolved log-file path, or ``None``."""
        if self.logging.file is None:
            return None
        path = self.logging.file
        return path if path.is_absolute() else self.output_dir / path

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | str, *, use_environment: bool = True) -> SearchConfig:
        """Load a configuration from a YAML file through the full precedence chain.

        This is the entry point advertised in the public API:

        .. code-block:: python

            config = SearchConfig.from_yaml("configs/random_search.yaml")

        Args:
            path: YAML file to read.
            use_environment: Whether ``NAS_ENGINE__*`` environment variables may override
                file values.

        Returns:
            The validated configuration.

        Raises:
            ConfigurationError: If the file is missing or the configuration is invalid.
        """
        from nas_engine.config.loader import load_config

        return load_config(path, use_environment=use_environment)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> SearchConfig:
        """Validate a plain mapping into a configuration.

        Args:
            payload: Configuration data.

        Returns:
            The validated configuration.

        Raises:
            ConfigurationError: If validation fails.
        """
        from nas_engine.config.loader import build_config

        return build_config(payload, source="mapping")

    def to_yaml(self, path: Path | None = None) -> str:
        """Render the configuration as YAML, optionally writing it to disk.

        Args:
            path: Destination file; ``None`` renders without writing.

        Returns:
            The YAML text.

        Raises:
            ConfigurationError: If the file cannot be written.
        """
        from nas_engine.config.loader import dump_yaml

        return dump_yaml(self, path)

    # -- serialisation --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation with paths as strings."""
        dumped: dict[str, Any] = self.model_dump(mode="json")
        return dumped

    def config_hash(self) -> str:
        """Return a stable hash of the configuration.

        Used to detect that a stored configuration was edited between a search and its
        resume — which would make the two halves of the run incomparable.

        Returns:
            A 32-character hexadecimal digest.
        """
        return stable_json_hash(self.to_dict())

    def describe(self) -> str:
        """Return a compact multi-line summary for CLI output."""
        objective_summary = ", ".join(
            f"{objective.direction.value} {objective.metric}"
            for objective in self.objectives.objectives
        )
        return "\n".join(
            [
                f"project        : {self.project.name}",
                f"dataset        : {self.dataset.provider} (batch {self.dataset.batch_size})",
                f"search space   : {self.search_space.preset}",
                f"algorithm      : {self.algorithm.name}",
                f"budget         : {self.budget.max_evaluations} evaluations x "
                f"{self.budget.epochs} epochs",
                "objectives     : " + objective_summary,
                f"device         : {self.hardware.device}",
                f"concurrency    : {self.concurrency.mode} ({self.concurrency.workers} worker(s))",
                f"seed           : {self.reproducibility.seed} "
                f"(deterministic={self.reproducibility.deterministic})",
                f"output         : {self.output_dir}",
                f"config hash    : {self.config_hash()}",
            ]
        )


#: Default activation and normalisation choices are re-exported so configuration files can
#: reference them by name without importing from the architectures package.
__all__ = [
    "CONFIG_VERSION",
    "ActivationType",
    "AlgorithmConfig",
    "BudgetConfig",
    "ConcurrencyConfig",
    "ConstraintEntry",
    "DatasetConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "LoggingConfig",
    "NormalizationType",
    "ObjectiveEntry",
    "ObjectivesConfig",
    "OptimizerConfig",
    "PersistenceConfig",
    "ProjectConfig",
    "ReproducibilityConfig",
    "RetryConfig",
    "SchedulerConfig",
    "SearchConfig",
    "SearchSpaceConfig",
    "TrainingConfig",
]
