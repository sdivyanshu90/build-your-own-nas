r"""The training loop.

Separation of concerns
----------------------
The trainer knows how to fit *a* model to *a* dataset. It knows nothing about search
strategies, candidates, databases, or objectives. That boundary is deliberate and is one
of the project's core design rules: a search strategy that contained training code could
not be unit-tested without a GPU, and a trainer that knew about candidates could not be
reused to train the final winner.

What the loop provides
----------------------
* Epoch-wise training and validation with example-weighted metric aggregation.
* Gradient clipping by global norm.
* Mixed precision on CUDA, with loss scaling.
* Early stopping with best-weight restoration.
* Per-step learning-rate scheduling.
* Checkpointing and exact resume.
* A wall-clock deadline.
* Divergence detection.

Gradient clipping
-----------------
Clipping rescales the gradient when its global :math:`L_2` norm exceeds a threshold:
:math:`g \leftarrow g \cdot \min(1, c/\lVert g \rVert)`. It preserves direction and only
bounds magnitude. In NAS it earns its place because the search deliberately proposes
unusual networks — very deep stacks without residuals, or unnormalised convolutions — and
a single exploding batch would otherwise turn an interesting candidate into a ``NaN`` and
lose the information it carried.

Divergence
----------
A non-finite loss is treated as a **permanent** failure, not a retriable one. The same
architecture with the same seed will diverge again, so retrying only burns budget. The
candidate is recorded as failed with a specific error code so the report can distinguish
"this architecture is unstable" from "the machine ran out of memory".
"""

from __future__ import annotations

import contextlib
import copy
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from nas_engine.datasets.loaders import DataLoaders
from nas_engine.exceptions import (
    ConfigurationError,
    EvaluationTimeoutError,
    NonFiniteLossError,
    TrainingError,
)
from nas_engine.observability.logging import get_logger
from nas_engine.training.checkpointing import (
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from nas_engine.training.early_stopping import EarlyStopping, MonitorMode
from nas_engine.training.metrics import (
    EpochMetrics,
    MetricAggregator,
    accuracy,
    topk_accuracy,
)
from nas_engine.training.optimizers import OptimizerSettings, build_optimizer
from nas_engine.training.schedulers import SchedulerSettings, build_scheduler
from nas_engine.utilities.timing import Stopwatch

_LOGGER = get_logger(__name__)

#: How often, in optimiser steps, the wall-clock deadline is checked. Checking every step
#: would add a syscall to the hot loop; checking only per epoch would let a single long
#: epoch blow through the deadline.
_DEADLINE_CHECK_INTERVAL: int = 20


@dataclass(frozen=True)
class TrainingSettings:
    """Everything the trainer needs, expressed as plain data.

    This is deliberately **not** a Pydantic model. The training package must not depend on
    the configuration framework, so the config layer converts its validated models into
    this dataclass at the boundary. That keeps the trainer usable from a plain script and
    testable without constructing a full configuration tree.

    Attributes:
        epochs: Maximum epochs to run.
        optimizer: Optimiser hyperparameters.
        scheduler: Learning-rate schedule hyperparameters.
        gradient_clip_norm: Global gradient-norm clip threshold, or ``None`` to disable.
        label_smoothing: Cross-entropy label smoothing in ``[0, 1)``.
        early_stopping_patience: Epochs without improvement tolerated; ``0`` disables.
        early_stopping_min_delta: Minimum improvement that resets the patience counter.
        mixed_precision: Whether to use autocast and loss scaling on CUDA.
        topk: ``k`` for the recorded top-k accuracy.
        max_seconds: Wall-clock limit for one ``fit`` call, or ``None`` for no limit.
        restore_best_weights: Whether to restore the best epoch's weights when finished.
        checkpoint_every_epochs: Write a checkpoint every N epochs; ``0`` writes only at
            the end.
        log_every_n_steps: Emit a debug log line every N optimiser steps; ``0`` disables.
    """

    epochs: int = 5
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    gradient_clip_norm: float | None = 5.0
    label_smoothing: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    mixed_precision: bool = False
    topk: int = 5
    max_seconds: float | None = None
    restore_best_weights: bool = True
    checkpoint_every_epochs: int = 0
    log_every_n_steps: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ConfigurationError: If any value is out of range.
        """
        if self.epochs < 1:
            msg = f"epochs must be >= 1, received {self.epochs}"
            raise ConfigurationError(msg, details={"epochs": self.epochs})
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            msg = f"gradient_clip_norm must be positive or None, received {self.gradient_clip_norm}"
            raise ConfigurationError(msg, details={"gradient_clip_norm": self.gradient_clip_norm})
        if not 0.0 <= self.label_smoothing < 1.0:
            msg = f"label_smoothing must lie in [0, 1), received {self.label_smoothing}"
            raise ConfigurationError(msg, details={"label_smoothing": self.label_smoothing})
        if self.topk < 1:
            msg = f"topk must be >= 1, received {self.topk}"
            raise ConfigurationError(msg, details={"topk": self.topk})
        if self.max_seconds is not None and self.max_seconds <= 0:
            msg = f"max_seconds must be positive or None, received {self.max_seconds}"
            raise ConfigurationError(msg, details={"max_seconds": self.max_seconds})
        if self.checkpoint_every_epochs < 0:
            msg = (
                "checkpoint_every_epochs must be non-negative, received "
                f"{self.checkpoint_every_epochs}"
            )
            raise ConfigurationError(
                msg, details={"checkpoint_every_epochs": self.checkpoint_every_epochs}
            )

    def with_epochs(self, epochs: int) -> TrainingSettings:
        """Return a copy with a different epoch budget.

        Used by multi-fidelity search, which trains the same architecture for
        progressively larger epoch budgets.

        Args:
            epochs: New epoch budget.

        Returns:
            A new :class:`TrainingSettings`.
        """
        return TrainingSettings(
            epochs=epochs,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            gradient_clip_norm=self.gradient_clip_norm,
            label_smoothing=self.label_smoothing,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            mixed_precision=self.mixed_precision,
            topk=self.topk,
            max_seconds=self.max_seconds,
            restore_best_weights=self.restore_best_weights,
            checkpoint_every_epochs=self.checkpoint_every_epochs,
            log_every_n_steps=self.log_every_n_steps,
        )


@dataclass(frozen=True)
class TrainingOutcome:
    """The result of a completed ``fit`` call.

    Attributes:
        epochs_completed: Number of epochs actually run.
        global_step: Total optimiser steps taken.
        history: Every epoch's training and validation metrics, in order.
        best_epoch: Epoch index with the best validation accuracy.
        best_validation_accuracy: Best validation accuracy observed.
        best_validation_loss: Validation loss at the best epoch.
        best_validation_topk: Top-k accuracy at the best epoch.
        final_train_loss: Training loss of the final epoch.
        stopped_early: Whether early stopping ended the run.
        duration_seconds: Wall-clock duration of the whole call.
        restored_best_weights: Whether the best epoch's weights were restored.
    """

    epochs_completed: int
    global_step: int
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_accuracy: float
    best_validation_loss: float
    best_validation_topk: float
    final_train_loss: float
    stopped_early: bool
    duration_seconds: float
    restored_best_weights: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "epochs_completed": self.epochs_completed,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_validation_accuracy,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_topk": self.best_validation_topk,
            "final_train_loss": self.final_train_loss,
            "stopped_early": self.stopped_early,
            "duration_seconds": self.duration_seconds,
            "restored_best_weights": self.restored_best_weights,
            "history": [entry.to_dict() for entry in self.history],
        }


def _make_grad_scaler(enabled: bool) -> Any:
    """Create a gradient scaler compatible with the installed PyTorch version.

    ``torch.amp.GradScaler`` is the modern spelling; older releases only provide
    ``torch.cuda.amp.GradScaler``. Both are looked up defensively so the trainer works
    across the supported PyTorch range.

    Args:
        enabled: Whether scaling should be active.

    Returns:
        A gradient scaler instance.
    """
    modern = getattr(torch.amp, "GradScaler", None)
    if modern is not None:
        with contextlib.suppress(TypeError):
            return modern("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Trains and evaluates a single model.

    Args:
        settings: Training hyperparameters.
        device: Device to train on.
        seed: Seed recorded in logs; the caller is responsible for having seeded the RNGs.

    Raises:
        ConfigurationError: If mixed precision is requested on a device that cannot
            support it.
    """

    def __init__(
        self,
        settings: TrainingSettings,
        *,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.settings = settings
        self.device = torch.device(device)
        self.seed = seed
        self._amp_enabled = settings.mixed_precision and self.device.type == "cuda"
        if settings.mixed_precision and self.device.type != "cuda":
            # Not an error: falling back keeps a CUDA-tuned configuration usable on CPU,
            # which is exactly what CI does. It is logged so the fallback is visible.
            _LOGGER.info(
                "mixed_precision.disabled",
                reason="autocast with loss scaling is only supported on CUDA",
                device=str(self.device),
            )

    # -- loss ----------------------------------------------------------------------
    def _criterion(self) -> nn.Module:
        """Return the loss function.

        Cross-entropy is the maximum-likelihood loss for a categorical distribution: it is
        the negative log-probability the model assigns to the true class. Label smoothing
        replaces the one-hot target with a mixture of the target and the uniform
        distribution, which discourages the network from driving logits to infinity and
        acts as a mild regulariser.

        Returns:
            The configured loss module.
        """
        return nn.CrossEntropyLoss(label_smoothing=self.settings.label_smoothing)

    def _autocast(self) -> Any:
        """Return the autocast context manager, or a null context when AMP is off."""
        if not self._amp_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast(  # type: ignore[attr-defined]
            device_type=self.device.type, enabled=True
        )

    # -- phases --------------------------------------------------------------------
    def _train_one_epoch(  # noqa: PLR0915 - the training loop is one procedure
        self,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, int]],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        *,
        epoch: int,
        global_step: int,
        deadline: float | None,
    ) -> tuple[EpochMetrics, int]:
        """Run one training epoch.

        Args:
            model: Model in training mode.
            loader: Training loader.
            optimizer: Optimiser.
            scheduler: Per-step scheduler.
            scaler: Gradient scaler (inactive when AMP is off).
            epoch: Epoch index.
            global_step: Steps completed before this epoch.
            deadline: Monotonic deadline, or ``None``.

        Returns:
            The epoch metrics and the updated global step counter.

        Raises:
            NonFiniteLossError: If the loss becomes NaN or infinite.
            EvaluationTimeoutError: If the deadline passes.
            TrainingError: If the backward or optimiser step fails.
        """
        model.train()
        criterion = self._criterion()
        aggregator = MetricAggregator()
        watch = Stopwatch().start()
        steps_in_epoch = 0

        for raw_inputs, raw_targets in loader:
            inputs = raw_inputs.to(self.device, non_blocking=True)
            targets = raw_targets.to(self.device, non_blocking=True)
            batch_size = int(targets.shape[0])
            if batch_size == 0:  # pragma: no cover - DataLoader never yields empty batches
                continue

            optimizer.zero_grad(set_to_none=True)
            try:
                with self._autocast():
                    logits = model(inputs)
                    loss = criterion(logits, targets)
            except RuntimeError as exc:
                msg = f"forward pass failed at epoch {epoch}, step {global_step}: {exc}"
                raise TrainingError(
                    msg, details={"epoch": epoch, "step": global_step, "error": str(exc)}
                ) from exc

            loss_value = float(loss.detach().item())
            if not torch.isfinite(loss.detach()):
                msg = (
                    f"loss became non-finite ({loss_value}) at epoch {epoch}, step "
                    f"{global_step}. This architecture is numerically unstable under the "
                    "current recipe; lower the learning rate or enable gradient clipping."
                )
                raise NonFiniteLossError(
                    msg, details={"epoch": epoch, "step": global_step, "loss": loss_value}
                )

            try:
                if self._amp_enabled:
                    scaler.scale(loss).backward()
                    if self.settings.gradient_clip_norm is not None:
                        # Gradients must be unscaled before their norm is meaningful.
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), self.settings.gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.settings.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), self.settings.gradient_clip_norm
                        )
                    optimizer.step()
            except RuntimeError as exc:
                msg = f"optimisation step failed at epoch {epoch}, step {global_step}: {exc}"
                raise TrainingError(
                    msg, details={"epoch": epoch, "step": global_step, "error": str(exc)}
                ) from exc

            scheduler.step()
            global_step += 1
            steps_in_epoch += 1

            with torch.no_grad():
                detached = logits.detach().float()
                aggregator.update(
                    {
                        "loss": loss_value,
                        "accuracy": accuracy(detached, targets),
                        "topk": topk_accuracy(detached, targets, self.settings.topk),
                    },
                    batch_size=batch_size,
                )

            log_interval = self.settings.log_every_n_steps
            if log_interval and global_step % log_interval == 0:
                _LOGGER.debug(
                    "training.step",
                    epoch=epoch,
                    step=global_step,
                    loss=loss_value,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
            if deadline is not None and steps_in_epoch % _DEADLINE_CHECK_INTERVAL == 0:
                self._check_deadline(deadline, epoch=epoch, step=global_step)

        values = aggregator.compute()
        metrics = EpochMetrics(
            epoch=epoch,
            phase="train",
            loss=values.get("loss", float("nan")),
            accuracy=values.get("accuracy", 0.0),
            topk_accuracy=values.get("topk", 0.0),
            examples=aggregator.examples,
            duration_seconds=watch.stop(),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        return metrics, global_step

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, int]],
        *,
        epoch: int = 0,
        phase: str = "validation",
    ) -> EpochMetrics:
        """Evaluate a model on a loader without updating weights.

        Args:
            model: Model to evaluate.
            loader: Data to evaluate on.
            epoch: Epoch index recorded in the returned metrics.
            phase: Phase label recorded in the returned metrics.

        Returns:
            The evaluation metrics.

        Raises:
            TrainingError: If the forward pass fails.
        """
        model.eval()
        criterion = self._criterion()
        aggregator = MetricAggregator()
        watch = Stopwatch().start()

        for raw_inputs, raw_targets in loader:
            inputs = raw_inputs.to(self.device, non_blocking=True)
            targets = raw_targets.to(self.device, non_blocking=True)
            batch_size = int(targets.shape[0])
            if batch_size == 0:  # pragma: no cover - DataLoader never yields empty batches
                continue
            try:
                with self._autocast():
                    logits = model(inputs)
                    loss = criterion(logits, targets)
            except RuntimeError as exc:
                msg = f"evaluation forward pass failed in phase '{phase}': {exc}"
                raise TrainingError(msg, details={"phase": phase, "error": str(exc)}) from exc
            detached = logits.detach().float()
            aggregator.update(
                {
                    "loss": float(loss.detach().item()),
                    "accuracy": accuracy(detached, targets),
                    "topk": topk_accuracy(detached, targets, self.settings.topk),
                },
                batch_size=batch_size,
            )

        values = aggregator.compute()
        return EpochMetrics(
            epoch=epoch,
            phase=phase,
            loss=values.get("loss", float("nan")),
            accuracy=values.get("accuracy", 0.0),
            topk_accuracy=values.get("topk", 0.0),
            examples=aggregator.examples,
            duration_seconds=watch.stop(),
        )

    # -- deadline ------------------------------------------------------------------
    def _check_deadline(self, deadline: float, *, epoch: int, step: int) -> None:
        """Raise when the wall-clock deadline has passed.

        Args:
            deadline: Monotonic deadline from :func:`time.perf_counter`.
            epoch: Current epoch, for the error message.
            step: Current step, for the error message.

        Raises:
            EvaluationTimeoutError: If the deadline has passed.
        """
        if time.perf_counter() < deadline:
            return
        limit = self.settings.max_seconds
        msg = (
            f"training exceeded its {limit:.1f}s budget at epoch {epoch}, step {step}. "
            "Raise training.max_seconds, reduce the epoch budget, or tighten the "
            "search-space parameter limit."
        )
        raise EvaluationTimeoutError(
            msg, details={"epoch": epoch, "step": step, "max_seconds": limit}
        )

    # -- main entry point ----------------------------------------------------------
    def fit(  # noqa: PLR0912, PLR0915 - setup, loop, and teardown in one readable pass
        self,
        model: nn.Module,
        loaders: DataLoaders,
        *,
        architecture_hash: str,
        checkpoint_path: Path | None = None,
        resume: bool = True,
        epochs: int | None = None,
    ) -> TrainingOutcome:
        """Train ``model`` and return its outcome.

        Args:
            model: Model to train; modified in place.
            loaders: Training and validation loaders.
            architecture_hash: Hash recorded in checkpoints for identity checking.
            checkpoint_path: Where to write (and read) the training checkpoint.
            resume: Whether to resume from an existing checkpoint at ``checkpoint_path``.
            epochs: Override the configured epoch budget, used by multi-fidelity search.

        Returns:
            A :class:`TrainingOutcome`.

        Raises:
            ConfigurationError: If the training loader is empty.
            NonFiniteLossError: If training diverges.
            EvaluationTimeoutError: If the wall-clock budget is exhausted.
            TrainingError: If a forward or backward pass fails.
        """
        total_epochs = epochs if epochs is not None else self.settings.epochs
        if total_epochs < 1:
            msg = f"epochs must be >= 1, received {total_epochs}"
            raise ConfigurationError(msg, details={"epochs": total_epochs})

        steps_per_epoch = len(loaders.train)
        if steps_per_epoch < 1:
            msg = (
                "the training loader yields no batches; the dataset split is empty or the "
                "batch size exceeds the split size with drop_last enabled"
            )
            raise ConfigurationError(msg, details={"steps_per_epoch": steps_per_epoch})

        model = model.to(self.device)
        optimizer = build_optimizer(model, self.settings.optimizer)
        scheduler = build_scheduler(
            optimizer,
            self.settings.scheduler,
            total_steps=total_epochs * steps_per_epoch,
            steps_per_epoch=steps_per_epoch,
        )
        scaler = _make_grad_scaler(self._amp_enabled)
        stopper = EarlyStopping(
            patience=self.settings.early_stopping_patience,
            min_delta=self.settings.early_stopping_min_delta,
            mode=MonitorMode.MAX,
        )

        history: list[EpochMetrics] = []
        start_epoch = 0
        global_step = 0

        if resume and checkpoint_path is not None and checkpoint_path.is_file():
            checkpoint = load_checkpoint(checkpoint_path, expected_hash=architecture_hash)
            model.load_state_dict(checkpoint.model_state)
            if checkpoint.optimizer_state:
                optimizer.load_state_dict(checkpoint.optimizer_state)
            if checkpoint.scheduler_state:
                scheduler.load_state_dict(checkpoint.scheduler_state)
            if checkpoint.early_stopping_state:
                stopper.load_state_dict(checkpoint.early_stopping_state)
            start_epoch = checkpoint.epoch
            global_step = checkpoint.global_step
            history = [
                EpochMetrics(
                    epoch=int(entry["epoch"]),
                    phase=str(entry["phase"]),
                    loss=float(entry["loss"]),
                    accuracy=float(entry["accuracy"]),
                    topk_accuracy=float(entry["topk_accuracy"]),
                    examples=int(entry["examples"]),
                    duration_seconds=float(entry["duration_seconds"]),
                    learning_rate=(
                        float(entry["learning_rate"])
                        if entry.get("learning_rate") is not None
                        else None
                    ),
                )
                for entry in checkpoint.history
            ]
            _LOGGER.info(
                "training.resumed",
                architecture_hash=architecture_hash,
                epoch=start_epoch,
                global_step=global_step,
            )

        watch = Stopwatch().start()
        deadline = (
            time.perf_counter() + self.settings.max_seconds
            if self.settings.max_seconds is not None
            else None
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: EpochMetrics | None = None
        stopped_early = False
        epochs_completed = start_epoch

        for epoch in range(start_epoch, total_epochs):
            train_metrics, global_step = self._train_one_epoch(
                model,
                loaders.train,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                global_step=global_step,
                deadline=deadline,
            )
            history.append(train_metrics)

            validation_metrics = self.evaluate(model, loaders.validation, epoch=epoch)
            history.append(validation_metrics)
            epochs_completed = epoch + 1

            improved = stopper.update(validation_metrics.accuracy, epoch=epoch)
            if improved:
                best_validation = validation_metrics
                if self.settings.restore_best_weights:
                    # A CPU copy so that the snapshot survives the model moving devices and
                    # does not pin GPU memory for the rest of the run.
                    best_state = {
                        name: tensor.detach().to("cpu", copy=True)
                        for name, tensor in model.state_dict().items()
                    }

            _LOGGER.debug(
                "training.epoch",
                architecture_hash=architecture_hash,
                epoch=epoch,
                train_loss=train_metrics.loss,
                validation_accuracy=validation_metrics.accuracy,
                improved=improved,
            )

            should_checkpoint = checkpoint_path is not None and (
                self.settings.checkpoint_every_epochs > 0
                and (epoch + 1) % self.settings.checkpoint_every_epochs == 0
            )
            if should_checkpoint:
                assert checkpoint_path is not None
                self._write_checkpoint(
                    checkpoint_path,
                    architecture_hash=architecture_hash,
                    epoch=epochs_completed,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stopper=stopper,
                    history=history,
                )

            if stopper.should_stop:
                stopped_early = True
                _LOGGER.info(
                    "training.early_stop",
                    architecture_hash=architecture_hash,
                    epoch=epoch,
                    best_epoch=stopper.best_epoch,
                    best_value=stopper.best_value,
                )
                break

            if deadline is not None:
                self._check_deadline(deadline, epoch=epoch, step=global_step)

        restored = False
        if self.settings.restore_best_weights and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)
            restored = True

        if checkpoint_path is not None:
            self._write_checkpoint(
                checkpoint_path,
                architecture_hash=architecture_hash,
                epoch=epochs_completed,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stopper=stopper,
                history=history,
            )

        if best_validation is None:  # pragma: no cover - only when zero epochs ran
            best_validation = self.evaluate(model, loaders.validation, epoch=epochs_completed)

        train_history = [entry for entry in history if entry.phase == "train"]
        return TrainingOutcome(
            epochs_completed=epochs_completed,
            global_step=global_step,
            history=tuple(history),
            best_epoch=stopper.best_epoch if stopper.best_epoch >= 0 else 0,
            best_validation_accuracy=best_validation.accuracy,
            best_validation_loss=best_validation.loss,
            best_validation_topk=best_validation.topk_accuracy,
            final_train_loss=train_history[-1].loss if train_history else float("nan"),
            stopped_early=stopped_early,
            duration_seconds=watch.stop(),
            restored_best_weights=restored,
        )

    def _write_checkpoint(
        self,
        path: Path,
        *,
        architecture_hash: str,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        stopper: EarlyStopping,
        history: list[EpochMetrics],
    ) -> None:
        """Write a training checkpoint.

        Args:
            path: Destination file.
            architecture_hash: Architecture identity recorded in the checkpoint.
            epoch: Number of completed epochs.
            global_step: Number of completed optimiser steps.
            model: Model whose weights are saved.
            optimizer: Optimiser whose state is saved.
            scheduler: Scheduler whose state is saved.
            stopper: Early-stopping state.
            history: Metrics recorded so far.
        """
        checkpoint = TrainingCheckpoint(
            architecture_hash=architecture_hash,
            epoch=epoch,
            global_step=global_step,
            model_state={
                name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()
            },
            optimizer_state=copy.deepcopy(optimizer.state_dict()),
            scheduler_state=copy.deepcopy(scheduler.state_dict()),
            early_stopping_state=dict(stopper.state_dict()),
            history=[entry.to_dict() for entry in history],
        )
        save_checkpoint(path, checkpoint)


@contextlib.contextmanager
def evaluation_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily switch a model to evaluation mode and restore the previous mode.

    A context manager rather than bare ``model.eval()`` calls because latency measurement
    and test evaluation must not leave a training-mode model in eval mode by accident —
    a mistake that silently freezes BatchNorm statistics for the rest of a run.

    Args:
        model: Model to switch.

    Yields:
        The model, in evaluation mode.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


__all__ = ["Trainer", "TrainingOutcome", "TrainingSettings", "evaluation_mode"]
