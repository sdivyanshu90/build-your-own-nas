r"""Learning-rate schedule construction.

Why a schedule matters more in NAS than elsewhere
--------------------------------------------------
Candidates are trained for very few epochs. A constant learning rate leaves every model
still oscillating around a minimum when training stops, and the size of that oscillation
depends on the architecture — so the ranking picks up noise proportional to each model's
curvature rather than its quality. Annealing the learning rate to (near) zero forces every
candidate into a comparable "settled" state before it is measured, which materially
reduces ranking noise.

Schedules
---------
``cosine``
    :math:`\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_0 - \eta_{min})(1 + \cos(\pi t/T))`
    where :math:`t` is the current step and :math:`T` the total. Smooth, hyperparameter-free
    beyond the horizon, and the standard choice for short budgets.
``step``
    Multiply by ``gamma`` every ``step_size`` epochs. Simple and predictable; useful when
    reproducing a published recipe.
``constant``
    No decay. Kept as a baseline and for debugging.

Warm-up
-------
An optional linear warm-up over the first few *steps* (not epochs — short budgets may only
have a few hundred steps in total). Warm-up exists because adaptive optimisers have
unreliable second-moment estimates in the first iterations, which can produce a very large
first step and destabilise BatchNorm statistics.

Stepping granularity
--------------------
All schedules here are stepped **per optimiser step**, not per epoch, so that the shape of
the schedule does not change when the dataset fraction changes under multi-fidelity
evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch.optim.lr_scheduler import LambdaLR

from nas_engine.exceptions import ConfigurationError


class SchedulerType(str, Enum):
    """Supported learning-rate schedules.

    Members:
        CONSTANT: No decay.
        COSINE: Cosine annealing to ``min_lr_factor`` of the base rate.
        STEP: Multiplicative decay every ``step_size`` epochs.
    """

    CONSTANT = "constant"
    COSINE = "cosine"
    STEP = "step"


@dataclass(frozen=True)
class SchedulerSettings:
    """Schedule hyperparameters.

    Attributes:
        name: Which schedule to build.
        warmup_steps: Number of linear warm-up steps before the main schedule begins.
        min_lr_factor: Floor for the cosine schedule, as a fraction of the base rate.
        step_size_epochs: Epoch interval between step decays.
        gamma: Multiplicative factor for the step schedule.
    """

    name: SchedulerType = SchedulerType.COSINE
    warmup_steps: int = 0
    min_lr_factor: float = 0.0
    step_size_epochs: int = 10
    gamma: float = 0.1

    def __post_init__(self) -> None:
        """Validate hyperparameter ranges.

        Raises:
            ConfigurationError: If any value is out of range.
        """
        if self.warmup_steps < 0:
            msg = f"warmup_steps must be non-negative, received {self.warmup_steps}"
            raise ConfigurationError(msg, details={"warmup_steps": self.warmup_steps})
        if not 0.0 <= self.min_lr_factor <= 1.0:
            msg = f"min_lr_factor must lie in [0, 1], received {self.min_lr_factor}"
            raise ConfigurationError(msg, details={"min_lr_factor": self.min_lr_factor})
        if self.step_size_epochs < 1:
            msg = f"step_size_epochs must be >= 1, received {self.step_size_epochs}"
            raise ConfigurationError(msg, details={"step_size_epochs": self.step_size_epochs})
        if not 0.0 < self.gamma <= 1.0:
            msg = f"gamma must lie in (0, 1], received {self.gamma}"
            raise ConfigurationError(msg, details={"gamma": self.gamma})


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Return the linear warm-up multiplier for ``step``.

    Args:
        step: Zero-based step index.
        warmup_steps: Length of the warm-up phase.

    Returns:
        A multiplier in ``(0, 1]``.
    """
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    # `step + 1` so the very first step is not multiplied by zero, which would freeze the
    # model for one iteration and make step counts off by one between schedules.
    return float(step + 1) / float(warmup_steps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: SchedulerSettings,
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> LambdaLR:
    """Construct a per-step learning-rate scheduler.

    A single :class:`~torch.optim.lr_scheduler.LambdaLR` implements every schedule, so
    warm-up composes with each of them without a chained-scheduler wrapper whose
    ``state_dict`` semantics vary between PyTorch versions.

    Args:
        optimizer: Optimiser to schedule.
        settings: Schedule hyperparameters.
        total_steps: Total optimiser steps planned for the run; must be positive.
        steps_per_epoch: Steps in one epoch, used by the step schedule.

    Returns:
        The configured scheduler, to be stepped once per optimiser step.

    Raises:
        ConfigurationError: If ``total_steps`` or ``steps_per_epoch`` is not positive, or
            the schedule is unsupported.
    """
    if total_steps < 1:
        msg = f"total_steps must be >= 1, received {total_steps}"
        raise ConfigurationError(msg, details={"total_steps": total_steps})
    if steps_per_epoch < 1:
        msg = f"steps_per_epoch must be >= 1, received {steps_per_epoch}"
        raise ConfigurationError(msg, details={"steps_per_epoch": steps_per_epoch})

    warmup = min(settings.warmup_steps, max(total_steps - 1, 0))
    decay_steps = max(total_steps - warmup, 1)

    if settings.name is SchedulerType.CONSTANT:

        def factor(step: int) -> float:
            return _warmup_factor(step, warmup)

    elif settings.name is SchedulerType.COSINE:

        def factor(step: int) -> float:
            warm = _warmup_factor(step, warmup)
            if step < warmup:
                return warm
            progress = min(1.0, (step - warmup) / decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return settings.min_lr_factor + (1.0 - settings.min_lr_factor) * cosine

    elif settings.name is SchedulerType.STEP:

        def factor(step: int) -> float:
            warm = _warmup_factor(step, warmup)
            if step < warmup:
                return warm
            epoch = (step - warmup) // steps_per_epoch
            return float(settings.gamma ** (epoch // settings.step_size_epochs))

    else:  # pragma: no cover - guarded by the closed enumeration
        msg = (  # type: ignore[unreachable]
            f"unsupported scheduler '{settings.name}'; expected one of "
            f"{[member.value for member in SchedulerType]}"
        )
        raise ConfigurationError(msg, details={"scheduler": str(settings.name)})

    return LambdaLR(optimizer, lr_lambda=factor)


__all__ = ["SchedulerSettings", "SchedulerType", "build_scheduler"]
