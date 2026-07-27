"""Optimiser construction.

Choice of optimisers
--------------------
``SGD`` with momentum is the reference for convolutional image classification: with a
tuned schedule it generalises at least as well as anything else, and its behaviour is
well understood. Its weakness for NAS is sensitivity — a learning rate tuned for one
architecture can be badly wrong for another, and the search would then be ranking
learning-rate compatibility rather than architecture quality.

``AdamW`` is the default here for exactly that reason. Per-parameter adaptive step sizes
make it far more forgiving of architectural variation, which matters when a single
training recipe must be applied unchanged to hundreds of different networks. The cost is
a modest generalisation gap on some vision tasks — a bias worth stating explicitly,
because it means the *ranking* this framework produces is a ranking under AdamW, not an
architecture-intrinsic truth. See ``docs/concepts/common-pitfalls.md``.

Decoupled weight decay
----------------------
``AdamW`` implements decoupled weight decay: the decay term is applied directly to the
weights rather than folded into the gradient. In plain Adam, L2 regularisation added to
the gradient is divided by the adaptive step size, so parameters with large gradients get
*less* regularisation — the opposite of the intent.

No decay on normalisation and bias parameters
----------------------------------------------
Weight decay pulls parameters towards zero. For a BatchNorm scale, zero means "delete this
channel", and for a bias it removes the layer's ability to shift its output. Neither is a
useful prior, and decaying them measurably hurts. Parameters are therefore split into two
groups, and the no-decay group is exempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from nas_engine.exceptions import ConfigurationError


class OptimizerType(str, Enum):
    """Supported optimisers.

    Members:
        SGD: Stochastic gradient descent with momentum.
        ADAMW: Adam with decoupled weight decay.
    """

    SGD = "sgd"
    ADAMW = "adamw"


@dataclass(frozen=True)
class OptimizerSettings:
    """Optimiser hyperparameters.

    Attributes:
        name: Which optimiser to build.
        learning_rate: Base learning rate.
        weight_decay: Decay coefficient applied to weight matrices only.
        momentum: Momentum coefficient; SGD only.
        nesterov: Whether to use Nesterov momentum; SGD only.
        beta1: First moment decay; AdamW only.
        beta2: Second moment decay; AdamW only.
        eps: Numerical stability term; AdamW only.
        decay_normalization: Whether normalisation and bias parameters receive weight
            decay. Defaults to ``False``, which is the recommended setting.
    """

    name: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    decay_normalization: bool = False

    def __post_init__(self) -> None:
        """Validate hyperparameter ranges.

        Raises:
            ConfigurationError: If any value is out of its valid range.
        """
        if self.learning_rate <= 0:
            msg = f"learning_rate must be positive, received {self.learning_rate}"
            raise ConfigurationError(msg, details={"learning_rate": self.learning_rate})
        if self.weight_decay < 0:
            msg = f"weight_decay must be non-negative, received {self.weight_decay}"
            raise ConfigurationError(msg, details={"weight_decay": self.weight_decay})
        if not 0.0 <= self.momentum < 1.0:
            msg = f"momentum must lie in [0, 1), received {self.momentum}"
            raise ConfigurationError(msg, details={"momentum": self.momentum})
        for label, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not 0.0 <= value < 1.0:
                msg = f"{label} must lie in [0, 1), received {value}"
                raise ConfigurationError(msg, details={label: value})
        if self.eps <= 0:
            msg = f"eps must be positive, received {self.eps}"
            raise ConfigurationError(msg, details={"eps": self.eps})


def split_parameter_groups(
    model: nn.Module, weight_decay: float, *, decay_normalization: bool
) -> list[dict[str, object]]:
    """Split model parameters into decayed and non-decayed groups.

    Args:
        model: Model whose parameters are being optimised.
        weight_decay: Decay coefficient for the decayed group.
        decay_normalization: Whether to decay normalisation and bias parameters too.

    Returns:
        Parameter groups suitable for a :class:`torch.optim.Optimizer`.
    """
    if decay_normalization or weight_decay == 0.0:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            }
        ]

    decayed: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            is_norm = isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
            if name.endswith("bias") or is_norm:
                plain.append(parameter)
            else:
                decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, settings: OptimizerSettings) -> torch.optim.Optimizer:
    """Construct an optimiser for ``model``.

    Args:
        model: Model to optimise.
        settings: Hyperparameters.

    Returns:
        The configured optimiser.

    Raises:
        ConfigurationError: If the optimiser type is unsupported or the model has no
            trainable parameters.
    """
    groups = split_parameter_groups(
        model, settings.weight_decay, decay_normalization=settings.decay_normalization
    )
    if not any(group["params"] for group in groups):
        msg = (
            "model has no trainable parameters; this usually means the architecture "
            "consists only of pooling and identity operations"
        )
        raise ConfigurationError(msg)

    if settings.name is OptimizerType.SGD:
        return torch.optim.SGD(
            groups,
            lr=settings.learning_rate,
            momentum=settings.momentum,
            nesterov=settings.nesterov and settings.momentum > 0,
        )
    if settings.name is OptimizerType.ADAMW:
        return torch.optim.AdamW(
            groups,
            lr=settings.learning_rate,
            betas=(settings.beta1, settings.beta2),
            eps=settings.eps,
        )
    msg = (  # type: ignore[unreachable]  # pragma: no cover - closed enumeration
        f"unsupported optimizer '{settings.name}'; expected one of "
        f"{[member.value for member in OptimizerType]}"
    )
    raise ConfigurationError(msg, details={"optimizer": str(settings.name)})


__all__ = [
    "OptimizerSettings",
    "OptimizerType",
    "build_optimizer",
    "split_parameter_groups",
]
