"""Primitive PyTorch layers used to realise a genotype.

Every function here is a *factory*: it maps a value from the closed enumerations in
:mod:`nas_engine.architectures.types` onto a concrete :class:`torch.nn.Module`. Keeping
the mapping in one place means adding an operation requires touching exactly two files
— the enum and this module — and it keeps the exhaustiveness check honest: an
unhandled enum member raises immediately rather than silently building the wrong layer.

Conventions that the analytic cost model in :mod:`nas_engine.architectures.cost`
depends on, and which must therefore never change silently:

* A convolution followed by a normalisation layer has **no bias**. The normalisation's
  own shift parameter makes the convolution's bias redundant — it would be immediately
  cancelled by the mean subtraction — so including it wastes parameters and slows
  convergence slightly.
* A convolution with ``NormalizationType.NONE`` **does** carry a bias, because otherwise
  the layer could not represent an affine shift at all.
* Padding is always ``kernel_size // 2`` ("same" padding for odd kernels).
"""

from __future__ import annotations

import torch
from torch import nn

from nas_engine.architectures.types import ActivationType, NormalizationType, PoolingType
from nas_engine.exceptions import ModelBuildError

#: Preferred number of groups for :class:`torch.nn.GroupNorm`. The actual count is the
#: largest divisor of the channel count that does not exceed this, so the layer is
#: always constructible regardless of width.
PREFERRED_GROUP_COUNT: int = 8


def group_count(channels: int, preferred: int = PREFERRED_GROUP_COUNT) -> int:
    """Return a valid ``num_groups`` for :class:`torch.nn.GroupNorm`.

    ``GroupNorm`` requires ``channels % num_groups == 0``. Widths in this project are
    multiples of 8 by construction, but a user-supplied space may use any width, so the
    group count is derived rather than assumed.

    Args:
        channels: Channel count to normalise.
        preferred: Upper bound on the number of groups.

    Returns:
        The largest divisor of ``channels`` that is at most ``preferred``; always ``>= 1``.

    Raises:
        ValueError: If ``channels`` is not positive.
    """
    if channels <= 0:
        msg = f"channels must be positive, received {channels}"
        raise ValueError(msg)
    for candidate in range(min(preferred, channels), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1  # pragma: no cover - unreachable: 1 always divides a positive integer


def build_activation(activation: ActivationType) -> nn.Module:
    """Return the module implementing an activation.

    Args:
        activation: Activation to build.

    Returns:
        A stateless activation module. ``inplace=True`` is deliberately *not* used:
        in-place activations break residual branches that need the pre-activation
        tensor, and the memory saving is irrelevant at the model sizes searched here.

    Raises:
        ModelBuildError: If the activation is not handled.
    """
    mapping: dict[ActivationType, type[nn.Module]] = {
        ActivationType.RELU: nn.ReLU,
        ActivationType.RELU6: nn.ReLU6,
        ActivationType.SILU: nn.SiLU,
        ActivationType.GELU: nn.GELU,
        ActivationType.HARDSWISH: nn.Hardswish,
        ActivationType.IDENTITY: nn.Identity,
    }
    factory = mapping.get(activation)
    if factory is None:  # pragma: no cover - guarded by the closed enumeration
        msg = f"activation '{activation}' has no implementation in build_activation"
        raise ModelBuildError(msg, details={"activation": str(activation)})
    return factory()


def build_normalization(normalization: NormalizationType, channels: int) -> nn.Module:
    """Return the module implementing a normalisation layer.

    Args:
        normalization: Normalisation to build.
        channels: Channel count the layer operates on.

    Returns:
        The normalisation module, or :class:`torch.nn.Identity` for ``NONE``.

    Raises:
        ModelBuildError: If the normalisation is not handled.
        ValueError: If ``channels`` is not positive.
    """
    if normalization is NormalizationType.BATCH:
        return nn.BatchNorm2d(channels)
    if normalization is NormalizationType.GROUP:
        return nn.GroupNorm(group_count(channels), channels)
    if normalization is NormalizationType.NONE:
        return nn.Identity()
    # mypy proves this unreachable from the closed enumeration, which is the point: adding
    # a member without handling it here becomes a type error. The runtime guard remains for
    # values that bypass the type system, such as a hand-edited stored architecture.
    msg = (  # type: ignore[unreachable]  # pragma: no cover
        f"normalization '{normalization}' has no implementation in build_normalization"
    )
    raise ModelBuildError(msg, details={"normalization": str(normalization)})


def build_global_pool(pooling: PoolingType) -> nn.Module:
    """Return the global pooling module for a classifier head.

    Args:
        pooling: Pooling type.

    Returns:
        An adaptive pooling module collapsing spatial dimensions to 1x1.

    Raises:
        ModelBuildError: If the pooling type is not handled.
    """
    if pooling is PoolingType.AVG:
        return nn.AdaptiveAvgPool2d(1)
    if pooling is PoolingType.MAX:
        return nn.AdaptiveMaxPool2d(1)
    msg = (  # type: ignore[unreachable]  # pragma: no cover - closed enumeration
        f"pooling '{pooling}' has no implementation in build_global_pool"
    )
    raise ModelBuildError(msg, details={"pooling": str(pooling)})


def build_conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    normalization: NormalizationType,
    activation: ActivationType,
    *,
    groups: int = 1,
) -> nn.Sequential:
    """Build the canonical convolution → normalisation → activation triple.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Square kernel extent (odd).
        stride: Stride.
        normalization: Normalisation to apply.
        activation: Nonlinearity to apply.
        groups: Convolution groups; equal to ``in_channels`` for a depthwise layer.

    Returns:
        A :class:`torch.nn.Sequential` containing the three layers.
    """
    bias = normalization is NormalizationType.NONE
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        ),
        build_normalization(normalization, out_channels),
        build_activation(activation),
    )


class GlobalPoolFlatten(nn.Module):
    """Global pooling followed by a flatten, as a single named module.

    Keeping the pair together makes the head's module tree easier to read in a printed
    summary and guarantees the two are never separated by a later refactor.
    """

    def __init__(self, pooling: PoolingType) -> None:
        """Initialise the module.

        Args:
            pooling: Global pooling type.
        """
        super().__init__()
        self.pool = build_global_pool(pooling)
        self.flatten = nn.Flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool spatially to 1x1 and flatten to ``(batch, channels)``.

        Args:
            inputs: Tensor of shape ``(batch, channels, height, width)``.

        Returns:
            Tensor of shape ``(batch, channels)``.
        """
        pooled: torch.Tensor = self.pool(inputs)
        flattened: torch.Tensor = self.flatten(pooled)
        return flattened


__all__ = [
    "PREFERRED_GROUP_COUNT",
    "GlobalPoolFlatten",
    "build_activation",
    "build_conv_bn_act",
    "build_global_pool",
    "build_normalization",
    "group_count",
]
