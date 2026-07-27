r"""Block modules: the phenotype of one :class:`~nas_engine.architectures.spec.BlockSpec`.

The block is where the genotype becomes computation. Each block owns one operation and,
optionally, an identity shortcut around it.

Depthwise-separable convolution
-------------------------------
A dense :math:`k \times k` convolution from :math:`C_{in}` to :math:`C_{out}` channels
costs :math:`k^2 C_{in} C_{out}` parameters. It performs two jobs at once: mixing
spatially (within a :math:`k \times k` neighbourhood) and mixing across channels.
Factorising those jobs gives

.. math::

    \underbrace{k^2 C_{in}}_{\text{depthwise}} + \underbrace{C_{in} C_{out}}_{\text{pointwise}}

parameters — for :math:`k=3, C_{in}=C_{out}=64` that is 4 672 instead of 36 864, an
8-fold reduction, at some cost in representational power.

Inverted bottleneck
-------------------
When ``expansion_ratio > 1`` the block first *widens* with a 1x1 convolution, applies the
depthwise convolution in the wider space, then projects back down. The projection has a
normalisation layer but **no activation** — the "linear bottleneck" of MobileNetV2. The
argument: a ReLU zeroes roughly half its inputs, and in a low-dimensional space that
destroys information that cannot be recovered. In a high-dimensional space the same
information survives in the surviving coordinates. So nonlinearity goes where the
representation is wide, and the narrow projection stays linear.

Residual connections
--------------------
``out = x + f(x)`` gives gradients a path that skips :math:`f` entirely, which is what
makes deep stacks trainable: the gradient of the sum is the gradient of the identity plus
the gradient through :math:`f`, so it cannot vanish through depth alone. Only *identity*
shortcuts are supported here; see
:func:`nas_engine.architectures.shapes._validate_residual` for why projections are
excluded.
"""

from __future__ import annotations

import torch
from torch import nn

from nas_engine.architectures.cost import separable_hidden_channels
from nas_engine.architectures.spec import BlockSpec
from nas_engine.architectures.types import ActivationType, NormalizationType, OperationType
from nas_engine.exceptions import ModelBuildError
from nas_engine.models.operations import build_activation, build_conv_bn_act, build_normalization


class SeparableConvBlock(nn.Module):
    """Depthwise-separable convolution with an optional inverted bottleneck.

    Attributes:
        expand: 1x1 widening convolution, or ``None`` when ``expansion_ratio == 1``.
        depthwise: Spatial convolution applied per channel.
        project: 1x1 convolution back to the output width, with no activation.
    """

    def __init__(self, block: BlockSpec, in_channels: int) -> None:
        """Build the block.

        Args:
            block: Specification; must use the depthwise-separable operation.
            in_channels: Channels entering the block.

        Raises:
            ModelBuildError: If ``block`` does not describe a separable convolution.
        """
        super().__init__()
        if block.operation is not OperationType.DW_SEP_CONV:
            msg = (
                "SeparableConvBlock requires operation "
                f"'{OperationType.DW_SEP_CONV.value}', received '{block.operation.value}'"
            )
            raise ModelBuildError(msg, details={"operation": block.operation.value})

        hidden = separable_hidden_channels(block, in_channels)
        bias = block.normalization is NormalizationType.NONE

        self.expand: nn.Module | None = None
        if hidden != in_channels:
            self.expand = build_conv_bn_act(
                in_channels, hidden, 1, 1, block.normalization, block.activation
            )
        self.depthwise = build_conv_bn_act(
            hidden,
            hidden,
            block.kernel_size,
            block.stride,
            block.normalization,
            block.activation,
            groups=hidden,
        )
        # Linear bottleneck: normalisation but no activation on the projection.
        self.project = nn.Sequential(
            nn.Conv2d(hidden, block.out_channels, kernel_size=1, bias=bias),
            build_normalization(block.normalization, block.out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply expand → depthwise → project.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor.
        """
        hidden = inputs if self.expand is None else self.expand(inputs)
        spatial: torch.Tensor = self.depthwise(hidden)
        projected: torch.Tensor = self.project(spatial)
        return projected


def build_operation(block: BlockSpec, in_channels: int) -> nn.Module:
    """Build the main-path module for a block.

    Args:
        block: Block specification.
        in_channels: Channels entering the block.

    Returns:
        The operation module.

    Raises:
        ModelBuildError: If the operation has no implementation.
    """
    operation = block.operation
    if operation is OperationType.CONV:
        return build_conv_bn_act(
            in_channels,
            block.out_channels,
            block.kernel_size,
            block.stride,
            block.normalization,
            block.activation,
        )
    if operation is OperationType.DW_SEP_CONV:
        return SeparableConvBlock(block, in_channels)
    if operation is OperationType.IDENTITY:
        return nn.Identity()
    if operation is OperationType.MAX_POOL:
        return nn.MaxPool2d(
            kernel_size=block.kernel_size,
            stride=block.stride,
            padding=block.kernel_size // 2,
        )
    if operation is OperationType.AVG_POOL:
        # `count_include_pad=False` keeps border averages from being biased towards zero
        # by the padded region, which matters most at the small feature-map sizes that
        # appear late in these networks.
        return nn.AvgPool2d(
            kernel_size=block.kernel_size,
            stride=block.stride,
            padding=block.kernel_size // 2,
            count_include_pad=False,
        )
    msg = (  # type: ignore[unreachable]  # pragma: no cover - closed enumeration
        f"operation '{operation}' has no implementation in build_operation"
    )
    raise ModelBuildError(msg, details={"operation": str(operation)})


class NasBlock(nn.Module):
    """One genotype block, with its optional identity shortcut.

    Attributes:
        operation: The main-path module.
        use_residual: Whether the input is added to the output.
        spec: The specification this block was built from, retained for debugging.
    """

    def __init__(self, block: BlockSpec, in_channels: int) -> None:
        """Build the block.

        Args:
            block: Block specification.
            in_channels: Channels entering the block.

        Raises:
            ModelBuildError: If a residual is requested where shapes cannot match. Shape
                validation normally happens earlier, in
                :func:`nas_engine.architectures.shapes.infer_shapes`; this check is a
                last line of defence for hand-constructed blocks.
        """
        super().__init__()
        if block.use_residual and (block.stride != 1 or block.out_channels != in_channels):
            msg = (
                "residual connection requires stride=1 and matching channel counts, "
                f"received stride={block.stride}, in_channels={in_channels}, "
                f"out_channels={block.out_channels}"
            )
            raise ModelBuildError(
                msg,
                details={
                    "stride": block.stride,
                    "in_channels": in_channels,
                    "out_channels": block.out_channels,
                },
            )
        self.spec = block
        self.operation = build_operation(block, in_channels)
        self.use_residual = block.use_residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the operation, adding the input back when a residual is configured.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor.
        """
        output: torch.Tensor = self.operation(inputs)
        if self.use_residual:
            return output + inputs
        return output

    def extra_repr(self) -> str:
        """Return the genotype description so ``print(model)`` is informative."""
        return self.spec.describe()


class ClassifierHead(nn.Module):
    """Global pooling, optional hidden layer, dropout, and the final linear classifier.

    Dropout sits immediately before the classifier because that is where overfitting
    concentrates: the classifier has the most parameters per unit of signal and sees the
    most compressed representation. Placing dropout earlier would also perturb the
    normalisation statistics of convolutional layers, which interacts badly with
    BatchNorm.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        pooling_module: nn.Module,
        hidden_units: int,
        dropout: float,
        activation: ActivationType,
    ) -> None:
        """Build the head.

        Args:
            in_features: Channels entering the head.
            num_classes: Number of logits to produce.
            pooling_module: Pooling-and-flatten module.
            hidden_units: Hidden width, or ``0`` for none.
            dropout: Dropout probability applied before the classifier.
            activation: Nonlinearity after the hidden layer.
        """
        super().__init__()
        self.pool = pooling_module
        layers: list[nn.Module] = []
        classifier_in = in_features
        if hidden_units > 0:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(build_activation(activation))
            classifier_in = hidden_units
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(classifier_in, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool, flatten, and classify.

        Args:
            inputs: Feature map of shape ``(batch, channels, height, width)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        pooled: torch.Tensor = self.pool(inputs)
        logits: torch.Tensor = self.classifier(pooled)
        return logits


__all__ = ["ClassifierHead", "NasBlock", "SeparableConvBlock", "build_operation"]
