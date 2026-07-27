r"""Static tensor-shape inference for architecture genotypes.

Why infer shapes before building a model
----------------------------------------
Constructing a ``nn.Module`` allocates parameters; running a forward pass allocates
activations. For a search that proposes thousands of candidates, discovering that an
architecture is invalid *after* paying that cost is wasteful, and discovering it in the
middle of training is worse — the failure surfaces as an opaque
``RuntimeError: Given groups=1, weight of size ...``.

This module reproduces PyTorch's shape arithmetic in pure Python. It costs
microseconds, requires no tensors, and produces an actionable error naming the exact
stage and block that fails. The engine therefore rejects structurally invalid
candidates before they ever reach a device.

The arithmetic is verified against PyTorch itself in
``tests/integration/test_shape_inference_matches_torch.py`` — a static model of a
library's behaviour is only trustworthy if it is continuously checked against it.

Convolution output size
-----------------------
For input extent :math:`H`, kernel :math:`k`, padding :math:`p`, stride :math:`s`,
dilation 1:

.. math::

    H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1

This project always uses *same* padding, :math:`p = \lfloor k/2 \rfloor`, with odd
:math:`k`. Substituting gives :math:`H_{out} = \lceil H / s \rceil`, so a stride of 1
preserves resolution exactly and a stride of 2 halves it (rounding up).
"""

from __future__ import annotations

from dataclasses import dataclass

from nas_engine.architectures.spec import ArchitectureSpec, BlockSpec
from nas_engine.architectures.types import OperationType
from nas_engine.exceptions import ShapeInferenceError

#: Channel counts are rounded to a multiple of this value. Hardware kernels (and
#: PyTorch's own dispatch) are markedly faster on channel counts divisible by 8, and
#: rounding also collapses near-duplicate architectures into one canonical form.
CHANNEL_DIVISOR: int = 8


def make_divisible(value: float, divisor: int = CHANNEL_DIVISOR, minimum: int | None = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor``, never below ``minimum``.

    This is the standard width-multiplier rounding rule from the MobileNet family. The
    final guard prevents rounding *down* by more than 10%, which would otherwise make a
    small expansion ratio silently collapse to nothing.

    Args:
        value: Desired channel count, possibly fractional.
        divisor: Multiple to round to.
        minimum: Lower bound; defaults to ``divisor``.

    Returns:
        A positive integer multiple of ``divisor``.

    Raises:
        ValueError: If ``divisor`` is not positive.
    """
    if divisor <= 0:
        msg = f"divisor must be positive, received {divisor}"
        raise ValueError(msg)
    floor = divisor if minimum is None else minimum
    rounded = max(floor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def conv_output_size(input_size: int, kernel_size: int, stride: int) -> int:
    """Compute the output extent of a same-padded convolution or pooling window.

    Args:
        input_size: Input spatial extent.
        kernel_size: Window size (odd).
        stride: Stride.

    Returns:
        Output spatial extent, at least ``0``.

    Raises:
        ValueError: If any argument is not positive.
    """
    if input_size <= 0 or kernel_size <= 0 or stride <= 0:
        msg = (
            "conv_output_size requires positive arguments, received "
            f"input_size={input_size}, kernel_size={kernel_size}, stride={stride}"
        )
        raise ValueError(msg)
    padding = kernel_size // 2
    return (input_size + 2 * padding - kernel_size) // stride + 1


@dataclass(frozen=True)
class TensorShape:
    """A channels-height-width shape, excluding the batch dimension.

    Attributes:
        channels: Channel count.
        height: Spatial height.
        width: Spatial width.
    """

    channels: int
    height: int
    width: int

    @property
    def elements(self) -> int:
        """Number of elements in one example."""
        return self.channels * self.height * self.width

    def as_tuple(self) -> tuple[int, int, int]:
        """Return ``(channels, height, width)``."""
        return (self.channels, self.height, self.width)

    def __str__(self) -> str:
        """Render as ``CxHxW``."""
        return f"{self.channels}x{self.height}x{self.width}"


@dataclass(frozen=True)
class LayerShape:
    """Input and output shapes of one named layer.

    Attributes:
        name: Dotted path identifying the layer, e.g. ``"stages.1.blocks.0"``.
        kind: Short description of what the layer does.
        input_shape: Shape entering the layer.
        output_shape: Shape leaving the layer.
    """

    name: str
    kind: str
    input_shape: TensorShape
    output_shape: TensorShape


@dataclass(frozen=True)
class ShapeTrace:
    """The complete shape history of an architecture.

    Attributes:
        input_shape: Network input shape.
        layers: Per-layer shapes in execution order.
        features_shape: Shape entering the head's global pooling.
        pooled_features: Feature width after global pooling and flattening.
        output_features: Number of logits produced.
    """

    input_shape: TensorShape
    layers: tuple[LayerShape, ...]
    features_shape: TensorShape
    pooled_features: int
    output_features: int

    def to_rows(self) -> list[tuple[str, str, str, str]]:
        """Return ``(name, kind, input, output)`` rows for tabular display."""
        return [
            (layer.name, layer.kind, str(layer.input_shape), str(layer.output_shape))
            for layer in self.layers
        ]


def _block_output_channels(block: BlockSpec, input_channels: int, location: str) -> int:
    """Determine and validate the channel count a block produces.

    Args:
        block: Block being inspected.
        input_channels: Channels entering the block.
        location: Dotted path used in error messages.

    Returns:
        The output channel count.

    Raises:
        ShapeInferenceError: If a channel-preserving operation declares a different
            output channel count than its input.
    """
    if block.operation.can_change_channels:
        return block.out_channels
    if block.out_channels != input_channels:
        msg = (
            f"{location}: operation '{block.operation.value}' cannot change the channel "
            f"count, but out_channels={block.out_channels} while the block receives "
            f"{input_channels} channels. Set out_channels={input_channels}, or use "
            f"'{OperationType.CONV.value}' / '{OperationType.DW_SEP_CONV.value}' to "
            "change width."
        )
        raise ShapeInferenceError(
            msg,
            details={
                "location": location,
                "operation": block.operation.value,
                "declared_out_channels": block.out_channels,
                "input_channels": input_channels,
            },
        )
    return input_channels


def _validate_residual(
    block: BlockSpec, shape_in: TensorShape, shape_out: TensorShape, location: str
) -> None:
    """Verify an identity residual connection is legal.

    A residual adds the block input to its output elementwise, which requires the two
    tensors to have identical shapes. This project deliberately supports *identity*
    shortcuts only; projection shortcuts (a 1x1 convolution on the skip path) are a
    documented extension rather than an implicit fallback, because silently inserting
    parameters would make the parameter-count objective misleading.

    Args:
        block: Block being inspected.
        shape_in: Shape entering the block.
        shape_out: Shape leaving the block's main path.
        location: Dotted path used in error messages.

    Raises:
        ShapeInferenceError: If the shapes do not match.
    """
    if not block.use_residual:
        return
    if shape_in.as_tuple() == shape_out.as_tuple():
        return
    reasons: list[str] = []
    if shape_in.channels != shape_out.channels:
        reasons.append(f"channel counts differ ({shape_in.channels} -> {shape_out.channels})")
    if (shape_in.height, shape_in.width) != (shape_out.height, shape_out.width):
        reasons.append(
            f"spatial sizes differ ({shape_in.height}x{shape_in.width} -> "
            f"{shape_out.height}x{shape_out.width}); stride={block.stride}"
        )
    msg = (
        f"{location}: use_residual=True requires the block input and output to have "
        f"identical shapes, but {' and '.join(reasons)}. Set stride=1 and "
        f"out_channels={shape_in.channels}, or set use_residual=False."
    )
    raise ShapeInferenceError(
        msg,
        details={
            "location": location,
            "input_shape": shape_in.as_tuple(),
            "output_shape": shape_out.as_tuple(),
            "stride": block.stride,
        },
    )


def infer_shapes(spec: ArchitectureSpec) -> ShapeTrace:
    """Compute the full shape trace of an architecture, validating it along the way.

    Args:
        spec: Architecture to analyse.

    Returns:
        A :class:`ShapeTrace` describing every intermediate shape.

    Raises:
        ShapeInferenceError: If the architecture is structurally invalid — an
            impossible downsampling sequence, a channel mismatch, or an illegal
            residual connection.
    """
    layers: list[LayerShape] = []
    current = TensorShape(spec.input_channels, spec.input_size, spec.input_size)
    input_shape = current

    stem_size = conv_output_size(current.height, spec.stem.kernel_size, spec.stem.stride)
    if stem_size < 1:
        msg = (
            f"stem with kernel_size={spec.stem.kernel_size} and stride={spec.stem.stride} "
            f"reduces the {current.height}x{current.width} input below 1x1"
        )
        raise ShapeInferenceError(msg, details={"input_size": spec.input_size})
    stem_shape = TensorShape(spec.stem.out_channels, stem_size, stem_size)
    layers.append(LayerShape("stem", "conv", current, stem_shape))
    current = stem_shape

    for stage_index, stage in enumerate(spec.stages):
        for block_index, block in enumerate(stage.blocks):
            location = f"stages.{stage_index}.blocks.{block_index}"
            out_channels = _block_output_channels(block, current.channels, location)
            out_size = conv_output_size(current.height, block.kernel_size, block.stride)
            if out_size < 1:
                msg = (
                    f"{location}: operation '{block.operation.value}' with "
                    f"kernel_size={block.kernel_size} and stride={block.stride} reduces "
                    f"the {current.height}x{current.width} feature map below 1x1. "
                    "Reduce the number of stride-2 blocks or increase input_size "
                    f"(currently {spec.input_size})."
                )
                raise ShapeInferenceError(
                    msg,
                    details={
                        "location": location,
                        "input_size": current.height,
                        "stride": block.stride,
                        "kernel_size": block.kernel_size,
                    },
                )
            block_out = TensorShape(out_channels, out_size, out_size)
            _validate_residual(block, current, block_out, location)
            layers.append(LayerShape(location, block.operation.value, current, block_out))
            current = block_out

    features_shape = current
    hidden = spec.head.hidden_units
    pooled = features_shape.channels
    pooled_shape = TensorShape(pooled, 1, 1)
    layers.append(
        LayerShape("head.pool", f"global_{spec.head.pooling.value}", features_shape, pooled_shape)
    )
    if hidden > 0:
        layers.append(LayerShape("head.hidden", "linear", pooled_shape, TensorShape(hidden, 1, 1)))
    layers.append(
        LayerShape(
            "head.classifier",
            "linear",
            TensorShape(hidden or pooled, 1, 1),
            TensorShape(spec.num_classes, 1, 1),
        )
    )

    return ShapeTrace(
        input_shape=input_shape,
        layers=tuple(layers),
        features_shape=features_shape,
        pooled_features=pooled,
        output_features=spec.num_classes,
    )


__all__ = [
    "CHANNEL_DIVISOR",
    "LayerShape",
    "ShapeTrace",
    "TensorShape",
    "conv_output_size",
    "infer_shapes",
    "make_divisible",
]
