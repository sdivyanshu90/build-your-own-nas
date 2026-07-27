"""Analytic cost model: parameters, buffers, and multiply-accumulate operations.

Why an analytic model rather than building the module and counting?
-------------------------------------------------------------------
Constraint checking happens on *every proposed candidate*, including ones that will be
rejected. Instantiating a ``nn.Module`` allocates and initialises every weight tensor,
which for a 5-million-parameter network costs tens of milliseconds and 20 MB. The
analytic model costs microseconds and no memory, so a parameter-count constraint can be
enforced before anything is allocated.

The obvious risk is **drift**: if the builder changes and the cost model does not, the
two disagree silently and the parameter objective becomes a lie. That risk is managed
by an exactness test — ``tests/property/test_cost_model_properties.py`` asserts that the
analytic count equals :func:`torch.nn.Module.parameters` counting for every sampled
architecture. The cost model is therefore not an estimate of the parameter count; it is
a second, independently derived computation of the same quantity.

Multiply-accumulates (MACs) *are* an estimate. They count convolution and linear
arithmetic only, ignoring normalisation, activation, and pooling, which is the usual
convention in the literature. MACs correlate with, but do not determine, latency —
memory bandwidth, kernel selection, and parallelism matter at least as much. See
``docs/concepts/training-and-evaluation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from nas_engine.architectures.shapes import ShapeTrace, infer_shapes, make_divisible
from nas_engine.architectures.spec import ArchitectureSpec, BlockSpec
from nas_engine.architectures.types import NormalizationType, OperationType

#: Bytes per parameter when weights are stored as ``float32``.
BYTES_PER_FLOAT32: int = 4


def normalization_parameters(norm: NormalizationType, channels: int) -> tuple[int, int]:
    """Return ``(trainable, non_trainable)`` parameter counts for a normalisation layer.

    ``BatchNorm2d`` owns an affine weight and bias (both trainable, ``C`` each) plus
    three buffers: ``running_mean`` (``C``), ``running_var`` (``C``), and the scalar
    ``num_batches_tracked``. Buffers are saved in the state dict and therefore count
    towards serialised model size, but they are not optimised.

    ``GroupNorm`` normalises within a single example, so it needs no running statistics
    and owns only the affine parameters.

    Args:
        norm: Normalisation type.
        channels: Channel count the layer operates on.

    Returns:
        A ``(trainable, non_trainable)`` tuple.
    """
    if norm is NormalizationType.BATCH:
        return (2 * channels, 2 * channels + 1)
    if norm is NormalizationType.GROUP:
        return (2 * channels, 0)
    return (0, 0)


def conv_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    groups: int = 1,
    bias: bool = False,
) -> int:
    """Return the trainable parameter count of a 2-D convolution.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Square kernel extent.
        groups: Grouped-convolution group count; ``in_channels`` for depthwise.
        bias: Whether a bias vector is present.

    Returns:
        Parameter count.
    """
    weights = out_channels * (in_channels // groups) * kernel_size * kernel_size
    return weights + (out_channels if bias else 0)


def conv_macs(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_height: int,
    output_width: int,
    *,
    groups: int = 1,
) -> int:
    """Return the multiply-accumulate count of a 2-D convolution.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Square kernel extent.
        output_height: Output spatial height.
        output_width: Output spatial width.
        groups: Grouped-convolution group count.

    Returns:
        MAC count for a single example.
    """
    per_output = (in_channels // groups) * kernel_size * kernel_size
    return per_output * out_channels * output_height * output_width


def separable_hidden_channels(block: BlockSpec, in_channels: int) -> int:
    """Return the inverted-bottleneck width of a depthwise-separable block.

    An expansion ratio of exactly 1 means "no expansion": the depthwise convolution
    runs directly on the input width and no expanding pointwise convolution is built.
    Any other ratio produces a hidden width rounded to a multiple of
    :data:`~nas_engine.architectures.shapes.CHANNEL_DIVISOR`.

    Args:
        block: The block; must use a depthwise-separable operation.
        in_channels: Channels entering the block.

    Returns:
        Hidden channel count used by the depthwise convolution.
    """
    if block.expansion_ratio == 1.0:
        return in_channels
    return make_divisible(in_channels * block.expansion_ratio)


@dataclass(frozen=True)
class ArchitectureCost:
    """Static cost summary of an architecture.

    Attributes:
        trainable_parameters: Parameters updated by the optimiser.
        non_trainable_parameters: Buffers persisted in the state dict, e.g. BatchNorm
            running statistics.
        multiply_accumulates: Estimated MACs for one forward pass of one example.
        parameter_bytes: Serialised size of parameters and buffers as ``float32``.
        depth: Number of blocks, excluding stem and head.
    """

    trainable_parameters: int
    non_trainable_parameters: int
    multiply_accumulates: int
    parameter_bytes: int
    depth: int

    @property
    def total_parameters(self) -> int:
        """Trainable plus non-trainable parameter count."""
        return self.trainable_parameters + self.non_trainable_parameters

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable representation."""
        return {
            "trainable_parameters": self.trainable_parameters,
            "non_trainable_parameters": self.non_trainable_parameters,
            "total_parameters": self.total_parameters,
            "multiply_accumulates": self.multiply_accumulates,
            "parameter_bytes": self.parameter_bytes,
            "depth": self.depth,
        }


def _block_cost(
    block: BlockSpec,
    in_channels: int,
    out_channels: int,
    output_height: int,
    output_width: int,
    input_height: int,
    input_width: int,
) -> tuple[int, int, int]:
    """Return ``(trainable, non_trainable, macs)`` for one block.

    The structures modelled here mirror :mod:`nas_engine.models.blocks` exactly.

    Args:
        block: Block specification.
        in_channels: Channels entering the block.
        out_channels: Channels leaving the block.
        output_height: Output spatial height.
        output_width: Output spatial width.
        input_height: Input spatial height (used for pre-stride pointwise layers).
        input_width: Input spatial width.

    Returns:
        Parameter and MAC counts for the block.
    """
    if not block.operation.is_parametric:
        return (0, 0, 0)

    norm = block.normalization
    bias = norm is NormalizationType.NONE
    trainable = 0
    non_trainable = 0
    macs = 0

    if block.operation is OperationType.CONV:
        trainable += conv_parameters(in_channels, out_channels, block.kernel_size, bias=bias)
        macs += conv_macs(in_channels, out_channels, block.kernel_size, output_height, output_width)
        norm_trainable, norm_buffers = normalization_parameters(norm, out_channels)
        trainable += norm_trainable
        non_trainable += norm_buffers
        return (trainable, non_trainable, macs)

    # Depthwise-separable, optionally with an inverted-bottleneck expansion.
    hidden = separable_hidden_channels(block, in_channels)

    if hidden != in_channels:
        # 1x1 expansion runs at the *input* resolution, before the strided depthwise.
        trainable += conv_parameters(in_channels, hidden, 1, bias=bias)
        macs += conv_macs(in_channels, hidden, 1, input_height, input_width)
        norm_trainable, norm_buffers = normalization_parameters(norm, hidden)
        trainable += norm_trainable
        non_trainable += norm_buffers

    trainable += conv_parameters(hidden, hidden, block.kernel_size, groups=hidden, bias=bias)
    macs += conv_macs(hidden, hidden, block.kernel_size, output_height, output_width, groups=hidden)
    norm_trainable, norm_buffers = normalization_parameters(norm, hidden)
    trainable += norm_trainable
    non_trainable += norm_buffers

    trainable += conv_parameters(hidden, out_channels, 1, bias=bias)
    macs += conv_macs(hidden, out_channels, 1, output_height, output_width)
    norm_trainable, norm_buffers = normalization_parameters(norm, out_channels)
    trainable += norm_trainable
    non_trainable += norm_buffers

    return (trainable, non_trainable, macs)


def compute_cost(spec: ArchitectureSpec, trace: ShapeTrace | None = None) -> ArchitectureCost:
    """Compute the analytic cost of an architecture.

    Args:
        spec: Architecture to cost.
        trace: Pre-computed shape trace; inferred when omitted.

    Returns:
        An :class:`ArchitectureCost`.

    Raises:
        ShapeInferenceError: If the architecture is structurally invalid.
    """
    shape_trace = trace if trace is not None else infer_shapes(spec)

    # Shape trace layer 0 is always the stem.
    stem_layer = shape_trace.layers[0]
    bias = spec.stem.normalization is NormalizationType.NONE
    trainable = conv_parameters(
        spec.input_channels, spec.stem.out_channels, spec.stem.kernel_size, bias=bias
    )
    macs = conv_macs(
        spec.input_channels,
        spec.stem.out_channels,
        spec.stem.kernel_size,
        stem_layer.output_shape.height,
        stem_layer.output_shape.width,
    )
    norm_trainable, non_trainable = normalization_parameters(
        spec.stem.normalization, spec.stem.out_channels
    )
    trainable += norm_trainable

    block_layers = shape_trace.layers[1 : 1 + spec.total_blocks]
    for (_, _, block), layer in zip(spec.iter_blocks(), block_layers, strict=True):
        block_trainable, block_buffers, block_macs = _block_cost(
            block,
            layer.input_shape.channels,
            layer.output_shape.channels,
            layer.output_shape.height,
            layer.output_shape.width,
            layer.input_shape.height,
            layer.input_shape.width,
        )
        trainable += block_trainable
        non_trainable += block_buffers
        macs += block_macs

    pooled = shape_trace.pooled_features
    if spec.head.hidden_units > 0:
        trainable += pooled * spec.head.hidden_units + spec.head.hidden_units
        macs += pooled * spec.head.hidden_units
        classifier_in = spec.head.hidden_units
    else:
        classifier_in = pooled
    trainable += classifier_in * spec.num_classes + spec.num_classes
    macs += classifier_in * spec.num_classes

    return ArchitectureCost(
        trainable_parameters=trainable,
        non_trainable_parameters=non_trainable,
        multiply_accumulates=macs,
        parameter_bytes=(trainable + non_trainable) * BYTES_PER_FLOAT32,
        depth=spec.total_blocks,
    )


__all__ = [
    "BYTES_PER_FLOAT32",
    "ArchitectureCost",
    "compute_cost",
    "conv_macs",
    "conv_parameters",
    "normalization_parameters",
    "separable_hidden_channels",
]
