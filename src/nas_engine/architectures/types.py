"""Enumerations shared by search spaces, architecture specs, and model builders.

These are the *atoms* of the genotype. Every one is a closed enumeration rather than a
free string, for three reasons:

1. Pydantic rejects unknown members at parse time, so an imported architecture JSON
   containing ``"operation": "__import__('os').system"`` fails validation instead of
   ever reaching a lookup table.
2. The canonical serialisation is stable: the wire value is the enum's value, which is
   fixed by this file, not by the order of a registry dictionary.
3. Exhaustiveness can be checked by tests — a new operation that the model builder does
   not handle is caught immediately.

Adding a member is a search-space change and therefore a **breaking change for
architecture hashes**: see ``docs/guides/adding-an-operation.md``.
"""

from __future__ import annotations

from enum import Enum


class OperationType(str, Enum):
    """The primitive operations a block may perform.

    Members:
        CONV: Standard dense convolution. Every output channel sees every input
            channel, giving maximum expressiveness at ``k*k*Cin*Cout`` parameters.
        DW_SEP_CONV: Depthwise-separable convolution, optionally with an inverted
            bottleneck expansion. Factorises the dense convolution into a spatial
            (depthwise) part and a cross-channel (pointwise) part, reducing cost from
            ``k*k*Cin*Cout`` to roughly ``k*k*Cin + Cin*Cout``.
        IDENTITY: Pass-through. Lets the search shorten the effective depth of a stage
            without changing the genotype length, which keeps mutation simple.
        MAX_POOL: Spatial max pooling. Selects the strongest activation in each window.
        AVG_POOL: Spatial average pooling. Smooths rather than selects.
    """

    CONV = "conv"
    DW_SEP_CONV = "dw_sep_conv"
    IDENTITY = "identity"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"

    @property
    def is_parametric(self) -> bool:
        """Whether the operation owns trainable weights."""
        return self in {OperationType.CONV, OperationType.DW_SEP_CONV}

    @property
    def can_change_channels(self) -> bool:
        """Whether the operation may produce a different channel count than it consumes.

        Pooling and identity are channel-preserving by construction, so a block using
        them must declare ``out_channels`` equal to its input channels. This is
        enforced during graph validation, where the input channel count is known.
        """
        return self.is_parametric

    @property
    def uses_kernel_size(self) -> bool:
        """Whether ``kernel_size`` is a meaningful (active) choice for this operation."""
        return self is not OperationType.IDENTITY

    @property
    def uses_expansion_ratio(self) -> bool:
        """Whether ``expansion_ratio`` is a meaningful choice for this operation."""
        return self is OperationType.DW_SEP_CONV


class NormalizationType(str, Enum):
    """Normalisation applied after a parametric operation.

    Members:
        BATCH: :class:`torch.nn.BatchNorm2d`. Normalises per channel over the batch and
            spatial dimensions. Strong regulariser and optimisation aid, but its
            statistics depend on batch size, which makes very small batches unstable.
        GROUP: :class:`torch.nn.GroupNorm`. Normalises within channel groups of a single
            example, so it is batch-size independent — useful for the tiny batches
            common in low-fidelity NAS evaluation.
        NONE: No normalisation. The preceding convolution then carries a bias term.
    """

    BATCH = "batch"
    GROUP = "group"
    NONE = "none"


class ActivationType(str, Enum):
    """Pointwise nonlinearity.

    Members:
        RELU: ``max(0, x)``. Cheap, sparse, the historical default.
        RELU6: ``min(max(0, x), 6)``. Bounded output improves low-precision robustness;
            standard in mobile architectures.
        SILU: ``x * sigmoid(x)``. Smooth and non-monotonic; often improves accuracy at
            slightly higher cost.
        GELU: Gaussian error linear unit; smooth, widely used in modern networks.
        HARDSWISH: Piecewise-linear approximation of SiLU designed for cheap inference.
        IDENTITY: No nonlinearity. Used as the canonical value for operations that have
            no activation, such as pooling.
    """

    RELU = "relu"
    RELU6 = "relu6"
    SILU = "silu"
    GELU = "gelu"
    HARDSWISH = "hardswish"
    IDENTITY = "identity"


class PoolingType(str, Enum):
    """Global pooling used by the classifier head.

    Members:
        AVG: Global average pooling — the standard choice; averages spatial evidence.
        MAX: Global max pooling — keeps only the strongest spatial response.
    """

    AVG = "avg"
    MAX = "max"


__all__ = ["ActivationType", "NormalizationType", "OperationType", "PoolingType"]
