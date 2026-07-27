"""Search-space definition: the set of architectures a search is allowed to consider.

A search space is the single most consequential design choice in NAS. The literature is
consistent on this point: a well-designed space with random search frequently matches a
poorly designed space with a sophisticated algorithm. The space encodes the prior
knowledge — "convolutional networks are organised as stages", "width grows as
resolution shrinks" — that lets a small budget find good architectures.

Structure
---------
The space is *factorised*: independent choice sets for the stem, for each block, and
for the head, plus macro choices for stage count, depth, width, and stride. Factorising
keeps the encoding short and makes mutation local — changing one kernel size touches one
integer.

Conditional choices
-------------------
Some choices only apply given other choices: ``expansion_ratios`` matters only when a
depthwise-separable operation is selected, and a channel-preserving operation cannot
appear where the width changes. The space declares the full menu; the sampler
(:mod:`nas_engine.search_space.sampler`) narrows it per position, and canonicalisation
in :mod:`nas_engine.architectures.spec` erases inactive values so they cannot affect the
architecture hash.

Constraints
-----------
Constraints are *hard* filters applied after a candidate is drawn — a parameter ceiling,
a MAC ceiling, a minimum final resolution. They are checked analytically, before any
tensor is allocated. Constraints shape the space by rejection rather than by
construction, which keeps sampling simple at the cost of some wasted draws; the sampler
reports its rejection rate so a badly configured constraint is visible rather than
silent.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)
from nas_engine.exceptions import SearchSpaceError

#: Version of the search-space schema. Architectures sampled from different schema
#: versions are not comparable, so this is persisted with every search run.
SEARCH_SPACE_SCHEMA_VERSION: int = 1


class _SpaceModel(BaseModel):
    """Base for search-space models: frozen and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _unique_ordered(values: tuple[Any, ...], field_name: str) -> tuple[Any, ...]:
    """Return ``values`` with duplicates removed, preserving order.

    Duplicates in a choice set silently bias sampling — an option listed twice is
    drawn twice as often — so they are removed rather than tolerated.

    Args:
        values: Raw choice tuple.
        field_name: Field name for error messages.

    Returns:
        The de-duplicated tuple.

    Raises:
        ValueError: If the tuple is empty.
    """
    if not values:
        msg = f"{field_name} must contain at least one choice"
        raise ValueError(msg)
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


class BlockChoices(_SpaceModel):
    """Choices available to an individual block.

    Attributes:
        operations: Permitted primitive operations.
        kernel_sizes: Permitted kernel sizes (odd values only).
        expansion_ratios: Permitted inverted-bottleneck ratios; used only by
            depthwise-separable convolutions.
        normalizations: Permitted normalisation layers for parametric operations.
        activations: Permitted nonlinearities for parametric operations.
        allow_residual: Whether identity shortcuts may be sampled where they are legal.
    """

    operations: tuple[OperationType, ...] = (
        OperationType.CONV,
        OperationType.DW_SEP_CONV,
        OperationType.IDENTITY,
        OperationType.MAX_POOL,
        OperationType.AVG_POOL,
    )
    kernel_sizes: tuple[int, ...] = (3, 5)
    expansion_ratios: tuple[float, ...] = (1.0, 2.0, 4.0)
    normalizations: tuple[NormalizationType, ...] = (NormalizationType.BATCH,)
    activations: tuple[ActivationType, ...] = (ActivationType.RELU, ActivationType.SILU)
    allow_residual: bool = True

    @model_validator(mode="after")
    def _validate(self) -> BlockChoices:
        """De-duplicate choice sets and reject even kernel sizes.

        Returns:
            ``self``.

        Raises:
            ValueError: If a kernel size is even or a choice set is empty.
        """
        object.__setattr__(self, "operations", _unique_ordered(self.operations, "operations"))
        object.__setattr__(self, "kernel_sizes", _unique_ordered(self.kernel_sizes, "kernel_sizes"))
        object.__setattr__(
            self, "expansion_ratios", _unique_ordered(self.expansion_ratios, "expansion_ratios")
        )
        object.__setattr__(
            self, "normalizations", _unique_ordered(self.normalizations, "normalizations")
        )
        object.__setattr__(self, "activations", _unique_ordered(self.activations, "activations"))

        even = [size for size in self.kernel_sizes if size % 2 == 0]
        if even:
            msg = (
                f"kernel_sizes must be odd so that same-padding is exact; received even "
                f"values {even}. Use 1, 3, 5, or 7."
            )
            raise ValueError(msg)
        unsupported = [size for size in self.kernel_sizes if size not in (1, 3, 5, 7)]
        if unsupported:
            msg = f"kernel_sizes {unsupported} are outside the supported set (1, 3, 5, 7)"
            raise ValueError(msg)
        if any(ratio <= 0 for ratio in self.expansion_ratios):
            msg = f"expansion_ratios must be positive, received {list(self.expansion_ratios)}"
            raise ValueError(msg)
        return self

    @property
    def parametric_operations(self) -> tuple[OperationType, ...]:
        """Operations that own weights and may change the channel count."""
        return tuple(op for op in self.operations if op.can_change_channels)


class StemChoices(_SpaceModel):
    """Choices available to the entry convolution.

    Attributes:
        out_channels: Permitted stem widths.
        kernel_sizes: Permitted stem kernel sizes.
        strides: Permitted stem strides.
        normalizations: Permitted normalisation layers.
        activations: Permitted nonlinearities.
    """

    out_channels: tuple[int, ...] = (16, 24, 32)
    kernel_sizes: tuple[int, ...] = (3,)
    strides: tuple[int, ...] = (1,)
    normalizations: tuple[NormalizationType, ...] = (NormalizationType.BATCH,)
    activations: tuple[ActivationType, ...] = (ActivationType.RELU,)

    @model_validator(mode="after")
    def _validate(self) -> StemChoices:
        """De-duplicate choice sets and validate ranges.

        Returns:
            ``self``.

        Raises:
            ValueError: If a choice set is empty or contains an invalid value.
        """
        object.__setattr__(self, "out_channels", _unique_ordered(self.out_channels, "out_channels"))
        object.__setattr__(self, "kernel_sizes", _unique_ordered(self.kernel_sizes, "kernel_sizes"))
        object.__setattr__(self, "strides", _unique_ordered(self.strides, "strides"))
        object.__setattr__(
            self, "normalizations", _unique_ordered(self.normalizations, "normalizations")
        )
        object.__setattr__(self, "activations", _unique_ordered(self.activations, "activations"))
        if any(width <= 0 for width in self.out_channels):
            msg = f"stem out_channels must be positive, received {list(self.out_channels)}"
            raise ValueError(msg)
        if any(size not in (1, 3, 5, 7) for size in self.kernel_sizes):
            msg = f"stem kernel_sizes must be in (1, 3, 5, 7), received {list(self.kernel_sizes)}"
            raise ValueError(msg)
        if any(stride < 1 or stride > 4 for stride in self.strides):
            msg = f"stem strides must be in [1, 4], received {list(self.strides)}"
            raise ValueError(msg)
        return self


class HeadChoices(_SpaceModel):
    """Choices available to the classifier head.

    Attributes:
        poolings: Permitted global pooling operations.
        hidden_units: Permitted hidden widths; ``0`` means no hidden layer.
        dropouts: Permitted dropout probabilities.
        activations: Permitted nonlinearities for the hidden layer.
    """

    poolings: tuple[PoolingType, ...] = (PoolingType.AVG,)
    hidden_units: tuple[int, ...] = (0, 64)
    dropouts: tuple[float, ...] = (0.0, 0.1)
    activations: tuple[ActivationType, ...] = (ActivationType.RELU,)

    @model_validator(mode="after")
    def _validate(self) -> HeadChoices:
        """De-duplicate choice sets and validate ranges.

        Returns:
            ``self``.

        Raises:
            ValueError: If a choice set is empty or a dropout is outside ``[0, 0.9]``.
        """
        object.__setattr__(self, "poolings", _unique_ordered(self.poolings, "poolings"))
        object.__setattr__(self, "hidden_units", _unique_ordered(self.hidden_units, "hidden_units"))
        object.__setattr__(self, "dropouts", _unique_ordered(self.dropouts, "dropouts"))
        object.__setattr__(self, "activations", _unique_ordered(self.activations, "activations"))
        if any(width < 0 for width in self.hidden_units):
            msg = f"head hidden_units must be non-negative, received {list(self.hidden_units)}"
            raise ValueError(msg)
        invalid = [value for value in self.dropouts if not 0.0 <= value <= 0.9]
        if invalid:
            msg = f"head dropouts must lie in [0.0, 0.9], received {invalid}"
            raise ValueError(msg)
        return self


class SpaceConstraints(_SpaceModel):
    """Hard feasibility limits applied to every candidate.

    A candidate violating any constraint is rejected before it is trained. Constraints
    are evaluated analytically from the genotype, so rejection costs microseconds.

    Attributes:
        max_parameters: Upper bound on trainable parameters, or ``None`` for no bound.
        min_parameters: Lower bound on trainable parameters. Useful to stop a search
            collapsing onto degenerate all-pooling networks that train fast and score
            poorly.
        max_multiply_accumulates: Upper bound on estimated MACs per image.
        max_total_stride: Upper bound on the product of all strides.
        min_final_resolution: Minimum spatial extent of the final feature map. A 1x1
            feature map discards all spatial structure before pooling, which is legal
            but usually accidental.
        max_depth: Upper bound on total block count.
    """

    max_parameters: int | None = Field(default=None, ge=1)
    min_parameters: int | None = Field(default=None, ge=1)
    max_multiply_accumulates: int | None = Field(default=None, ge=1)
    max_total_stride: int | None = Field(default=None, ge=1)
    min_final_resolution: Annotated[int, Field(ge=1)] = 1
    max_depth: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> SpaceConstraints:
        """Reject an empty feasible parameter interval.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``min_parameters`` exceeds ``max_parameters``.
        """
        if (
            self.min_parameters is not None
            and self.max_parameters is not None
            and self.min_parameters > self.max_parameters
        ):
            msg = (
                f"min_parameters={self.min_parameters} exceeds "
                f"max_parameters={self.max_parameters}; no architecture can satisfy both"
            )
            raise ValueError(msg)
        return self


class SearchSpace(_SpaceModel):
    """A complete, validated search space.

    Attributes:
        name: Human-readable identifier persisted with search runs.
        schema_version: Version of the search-space schema.
        input_channels: Channels of the input tensor.
        input_size: Spatial extent of the square input.
        num_classes: Number of classification outputs.
        num_stages: Permitted stage counts.
        blocks_per_stage: Permitted block counts within a stage.
        stage_channels: Permitted stage widths, in ascending order.
        stage_strides: Permitted strides for the first block of a stage.
        monotonic_widths: When ``True``, a stage may never be narrower than the stage
            before it. This encodes the standard pyramidal design and removes a large
            region of the space that is empirically unproductive. Documented as a bias:
            architectures that widen then narrow are excluded by construction.
        block: Per-block choices.
        stem: Stem choices.
        head: Head choices.
        constraints: Hard feasibility limits.
    """

    name: str = "default_cnn"
    schema_version: int = SEARCH_SPACE_SCHEMA_VERSION
    input_channels: Annotated[int, Field(ge=1, le=16)] = 3
    input_size: Annotated[int, Field(ge=4, le=1024)] = 32
    num_classes: Annotated[int, Field(ge=2, le=100_000)] = 10
    num_stages: tuple[int, ...] = (2, 3)
    blocks_per_stage: tuple[int, ...] = (1, 2, 3)
    stage_channels: tuple[int, ...] = (16, 32, 64, 128)
    stage_strides: tuple[int, ...] = (1, 2)
    monotonic_widths: bool = True
    block: BlockChoices = Field(default_factory=BlockChoices)
    stem: StemChoices = Field(default_factory=StemChoices)
    head: HeadChoices = Field(default_factory=HeadChoices)
    constraints: SpaceConstraints = Field(default_factory=SpaceConstraints)

    @model_validator(mode="after")
    def _validate(self) -> SearchSpace:
        """Validate macro choice sets and check that the space can produce anything.

        Returns:
            ``self``.

        Raises:
            ValueError: If a macro choice set is empty or contains invalid values, or if
                the space contains no operation capable of changing the channel count
                while widths are expected to change.
        """
        object.__setattr__(self, "num_stages", _unique_ordered(self.num_stages, "num_stages"))
        object.__setattr__(
            self, "blocks_per_stage", _unique_ordered(self.blocks_per_stage, "blocks_per_stage")
        )
        object.__setattr__(self, "stage_channels", tuple(sorted(set(self.stage_channels))))
        object.__setattr__(
            self, "stage_strides", _unique_ordered(self.stage_strides, "stage_strides")
        )
        if not self.stage_channels:
            msg = "stage_channels must contain at least one width"
            raise ValueError(msg)
        if any(count < 1 for count in self.num_stages):
            msg = f"num_stages values must be >= 1, received {list(self.num_stages)}"
            raise ValueError(msg)
        if max(self.num_stages) > 8:
            msg = f"num_stages values must be <= 8, received {list(self.num_stages)}"
            raise ValueError(msg)
        if any(count < 1 for count in self.blocks_per_stage):
            msg = f"blocks_per_stage values must be >= 1, received {list(self.blocks_per_stage)}"
            raise ValueError(msg)
        if max(self.blocks_per_stage) > 16:
            msg = f"blocks_per_stage values must be <= 16, received {list(self.blocks_per_stage)}"
            raise ValueError(msg)
        if any(width < 1 for width in self.stage_channels):
            msg = f"stage_channels must be positive, received {list(self.stage_channels)}"
            raise ValueError(msg)
        if any(stride < 1 or stride > 4 for stride in self.stage_strides):
            msg = f"stage_strides must lie in [1, 4], received {list(self.stage_strides)}"
            raise ValueError(msg)
        if not self.block.parametric_operations:
            msg = (
                "block.operations contains no parametric operation "
                f"({OperationType.CONV.value} or {OperationType.DW_SEP_CONV.value}); "
                "such a space can only produce networks with no learnable features"
            )
            raise ValueError(msg)
        return self

    # -- introspection ------------------------------------------------------------
    def per_block_choice_count(self) -> int:
        """Return the number of distinct configurations one block position admits.

        This is an upper bound: conditional narrowing (a pooling operation ignoring
        expansion ratio, for instance) collapses some of these into the same canonical
        block, so the true count is smaller.

        Returns:
            The upper bound on block configurations.
        """
        total = 0
        residual_factor = 2 if self.block.allow_residual else 1
        for operation in self.block.operations:
            kernels = len(self.block.kernel_sizes) if operation.uses_kernel_size else 1
            expansions = len(self.block.expansion_ratios) if operation.uses_expansion_ratio else 1
            norms = len(self.block.normalizations) if operation.is_parametric else 1
            activations = len(self.block.activations) if operation.is_parametric else 1
            residual = residual_factor if operation is not OperationType.IDENTITY else 1
            total += kernels * expansions * norms * activations * residual
        return total

    def cardinality_upper_bound(self) -> int:
        """Return an upper bound on the number of distinct architectures.

        The bound multiplies independent choice counts and therefore over-counts:
        constraints, the monotonic-width rule, and conditional canonicalisation all
        remove members. It is still the right order-of-magnitude statement to make in a
        report — "this space contains roughly ``10^k`` architectures" — and it makes
        vivid why exhaustive enumeration is not an option.

        Returns:
            An integer upper bound (possibly astronomically large).
        """
        per_block = self.per_block_choice_count()
        stem = (
            len(self.stem.out_channels)
            * len(self.stem.kernel_sizes)
            * len(self.stem.strides)
            * len(self.stem.normalizations)
            * len(self.stem.activations)
        )
        head = (
            len(self.head.poolings)
            * len(self.head.hidden_units)
            * len(self.head.dropouts)
            * len(self.head.activations)
        )
        per_stage = sum(
            len(self.stage_channels) * len(self.stage_strides) * per_block**depth
            for depth in self.blocks_per_stage
        )
        body = sum(per_stage**count for count in self.num_stages)
        return int(stem * head * body)

    def log10_cardinality(self) -> float:
        """Return ``log10`` of :meth:`cardinality_upper_bound`, for compact display."""
        bound = self.cardinality_upper_bound()
        return math.log10(bound) if bound > 0 else 0.0

    def describe(self) -> str:
        """Return a human-readable multi-line description of the space."""
        return "\n".join(
            [
                f"Search space '{self.name}' (schema v{self.schema_version})",
                f"  input           : {self.input_channels}x{self.input_size}x{self.input_size}"
                f" -> {self.num_classes} classes",
                f"  stages          : {list(self.num_stages)}",
                f"  blocks per stage: {list(self.blocks_per_stage)}",
                f"  stage widths    : {list(self.stage_channels)}"
                f"{' (monotonic)' if self.monotonic_widths else ''}",
                f"  stage strides   : {list(self.stage_strides)}",
                f"  operations      : {[op.value for op in self.block.operations]}",
                f"  kernel sizes    : {list(self.block.kernel_sizes)}",
                f"  expansions      : {list(self.block.expansion_ratios)}",
                f"  normalisations  : {[n.value for n in self.block.normalizations]}",
                f"  activations     : {[a.value for a in self.block.activations]}",
                f"  residuals       : {'allowed' if self.block.allow_residual else 'disabled'}",
                f"  approx. size    : 1e{self.log10_cardinality():.1f} architectures (upper bound)",
            ]
        )

    def require_non_empty(self) -> None:
        """Raise if the space is obviously infeasible.

        This catches the common misconfiguration of a parameter ceiling so low that no
        architecture can satisfy it, which would otherwise show up as an opaque
        "sampler exhausted" error much later.

        Raises:
            SearchSpaceError: If the smallest expressible architecture already exceeds
                ``constraints.max_parameters``.
        """
        limit = self.constraints.max_parameters
        if limit is None:
            return
        # Smallest possible: one stage, one block, narrowest width, no hidden head.
        narrowest = min(self.stage_channels)
        smallest_stem = min(self.stem.out_channels)
        # A 1x1 convolution from the narrowest stem width plus the classifier.
        floor = smallest_stem * self.input_channels + narrowest * smallest_stem
        floor += narrowest * self.num_classes + self.num_classes
        if floor > limit:
            msg = (
                f"constraints.max_parameters={limit} is below the minimum possible "
                f"parameter count of roughly {floor} for this space; no candidate can "
                "ever be feasible. Raise max_parameters or narrow stage_channels."
            )
            raise SearchSpaceError(
                msg, details={"max_parameters": limit, "minimum_possible": floor}
            )


__all__ = [
    "SEARCH_SPACE_SCHEMA_VERSION",
    "BlockChoices",
    "HeadChoices",
    "SearchSpace",
    "SpaceConstraints",
    "StemChoices",
]
