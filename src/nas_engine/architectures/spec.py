"""The architecture genotype: a validated, immutable, canonicalising data model.

Genotype versus phenotype
-------------------------
The **genotype** is the description of an architecture — a small tree of plain data.
The **phenotype** is the :class:`torch.nn.Module` produced from it. Keeping them apart
is the single most important structural decision in this project:

* Genotypes are cheap. Millions can be sampled, hashed, compared, stored, mutated, and
  shipped between processes without allocating a single tensor.
* Genotypes are serialisable. A ``nn.Module`` is not portably serialisable without
  ``pickle``, which is unsafe for untrusted input; a genotype is pure JSON.
* Genotypes are comparable. Two modules with identical structure are different Python
  objects; two genotypes with identical structure are equal and hash identically.

Canonicalisation
----------------
The search space contains **conditional** choices: ``expansion_ratio`` only means
something for a depthwise-separable convolution; ``kernel_size`` means nothing for an
identity operation. If those inactive values were preserved, two architectures that
build byte-identical models would receive different hashes, and the engine would waste
budget training the same network repeatedly.

Every model in this module therefore **canonicalises on construction**: inactive fields
are forced to a fixed sentinel value. Canonicalisation is idempotent
(``canon(canon(x)) == canon(x)``) and is verified by property tests in
``tests/property/test_canonicalisation_properties.py``.

Immutability
------------
All models are frozen. Mutation operators must build a new object via
:meth:`pydantic.BaseModel.model_copy` with ``update=...``, which makes it structurally
impossible for a mutation to modify a parent that is still referenced by the
evolutionary population.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)

#: Version of the genotype schema. Bump when the *meaning* or *shape* of the genotype
#: changes in a way that invalidates stored hashes. Persisted with every specification
#: so old records remain interpretable.
ARCHITECTURE_SCHEMA_VERSION: int = 1

#: Floats in the genotype are quantised to this many decimal places before hashing.
#: Without quantisation, ``0.1 + 0.2`` and ``0.3`` would serialise differently and
#: produce different architecture hashes for the same network.
FLOAT_PRECISION: int = 6

#: Kernel sizes must be odd so that "same" padding (``k // 2``) is exact. An even
#: kernel cannot preserve spatial size with symmetric padding.
_ALLOWED_KERNEL_SIZES: frozenset[int] = frozenset({1, 3, 5, 7})

_FrozenModelT = TypeVar("_FrozenModelT", bound="_FrozenModel")


def quantise(value: float) -> float:
    """Round a float to the canonical precision used for hashing.

    Args:
        value: Raw float.

    Returns:
        ``value`` rounded to :data:`FLOAT_PRECISION` decimal places, with ``-0.0``
        normalised to ``0.0`` so the two never hash differently.
    """
    rounded = round(float(value), FLOAT_PRECISION)
    return 0.0 if rounded == 0.0 else rounded


class _FrozenModel(BaseModel):
    """Base for every genotype node: frozen, strict about unknown fields.

    ``extra="forbid"`` matters for security as well as correctness: architecture JSON
    imported from disk is untrusted, and silently ignoring unknown keys would let a
    malformed or malicious document appear valid while carrying hidden payloads.

    Canonicalising validators mutate ``self`` through :func:`_force`, because Pydantic
    v2 ignores a *new* instance returned from an ``after`` model validator when the
    model is built through ``__init__``. Mutating in place is the supported way to
    normalise a frozen model during validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def evolve(self: _FrozenModelT, **changes: Any) -> _FrozenModelT:
        """Return a re-validated copy with ``changes`` applied.

        Unlike :meth:`pydantic.BaseModel.model_copy`, this routes through the model
        constructor, so canonicalising validators run again. Mutation operators must
        use this: ``model_copy(update=...)`` would happily produce a non-canonical
        genotype whose hash disagrees with an equivalent freshly built one.

        Args:
            **changes: Field values to override.

        Returns:
            A new, canonical instance of the same model type.
        """
        data = {**self.__dict__, **changes}
        return type(self)(**data)


def _force(model: BaseModel, field: str, value: Any) -> None:
    """Assign a canonical value to a frozen model during validation.

    Args:
        model: Model being validated.
        field: Field name to overwrite.
        value: Canonical value.
    """
    object.__setattr__(model, field, value)


class BlockSpec(_FrozenModel):
    """One operation inside a stage.

    Attributes:
        operation: Which primitive the block performs.
        kernel_size: Spatial extent of the operation's window. Forced to ``1`` for
            :attr:`~nas_engine.architectures.types.OperationType.IDENTITY`.
        expansion_ratio: Inverted-bottleneck width multiplier. Active only for
            depthwise-separable convolutions; forced to ``1.0`` otherwise.
        out_channels: Channels produced. Must equal the input channel count for
            channel-preserving operations (validated at graph level, where the input
            count is known).
        stride: Spatial stride. Forced to ``1`` for identity.
        use_residual: Whether to add the block input to its output. Requires
            ``stride == 1`` and matching channel counts; forced to ``False`` for
            identity, where a residual would merely double the signal.
        normalization: Normalisation layer. Forced to ``NONE`` for non-parametric
            operations, which have nothing to normalise in a canonical form.
        activation: Nonlinearity. Forced to ``IDENTITY`` for non-parametric operations.
    """

    operation: OperationType
    kernel_size: Annotated[int, Field(ge=1, le=7)] = 3
    expansion_ratio: Annotated[float, Field(ge=0.25, le=8.0)] = 1.0
    out_channels: Annotated[int, Field(ge=1, le=4096)] = 32
    stride: Annotated[int, Field(ge=1, le=4)] = 1
    use_residual: bool = False
    normalization: NormalizationType = NormalizationType.BATCH
    activation: ActivationType = ActivationType.RELU

    @model_validator(mode="after")
    def _canonicalise(self) -> BlockSpec:
        """Force inactive conditional fields to their canonical sentinel values.

        Returns:
            ``self``, canonicalised in place.

        Raises:
            ValueError: If an active kernel size is even, since symmetric "same"
                padding is impossible for even kernels.
        """
        operation = self.operation

        if operation.uses_kernel_size:
            if self.kernel_size not in _ALLOWED_KERNEL_SIZES:
                allowed = sorted(_ALLOWED_KERNEL_SIZES)
                msg = (
                    f"kernel_size={self.kernel_size} is not supported for operation "
                    f"'{operation.value}'; expected one of {allowed} (odd sizes only, "
                    "so that symmetric same-padding is exact)"
                )
                raise ValueError(msg)
        elif self.kernel_size != 1:
            _force(self, "kernel_size", 1)

        expansion = 1.0 if not operation.uses_expansion_ratio else quantise(self.expansion_ratio)
        if expansion != self.expansion_ratio:
            _force(self, "expansion_ratio", expansion)

        if operation is OperationType.IDENTITY:
            if self.stride != 1:
                _force(self, "stride", 1)
            if self.use_residual:
                _force(self, "use_residual", False)

        if not operation.is_parametric:
            if self.normalization is not NormalizationType.NONE:
                _force(self, "normalization", NormalizationType.NONE)
            if self.activation is not ActivationType.IDENTITY:
                _force(self, "activation", ActivationType.IDENTITY)

        return self

    def describe(self) -> str:
        """Return a compact one-line description used in architecture summaries."""
        if self.operation is OperationType.IDENTITY:
            return "identity"
        parts = [f"{self.operation.value}", f"k{self.kernel_size}", f"s{self.stride}"]
        if self.operation.can_change_channels:
            parts.append(f"c{self.out_channels}")
        if self.operation.uses_expansion_ratio:
            parts.append(f"e{self.expansion_ratio:g}")
        if self.operation.is_parametric:
            parts.append(self.normalization.value)
            parts.append(self.activation.value)
        if self.use_residual:
            parts.append("res")
        return "-".join(parts)


class StageSpec(_FrozenModel):
    """A contiguous group of blocks operating at one spatial resolution family.

    Stages exist because real convolutional networks are organised as a sequence of
    resolution levels: the first block of a stage typically downsamples and widens, and
    the remaining blocks refine features at the new resolution. Encoding that structure
    into the genotype rather than leaving it to chance dramatically shrinks the search
    space without excluding good architectures.

    Attributes:
        blocks: Ordered, non-empty tuple of blocks.
    """

    blocks: Annotated[tuple[BlockSpec, ...], Field(min_length=1, max_length=16)]

    @property
    def stride_product(self) -> int:
        """Total spatial downsampling factor contributed by this stage."""
        product = 1
        for block in self.blocks:
            product *= block.stride
        return product

    @property
    def out_channels(self) -> int:
        """Channel count produced by the final block of the stage."""
        return self.blocks[-1].out_channels


class StemSpec(_FrozenModel):
    """The fixed entry convolution applied before the first stage.

    A stem is not searched over as richly as the body because its job is narrow: lift
    the input from 3 channels to a workable width and optionally reduce resolution once.
    Searching it aggressively adds cardinality with little payoff.

    Attributes:
        out_channels: Width produced by the stem.
        kernel_size: Stem kernel size.
        stride: Stem stride; ``2`` halves the resolution immediately, which is the usual
            choice for larger inputs.
        normalization: Normalisation after the stem convolution.
        activation: Nonlinearity after normalisation.
    """

    out_channels: Annotated[int, Field(ge=1, le=1024)] = 16
    kernel_size: Annotated[int, Field(ge=1, le=7)] = 3
    stride: Annotated[int, Field(ge=1, le=4)] = 1
    normalization: NormalizationType = NormalizationType.BATCH
    activation: ActivationType = ActivationType.RELU

    @model_validator(mode="after")
    def _check_kernel(self) -> StemSpec:
        """Reject even kernel sizes, which cannot use exact same-padding.

        Returns:
            ``self`` when valid.

        Raises:
            ValueError: If the kernel size is not an allowed odd value.
        """
        if self.kernel_size not in _ALLOWED_KERNEL_SIZES:
            allowed = sorted(_ALLOWED_KERNEL_SIZES)
            msg = f"stem kernel_size={self.kernel_size} is invalid; expected one of {allowed}"
            raise ValueError(msg)
        return self


class HeadSpec(_FrozenModel):
    """The classifier head applied after the final stage.

    Attributes:
        pooling: Global pooling that collapses the spatial dimensions.
        hidden_units: Width of an optional hidden layer; ``0`` means the pooled
            features feed the classifier directly.
        dropout: Dropout probability applied immediately before the classifier.
        activation: Nonlinearity after the hidden layer. Canonicalised to ``IDENTITY``
            when there is no hidden layer.
    """

    pooling: PoolingType = PoolingType.AVG
    hidden_units: Annotated[int, Field(ge=0, le=4096)] = 0
    dropout: Annotated[float, Field(ge=0.0, le=0.9)] = 0.0
    activation: ActivationType = ActivationType.RELU

    @model_validator(mode="after")
    def _canonicalise(self) -> HeadSpec:
        """Quantise dropout and neutralise the activation when unused.

        Returns:
            ``self``, canonicalised in place.
        """
        dropout = quantise(self.dropout)
        if dropout != self.dropout:
            _force(self, "dropout", dropout)
        if self.hidden_units == 0 and self.activation is not ActivationType.IDENTITY:
            _force(self, "activation", ActivationType.IDENTITY)
        return self


class ArchitectureSpec(_FrozenModel):
    """A complete, self-contained description of one candidate network.

    The specification carries its own input and output shape so that it can be
    validated, costed, and rebuilt in isolation — a stored architecture never depends
    on ambient configuration to be interpretable.

    Attributes:
        schema_version: Genotype schema version; see
            :data:`ARCHITECTURE_SCHEMA_VERSION`.
        input_channels: Channels of the input tensor (3 for RGB images).
        input_size: Spatial extent of the square input.
        num_classes: Number of classification outputs.
        stem: Entry convolution.
        stages: Ordered, non-empty tuple of stages.
        head: Classifier head.

    Example:
        >>> spec = ArchitectureSpec(
        ...     input_channels=3,
        ...     input_size=32,
        ...     num_classes=10,
        ...     stem=StemSpec(out_channels=16),
        ...     stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV,
        ...                                         out_channels=16),)),),
        ...     head=HeadSpec(),
        ... )
        >>> spec.total_blocks
        1
    """

    schema_version: int = ARCHITECTURE_SCHEMA_VERSION
    input_channels: Annotated[int, Field(ge=1, le=16)] = 3
    input_size: Annotated[int, Field(ge=4, le=1024)] = 32
    num_classes: Annotated[int, Field(ge=2, le=100_000)] = 10
    stem: StemSpec = Field(default_factory=StemSpec)
    stages: Annotated[tuple[StageSpec, ...], Field(min_length=1, max_length=8)]
    head: HeadSpec = Field(default_factory=HeadSpec)

    @model_validator(mode="after")
    def _check_schema_version(self) -> ArchitectureSpec:
        """Reject genotypes written by a newer, unknown schema.

        Returns:
            ``self`` when the version is supported.

        Raises:
            ValueError: If ``schema_version`` exceeds the version this build understands.
        """
        if self.schema_version > ARCHITECTURE_SCHEMA_VERSION:
            msg = (
                f"architecture schema_version={self.schema_version} is newer than the "
                f"supported version {ARCHITECTURE_SCHEMA_VERSION}; upgrade nas-engine "
                "to read this specification"
            )
            raise ValueError(msg)
        if self.schema_version < 1:
            msg = f"architecture schema_version must be >= 1, received {self.schema_version}"
            raise ValueError(msg)
        return self

    # -- derived properties -------------------------------------------------------
    @property
    def num_stages(self) -> int:
        """Number of stages."""
        return len(self.stages)

    @property
    def total_blocks(self) -> int:
        """Total number of blocks across all stages."""
        return sum(len(stage.blocks) for stage in self.stages)

    @property
    def total_stride(self) -> int:
        """Product of every stride in the network, including the stem.

        The final feature map has spatial size ``ceil(input_size / total_stride)``;
        a total stride larger than ``input_size`` is a structural error.
        """
        product = self.stem.stride
        for stage in self.stages:
            product *= stage.stride_product
        return product

    @property
    def final_channels(self) -> int:
        """Channel count entering the head."""
        return self.stages[-1].out_channels

    def iter_blocks(self) -> list[tuple[int, int, BlockSpec]]:
        """Return ``(stage_index, block_index, block)`` triples in execution order."""
        return [
            (stage_index, block_index, block)
            for stage_index, stage in enumerate(self.stages)
            for block_index, block in enumerate(stage.blocks)
        ]

    def with_block(self, stage_index: int, block_index: int, block: BlockSpec) -> ArchitectureSpec:
        """Return a copy with one block replaced, leaving ``self`` untouched.

        Args:
            stage_index: Index of the stage containing the block.
            block_index: Index of the block within the stage.
            block: Replacement block.

        Returns:
            A new :class:`ArchitectureSpec`.

        Raises:
            IndexError: If either index is out of range.
        """
        if not 0 <= stage_index < len(self.stages):
            msg = f"stage_index {stage_index} out of range for {len(self.stages)} stages"
            raise IndexError(msg)
        stage = self.stages[stage_index]
        if not 0 <= block_index < len(stage.blocks):
            msg = f"block_index {block_index} out of range for {len(stage.blocks)} blocks"
            raise IndexError(msg)
        blocks = list(stage.blocks)
        blocks[block_index] = block
        stages = list(self.stages)
        stages[stage_index] = stage.evolve(blocks=tuple(blocks))
        return self.evolve(stages=tuple(stages))

    def with_stages(self, stages: tuple[StageSpec, ...]) -> ArchitectureSpec:
        """Return a copy with a replaced stage tuple.

        Args:
            stages: Replacement stages.

        Returns:
            A new :class:`ArchitectureSpec`.
        """
        return self.evolve(stages=stages)


__all__ = [
    "ARCHITECTURE_SCHEMA_VERSION",
    "FLOAT_PRECISION",
    "ArchitectureSpec",
    "BlockSpec",
    "HeadSpec",
    "StageSpec",
    "StemSpec",
    "quantise",
]
