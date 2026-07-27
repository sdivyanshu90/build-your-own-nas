"""Define a search space by hand and inspect what it contains.

Search-space design frequently matters more than the choice of search algorithm, so it is
worth being deliberate about it. This example builds a space from scratch, reports its
size, samples from it, and shows what happens when a constraint is too tight.

Run it with::

    python examples/custom_search_space.py
"""

from __future__ import annotations

from nas_engine import get_preset
from nas_engine.architectures import architecture_hash, summarise
from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)
from nas_engine.exceptions import SearchSpaceError
from nas_engine.search_space import ArchitectureSampler, SearchSpace, validate_architecture
from nas_engine.search_space.space import (
    BlockChoices,
    HeadChoices,
    SpaceConstraints,
    StemChoices,
)


def build_mobile_style_space() -> SearchSpace:
    """Build a space biased towards small, deployable architectures.

    Design decisions encoded here, and why:

    * Only depthwise-separable convolutions and pooling. Dense convolutions dominate the
      parameter budget, and this space is for models that must be small.
    * Kernel sizes 3 and 5. Kernel 7 rarely pays for itself at 32x32 resolution.
    * Group normalisation as well as batch normalisation, because a deployed model may run
      with batch size 1 where batch statistics are unusable.
    * A hard parameter ceiling of 300 000, enforced analytically before anything is trained.

    Returns:
        The validated space.
    """
    return SearchSpace(
        name="mobile_style",
        input_channels=3,
        input_size=32,
        num_classes=10,
        num_stages=(2, 3, 4),
        blocks_per_stage=(1, 2),
        stage_channels=(16, 24, 32, 48),
        stage_strides=(1, 2),
        monotonic_widths=True,
        block=BlockChoices(
            operations=(
                OperationType.DW_SEP_CONV,
                OperationType.IDENTITY,
                OperationType.AVG_POOL,
            ),
            kernel_sizes=(3, 5),
            expansion_ratios=(1.0, 3.0, 6.0),
            normalizations=(NormalizationType.BATCH, NormalizationType.GROUP),
            activations=(ActivationType.RELU6, ActivationType.HARDSWISH),
            allow_residual=True,
        ),
        stem=StemChoices(
            out_channels=(8, 16),
            kernel_sizes=(3,),
            strides=(1, 2),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.HARDSWISH,),
        ),
        head=HeadChoices(
            poolings=(PoolingType.AVG,),
            hidden_units=(0, 96),
            dropouts=(0.0, 0.2),
            activations=(ActivationType.HARDSWISH,),
        ),
        constraints=SpaceConstraints(
            max_parameters=300_000,
            min_parameters=2_000,
            max_multiply_accumulates=60_000_000,
            min_final_resolution=2,
            max_depth=8,
        ),
    )


def main() -> int:
    """Build the space, sample from it, and demonstrate constraint rejection.

    Returns:
        A process exit code.
    """
    space = build_mobile_style_space()
    print(space.describe())
    print()

    sampler = ArchitectureSampler(space, seed=42)
    print("Five samples:")
    for index in range(5):
        spec = sampler.sample()
        validate_architecture(spec, space)
        print(f"  {index}. {summarise(spec).compact()}")
    print(f"\nSampler statistics: {sampler.statistics.to_dict()}")

    print("\nFull summary of the first sample:")
    first = ArchitectureSampler(space, seed=42).sample()
    print(summarise(first).to_text())
    print(f"\nIts stable identity: {architecture_hash(first)}")

    # A constraint tight enough to be infeasible is reported immediately, not after a
    # thousand wasted draws.
    print("\nWhat an impossible constraint looks like:")
    impossible = space.model_copy(update={"constraints": SpaceConstraints(max_parameters=10)})
    try:
        impossible.require_non_empty()
    except SearchSpaceError as error:
        print(f"  {error.message}")

    # Comparing against a preset shows how much the choice sets change the space size.
    preset = get_preset("default_cnn")
    print(
        f"\nSpace size (upper bound): mobile_style 1e{space.log10_cardinality():.1f} "
        f"versus default_cnn 1e{preset.log10_cardinality():.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
