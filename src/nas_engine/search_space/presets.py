"""Named, ready-to-use search spaces.

Three presets cover the project's needs:

``default_cnn``
    The demonstration space for 32x32 image classification. Roughly the size of a small
    ResNet/MobileNet hybrid family: two or three stages, up to three blocks each, widths
    from 16 to 128. Large enough to contain genuinely different architectures, small
    enough that a few dozen evaluations on CPU produce a meaningful ranking.

``tiny_cnn``
    Deliberately small: two stages, one or two blocks, two widths, two operations. Used
    by the test suite so that end-to-end runs finish in seconds and so that duplicate
    detection and exhaustion behaviour can actually be triggered.

``micro_cnn``
    Smaller still, and nearly exhaustible. Used by the smoke configuration and by tests
    that need the sampler to run out of novel candidates.

Presets are *functions*, not module-level constants, because a search space is
parameterised by the dataset's input size and class count. Returning a fresh frozen
object each call also removes any chance of one test mutating a shared instance.
"""

from __future__ import annotations

from collections.abc import Callable

from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)
from nas_engine.exceptions import SearchSpaceError
from nas_engine.search_space.space import (
    BlockChoices,
    HeadChoices,
    SearchSpace,
    SpaceConstraints,
    StemChoices,
)


def default_cnn_space(
    *, input_size: int = 32, num_classes: int = 10, input_channels: int = 3
) -> SearchSpace:
    """Return the demonstration CNN space for small images.

    Args:
        input_size: Square input extent.
        num_classes: Number of classes.
        input_channels: Input channel count.

    Returns:
        A validated :class:`~nas_engine.search_space.space.SearchSpace`.
    """
    return SearchSpace(
        name="default_cnn",
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        num_stages=(2, 3),
        blocks_per_stage=(1, 2, 3),
        stage_channels=(16, 32, 64, 128),
        stage_strides=(1, 2),
        monotonic_widths=True,
        block=BlockChoices(
            operations=(
                OperationType.CONV,
                OperationType.DW_SEP_CONV,
                OperationType.IDENTITY,
                OperationType.MAX_POOL,
                OperationType.AVG_POOL,
            ),
            kernel_sizes=(3, 5),
            expansion_ratios=(1.0, 2.0, 4.0),
            normalizations=(NormalizationType.BATCH, NormalizationType.GROUP),
            activations=(ActivationType.RELU, ActivationType.SILU),
            allow_residual=True,
        ),
        stem=StemChoices(
            out_channels=(16, 24, 32),
            kernel_sizes=(3,),
            strides=(1,),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.RELU,),
        ),
        head=HeadChoices(
            poolings=(PoolingType.AVG, PoolingType.MAX),
            hidden_units=(0, 64, 128),
            dropouts=(0.0, 0.1, 0.2),
            activations=(ActivationType.RELU,),
        ),
        constraints=SpaceConstraints(
            max_parameters=2_000_000,
            min_parameters=1_000,
            max_multiply_accumulates=200_000_000,
            min_final_resolution=2,
            max_depth=12,
        ),
    )


def tiny_cnn_space(
    *, input_size: int = 16, num_classes: int = 4, input_channels: int = 3
) -> SearchSpace:
    """Return a small space suitable for fast tests.

    Args:
        input_size: Square input extent.
        num_classes: Number of classes.
        input_channels: Input channel count.

    Returns:
        A validated search space with a few thousand members.
    """
    return SearchSpace(
        name="tiny_cnn",
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        num_stages=(1, 2),
        blocks_per_stage=(1, 2),
        stage_channels=(8, 16),
        stage_strides=(1, 2),
        monotonic_widths=True,
        block=BlockChoices(
            operations=(
                OperationType.CONV,
                OperationType.DW_SEP_CONV,
                OperationType.MAX_POOL,
                OperationType.IDENTITY,
            ),
            kernel_sizes=(3,),
            expansion_ratios=(1.0, 2.0),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.RELU,),
            allow_residual=True,
        ),
        stem=StemChoices(
            out_channels=(8,),
            kernel_sizes=(3,),
            strides=(1,),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.RELU,),
        ),
        head=HeadChoices(
            poolings=(PoolingType.AVG,),
            hidden_units=(0, 16),
            dropouts=(0.0,),
            activations=(ActivationType.RELU,),
        ),
        constraints=SpaceConstraints(
            max_parameters=200_000,
            min_final_resolution=1,
            max_depth=4,
        ),
    )


def micro_cnn_space(
    *, input_size: int = 8, num_classes: int = 3, input_channels: int = 3
) -> SearchSpace:
    """Return a near-exhaustible space used for smoke tests and exhaustion behaviour.

    Args:
        input_size: Square input extent.
        num_classes: Number of classes.
        input_channels: Input channel count.

    Returns:
        A validated search space with only a handful of members.
    """
    return SearchSpace(
        name="micro_cnn",
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        num_stages=(1,),
        blocks_per_stage=(1,),
        stage_channels=(4, 8),
        stage_strides=(1,),
        monotonic_widths=True,
        block=BlockChoices(
            operations=(OperationType.CONV,),
            kernel_sizes=(3,),
            expansion_ratios=(1.0,),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.RELU,),
            allow_residual=False,
        ),
        stem=StemChoices(
            out_channels=(4,),
            kernel_sizes=(3,),
            strides=(1,),
            normalizations=(NormalizationType.BATCH,),
            activations=(ActivationType.RELU,),
        ),
        head=HeadChoices(
            poolings=(PoolingType.AVG,),
            hidden_units=(0,),
            dropouts=(0.0,),
            activations=(ActivationType.RELU,),
        ),
        constraints=SpaceConstraints(min_final_resolution=1),
    )


#: Registry mapping preset names to factory functions.
PRESETS: dict[str, Callable[..., SearchSpace]] = {
    "default_cnn": default_cnn_space,
    "tiny_cnn": tiny_cnn_space,
    "micro_cnn": micro_cnn_space,
}


def get_preset(
    name: str,
    *,
    input_size: int | None = None,
    num_classes: int | None = None,
    input_channels: int | None = None,
) -> SearchSpace:
    """Build a preset space by name.

    Args:
        name: Preset identifier; one of :data:`PRESETS`.
        input_size: Override the preset's input extent.
        num_classes: Override the preset's class count.
        input_channels: Override the preset's input channel count.

    Returns:
        The constructed space.

    Raises:
        SearchSpaceError: If ``name`` is not a known preset.
    """
    factory = PRESETS.get(name)
    if factory is None:
        msg = f"unknown search-space preset '{name}'; available presets are {sorted(PRESETS)}"
        raise SearchSpaceError(msg, details={"name": name, "available": sorted(PRESETS)})
    overrides: dict[str, int] = {}
    if input_size is not None:
        overrides["input_size"] = input_size
    if num_classes is not None:
        overrides["num_classes"] = num_classes
    if input_channels is not None:
        overrides["input_channels"] = input_channels
    return factory(**overrides)


__all__ = [
    "PRESETS",
    "default_cnn_space",
    "get_preset",
    "micro_cnn_space",
    "tiny_cnn_space",
]
