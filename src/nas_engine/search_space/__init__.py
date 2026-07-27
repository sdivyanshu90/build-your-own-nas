"""Search-space definition, sampling, repair, mutation, and validation.

This package answers "which architectures may the search consider, and how do we draw
and perturb them?" It depends on :mod:`nas_engine.architectures` for the genotype and on
nothing else in the project, so a search space can be defined, sampled, and validated
without a database, a dataset, or PyTorch.
"""

from nas_engine.search_space.mutation import (
    DEFAULT_OPERATORS,
    MutationOperator,
    MutationResult,
    MutationStatistics,
)
from nas_engine.search_space.presets import (
    PRESETS,
    default_cnn_space,
    get_preset,
    micro_cnn_space,
    tiny_cnn_space,
)
from nas_engine.search_space.repair import RepairReport, repair_architecture, stage_widths
from nas_engine.search_space.sampler import ArchitectureSampler, SamplerStatistics
from nas_engine.search_space.space import (
    SEARCH_SPACE_SCHEMA_VERSION,
    BlockChoices,
    HeadChoices,
    SearchSpace,
    SpaceConstraints,
    StemChoices,
)
from nas_engine.search_space.validation import (
    ValidationIssue,
    ValidationReport,
    check_architecture,
    validate_architecture,
)

__all__ = [
    "DEFAULT_OPERATORS",
    "PRESETS",
    "SEARCH_SPACE_SCHEMA_VERSION",
    "ArchitectureSampler",
    "BlockChoices",
    "HeadChoices",
    "MutationOperator",
    "MutationResult",
    "MutationStatistics",
    "RepairReport",
    "SamplerStatistics",
    "SearchSpace",
    "SpaceConstraints",
    "StemChoices",
    "ValidationIssue",
    "ValidationReport",
    "check_architecture",
    "default_cnn_space",
    "get_preset",
    "micro_cnn_space",
    "repair_architecture",
    "stage_widths",
    "tiny_cnn_space",
    "validate_architecture",
]
