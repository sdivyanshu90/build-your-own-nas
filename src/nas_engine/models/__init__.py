"""PyTorch phenotypes: turning architecture genotypes into runnable networks.

This package depends on :mod:`nas_engine.architectures` and on PyTorch, and on nothing
else. It knows how to build and measure a model; it knows nothing about datasets,
training loops, search strategies, or persistence.
"""

from nas_engine.models.blocks import ClassifierHead, NasBlock, SeparableConvBlock
from nas_engine.models.builder import (
    ModelBuilder,
    ModelSummary,
    NasNetwork,
    build_model,
    count_parameters,
    state_dict_bytes,
    summarize_model,
)
from nas_engine.models.initialization import initialize_weights
from nas_engine.models.operations import (
    build_activation,
    build_conv_bn_act,
    build_normalization,
    group_count,
)

__all__ = [
    "ClassifierHead",
    "ModelBuilder",
    "ModelSummary",
    "NasBlock",
    "NasNetwork",
    "SeparableConvBlock",
    "build_activation",
    "build_conv_bn_act",
    "build_model",
    "build_normalization",
    "count_parameters",
    "group_count",
    "initialize_weights",
    "state_dict_bytes",
    "summarize_model",
]
