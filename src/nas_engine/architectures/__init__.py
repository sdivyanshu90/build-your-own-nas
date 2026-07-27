"""Architecture genotypes: specification, canonical form, hashing, shapes, and cost.

This package is the vocabulary the whole system speaks. It has no dependency on
PyTorch, on the database, or on configuration — an architecture is pure data. The
:mod:`nas_engine.models` package turns that data into a network; every other package
merely stores, hashes, mutates, or ranks it.
"""

from nas_engine.architectures.canonical import (
    architectures_equal,
    from_canonical_dict,
    from_canonical_json,
    to_canonical_dict,
    to_canonical_json,
)
from nas_engine.architectures.cost import ArchitectureCost, compute_cost
from nas_engine.architectures.hashing import architecture_hash, short_hash
from nas_engine.architectures.lineage import LineageChain, LineageGraph, LineageNode
from nas_engine.architectures.shapes import (
    LayerShape,
    ShapeTrace,
    TensorShape,
    conv_output_size,
    infer_shapes,
    make_divisible,
)
from nas_engine.architectures.spec import (
    ARCHITECTURE_SCHEMA_VERSION,
    ArchitectureSpec,
    BlockSpec,
    HeadSpec,
    StageSpec,
    StemSpec,
)
from nas_engine.architectures.summary import ArchitectureSummary, summarise
from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)

__all__ = [
    "ARCHITECTURE_SCHEMA_VERSION",
    "ActivationType",
    "ArchitectureCost",
    "ArchitectureSpec",
    "ArchitectureSummary",
    "BlockSpec",
    "HeadSpec",
    "LayerShape",
    "LineageChain",
    "LineageGraph",
    "LineageNode",
    "NormalizationType",
    "OperationType",
    "PoolingType",
    "ShapeTrace",
    "StageSpec",
    "StemSpec",
    "TensorShape",
    "architecture_hash",
    "architectures_equal",
    "compute_cost",
    "conv_output_size",
    "from_canonical_dict",
    "from_canonical_json",
    "infer_shapes",
    "make_divisible",
    "short_hash",
    "summarise",
    "to_canonical_dict",
    "to_canonical_json",
]
