"""nas-engine: Neural Architecture Search from scratch.

Quick start
-----------
.. code-block:: python

    from nas_engine import SearchConfig, SearchEngine

    config = SearchConfig.from_yaml("configs/random_search.yaml")
    engine = SearchEngine(config)
    result = engine.run()
    print(result.summary())

The public stability boundary
-----------------------------
**Everything exported from this module is public** and follows semantic versioning: names
will not be removed or given incompatible signatures within a major version. That covers
:class:`~nas_engine.config.models.SearchConfig`,
:class:`~nas_engine.orchestration.engine.SearchEngine`,
:class:`~nas_engine.orchestration.result.SearchResult`, the architecture genotype types,
the search-space types, the search-strategy interface, and the exception taxonomy.

**Everything else is internal.** Submodules may be renamed, split, or rewritten in a minor
release. In particular, no guarantee is made about:

* the database schema or the ORM models (use
  :class:`~nas_engine.persistence.repository.SearchRepository`, which is public);
* the exact contents of a strategy's ``state_dict``;
* module paths of helper functions not re-exported here.

Extension points that *are* public:

* :class:`~nas_engine.search.strategy.SearchStrategy` — implement it and register it with
  :func:`~nas_engine.search.registry.register_strategy`.
* :class:`~nas_engine.datasets.base.DatasetProvider` — implement it and register it with
  :func:`~nas_engine.datasets.registry.register_provider`.
* :class:`~nas_engine.objectives.objective.Objective` and
  :class:`~nas_engine.objectives.constraints.MetricConstraint` — construct them directly.

See ``docs/architecture/component-design.md`` for the full boundary discussion.
"""

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import (
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
from nas_engine.config.models import SearchConfig
from nas_engine.datasets.base import DatasetBundle, DatasetProvider
from nas_engine.datasets.registry import register_provider
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.result import EvaluationResult
from nas_engine.exceptions import (
    ArchitectureValidationError,
    ConfigurationError,
    NasEngineError,
    OrchestrationError,
    PersistenceError,
    SearchSpaceError,
)
from nas_engine.models.builder import ModelBuilder, NasNetwork, build_model
from nas_engine.objectives.constraints import ConstraintSet, MetricConstraint
from nas_engine.objectives.objective import (
    Objective,
    ObjectiveDirection,
    ObjectiveSet,
    default_objectives,
)
from nas_engine.objectives.ranking import RankedCandidate, RankingResult, rank_candidates
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.orchestration.result import SearchResult, StopReason
from nas_engine.persistence.repository import (
    CandidateSummary,
    SearchRepository,
    SearchSummary,
)
from nas_engine.search.registry import available_strategies, register_strategy
from nas_engine.search.strategy import Observation, Proposal, SearchStrategy
from nas_engine.search_space.presets import get_preset
from nas_engine.search_space.space import SearchSpace

try:  # pragma: no cover - trivial metadata lookup
    from importlib.metadata import version

    __version__ = version("nas-engine")
except Exception:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ActivationType",
    "ArchitectureSpec",
    "ArchitectureSummary",
    "ArchitectureValidationError",
    "BlockSpec",
    "CandidateState",
    "CandidateSummary",
    "ConfigurationError",
    "ConstraintSet",
    "DatasetBundle",
    "DatasetProvider",
    "EvaluationResult",
    "HeadSpec",
    "MetricConstraint",
    "ModelBuilder",
    "NasEngineError",
    "NasNetwork",
    "NormalizationType",
    "Objective",
    "ObjectiveDirection",
    "ObjectiveSet",
    "Observation",
    "OperationType",
    "OrchestrationError",
    "PersistenceError",
    "PoolingType",
    "Proposal",
    "RankedCandidate",
    "RankingResult",
    "SearchConfig",
    "SearchEngine",
    "SearchRepository",
    "SearchResult",
    "SearchSpace",
    "SearchSpaceError",
    "SearchStrategy",
    "SearchSummary",
    "StageSpec",
    "StemSpec",
    "StopReason",
    "TrainingBudget",
    "__version__",
    "architecture_hash",
    "available_strategies",
    "build_model",
    "default_objectives",
    "get_preset",
    "rank_candidates",
    "register_provider",
    "register_strategy",
    "summarise",
]
