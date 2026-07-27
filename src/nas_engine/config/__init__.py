"""Validated configuration: models, loading, and precedence.

This package sits at the boundary between untrusted input (YAML files, environment
variables, command-line flags) and the domain. It validates once and hands the domain
plain, already-checked dataclasses, which is why no domain module needs a Pydantic import.
"""

from nas_engine.config.loader import (
    ENV_PREFIX,
    build_config,
    check_config_compatibility,
    deep_merge,
    dump_yaml,
    load_config,
    parse_environment,
    parse_overrides,
    read_yaml,
)
from nas_engine.config.models import (
    CONFIG_VERSION,
    AlgorithmConfig,
    BudgetConfig,
    ConcurrencyConfig,
    ConstraintEntry,
    DatasetConfig,
    EvaluationConfig,
    HardwareConfig,
    LoggingConfig,
    ObjectiveEntry,
    ObjectivesConfig,
    OptimizerConfig,
    PersistenceConfig,
    ProjectConfig,
    ReproducibilityConfig,
    RetryConfig,
    SchedulerConfig,
    SearchConfig,
    SearchSpaceConfig,
    TrainingConfig,
)

__all__ = [
    "CONFIG_VERSION",
    "ENV_PREFIX",
    "AlgorithmConfig",
    "BudgetConfig",
    "ConcurrencyConfig",
    "ConstraintEntry",
    "DatasetConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "LoggingConfig",
    "ObjectiveEntry",
    "ObjectivesConfig",
    "OptimizerConfig",
    "PersistenceConfig",
    "ProjectConfig",
    "ReproducibilityConfig",
    "RetryConfig",
    "SchedulerConfig",
    "SearchConfig",
    "SearchSpaceConfig",
    "TrainingConfig",
    "build_config",
    "check_config_compatibility",
    "deep_merge",
    "dump_yaml",
    "load_config",
    "parse_environment",
    "parse_overrides",
    "read_yaml",
]
