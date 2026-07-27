# Repository manifest

Every file in the project: what it is for, what it exposes, what it depends on, and what
tests it.

Regenerate the source tables with `make manifest`. The prose around them is written by hand.

## How to read the tables

**Purpose** is each module's first docstring line, so the table cannot drift from the code
without the docstring drifting too.

**Public symbols** lists top-level names not starting with `_`, truncated with a `+n`
count. A module's `__init__.py` re-exports its subpackage's symbols; those rows show `—`
because the names belong to the modules underneath.

**Depends on** lists *internal* subpackages only — third-party imports are in
`pyproject.toml`. The list is the honest import graph, extracted from the AST, and it is
what makes the layering below checkable rather than aspirational.

**Tests** are the test files that import the module, abbreviated by directory:

| Prefix  | Directory                 |
| ------- | ------------------------- |
| `u:`    | `tests/unit/`             |
| `p:`    | `tests/property/`         |
| `i:`    | `tests/integration/`      |
| `e:`    | `tests/end_to_end/`       |
| `r:`    | `tests/regression/`       |
| `perf:` | `tests/performance/`      |
| `f:`    | `tests/failure_recovery/` |

`(via package)` marks a package `__init__.py`: it is exercised through the modules it
re-exports, and `tests/unit/test_public_api.py` asserts the re-exports themselves.

**Every non-`__init__` module is imported by at least one test file.** That is a weaker
statement than the coverage threshold, and a useful one: it means no module is entirely
unreached.

## Layering

Dependencies point one way. Reading the *Depends on* column top to bottom, no module
depends on anything below it in this list:

```text
utilities          no internal dependencies at all
observability      no internal dependencies at all
architectures      utilities
search_space       architectures, utilities
models             architectures
datasets           utilities
training           datasets, utilities, observability
objectives         (self-contained)
evaluation         architectures, datasets, models, training, utilities
search             architectures, search_space, evaluation
persistence        architectures, evaluation, orchestration.lifecycle
config             everything it constructs
orchestration      everything
reporting          objectives, persistence, architectures
cli                everything
```

Two entries deserve a note.

`persistence` importing `orchestration.lifecycle` looks like a layering violation and is
not: it imports the `CandidateState` enum and the transition table, so the repository can
*validate* a transition before writing it. The state machine is a domain vocabulary, not
orchestration logic. Putting it in `orchestration` keeps it next to the engine that drives
it; the alternative — a fifteenth top-level module for one enum — is worse.

`config` importing broadly is deliberate: a configuration model's job is to *build* the
thing it configures, so `SearchSpaceConfig.build()` returns a `SearchSpace` and
`ObjectivesConfig.build_objectives()` returns an `ObjectiveSet`. Validation therefore
happens once, in the constructor of the real object, rather than being duplicated in the
config layer.

---

## Source: `src/nas_engine/`

<!-- BEGIN GENERATED SOURCE TABLES -->

### `nas_engine/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `__init__.py` | 152 | nas-engine: Neural Architecture Search from scratch | — | architectures, config, datasets, evaluation, exceptions, models, objectives, orchestration, persistence, search, search_space | (via package) |
| `cli.py` | 1108 | The ``nas-engine`` command-line interface | `ExitCode`, `init`, `validate_config`, `search`, `resume`, `status` +11 | architectures, config, exceptions, objectives, observability, orchestration, persistence, reporting, utilities | e:full_searches, u:cli |
| `exceptions.py` | 291 | Error taxonomy for ``nas_engine`` | `NasEngineError`, `ConfigurationError`, `ConfigVersionError`, `SearchSpaceError`, `ArchitectureValidationError`, `ShapeInferenceError` +22 | — | f:failure_recovery, p:architecture_properties, p:search_properties, u:architectures +13 |

### `nas_engine/utilities/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `utilities/__init__.py` | 61 | Cross-cutting helpers with no dependencies on other ``nas_engine`` subpackages | — | — | (via package) |
| `utilities/determinism.py` | 123 | PyTorch determinism configuration | `DeterminismReport`, `configure_determinism` | — | r:determinism, u:utilities |
| `utilities/environment.py` | 193 | Environment capture for reproducibility records | `EnvironmentInfo`, `collect_environment` | — | u:utilities |
| `utilities/hashing.py` | 89 | Stable content hashing | `stable_hash_bytes`, `stable_hash`, `stable_json_hash` | — | u:utilities |
| `utilities/json_io.py` | 140 | JSON helpers with canonical encoding and bounded, validated reads | `canonical_json_dumps`, `write_json`, `read_json_bytes`, `read_json` | exceptions | u:utilities |
| `utilities/paths.py` | 147 | Filesystem path validation | `safe_filename`, `is_within`, `resolve_under_root`, `ensure_directory` | exceptions | u:utilities |
| `utilities/seeding.py` | 284 | Centralised seed management | `derive_seed`, `SeedBundle`, `seed_everything`, `torch_generator`, `dataloader_worker_init`, `rng_state_to_json` +2 | — | i:component_integration, r:determinism, u:models, u:utilities |
| `utilities/timing.py` | 89 | Timing helpers and timezone-aware timestamps | `utc_now`, `utc_now_iso`, `Stopwatch` | — | u:utilities |

### `nas_engine/observability/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `observability/__init__.py` | 27 | Structured logging, event vocabulary, and in-process counters | — | — | (via package) |
| `observability/context.py` | 118 | Ambient logging context carried by :mod:`contextvars` | `current_context`, `bind_context`, `search_context`, `candidate_context`, `worker_context` | — | u:observability |
| `observability/events.py` | 96 | The closed vocabulary of structured search events | `Event`, `emit` | — | r:event_vocabulary, u:observability |
| `observability/logging.py` | 202 | Structured logging configuration built on :mod:`structlog` | `REDACTED`, `redact_mapping`, `configure_logging`, `get_logger` | — | u:observability |
| `observability/metrics.py` | 204 | In-process counters and gauges | `DurationSummary`, `MetricsSnapshot`, `CounterRegistry` | — | u:observability |

### `nas_engine/architectures/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `architectures/__init__.py` | 74 | Architecture genotypes: specification, canonical form, hashing, shapes, and cost | — | — | (via package) |
| `architectures/canonical.py` | 148 | Canonical serialisation of architecture genotypes | `to_canonical_dict`, `to_canonical_json`, `from_canonical_dict`, `from_canonical_json`, `architectures_equal` | exceptions, utilities | i:worker_process, p:architecture_properties, r:golden_fixtures, u:architectures |
| `architectures/cost.py` | 311 | Analytic cost model: parameters, buffers, and multiply-accumulate operations | `normalization_parameters`, `conv_parameters`, `conv_macs`, `separable_hidden_channels`, `ArchitectureCost`, `compute_cost` | — | perf:performance_guards, p:architecture_properties, r:golden_fixtures, u:architectures |
| `architectures/hashing.py` | 83 | Architecture hashing: stable identity for candidate networks | `architecture_hash`, `short_hash` | utilities | f:failure_recovery, i:component_integration, i:worker_process, perf:performance_guards +9 |
| `architectures/lineage.py` | 187 | Architecture lineage reconstruction | `LineageNode`, `LineageChain`, `LineageGraph` | — | e:full_searches, u:architectures |
| `architectures/shapes.py` | 347 | Static tensor-shape inference for architecture genotypes | `make_divisible`, `conv_output_size`, `TensorShape`, `LayerShape`, `ShapeTrace`, `infer_shapes` | exceptions | i:component_integration, perf:performance_guards, p:architecture_properties, p:search_properties +2 |
| `architectures/spec.py` | 460 | The architecture genotype: a validated, immutable, canonicalising data model | `quantise`, `BlockSpec`, `StageSpec`, `StemSpec`, `HeadSpec`, `ArchitectureSpec` | — | f:failure_recovery, p:architecture_properties, r:golden_fixtures, u:architectures +7 |
| `architectures/summary.py` | 140 | Human-readable architecture summaries | `ArchitectureSummary`, `summarise` | — | u:architectures |
| `architectures/types.py` | 124 | Enumerations shared by search spaces, architecture specs, and model builders | `OperationType`, `NormalizationType`, `ActivationType`, `PoolingType` | — | p:architecture_properties, u:architectures, u:models, u:search_space |

### `nas_engine/search_space/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `search_space/__init__.py` | 64 | Search-space definition, sampling, repair, mutation, and validation | — | — | (via package) |
| `search_space/mutation.py` | 800 | Mutation operators for evolutionary architecture search | `MutationResult`, `mutate_operation`, `mutate_kernel_size`, `mutate_expansion_ratio`, `mutate_activation`, `mutate_normalization` +9 | architectures, exceptions, utilities | p:search_properties, r:determinism, r:documentation_claims, u:search_space |
| `search_space/presets.py` | 259 | Named, ready-to-use search spaces | `default_cnn_space`, `tiny_cnn_space`, `micro_cnn_space`, `get_preset` | architectures, exceptions | f:failure_recovery, i:worker_process, perf:performance_guards, p:search_properties +2 |
| `search_space/repair.py` | 196 | Structural repair of architecture genotypes | `RepairReport`, `stage_widths`, `repair_architecture` | architectures | p:search_properties, u:search_space |
| `search_space/sampler.py` | 437 | Seeded sampling of architectures from a search space | `SamplerStatistics`, `ArchitectureSampler` | architectures, exceptions, utilities | f:failure_recovery, i:component_integration, i:worker_process, perf:performance_guards +3 |
| `search_space/space.py` | 497 | Search-space definition: the set of architectures a search is allowed to consider | `BlockChoices`, `StemChoices`, `HeadChoices`, `SpaceConstraints`, `SearchSpace` | architectures, exceptions | p:search_properties, u:search_space, u:strategies |
| `search_space/validation.py` | 520 | Architecture validation: schema, semantics, membership, and constraints | `ValidationIssue`, `ValidationReport`, `check_membership`, `check_constraints`, `check_architecture`, `validate_architecture` | architectures, exceptions | perf:performance_guards, p:search_properties, u:search_space |

### `nas_engine/models/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `models/__init__.py` | 42 | PyTorch phenotypes: turning architecture genotypes into runnable networks | — | — | (via package) |
| `models/blocks.py` | 283 | Block modules: the phenotype of one :class:`~nas_engine.architectures.spec.BlockSpec` | `SeparableConvBlock`, `build_operation`, `NasBlock`, `ClassifierHead` | architectures, exceptions | u:models |
| `models/builder.py` | 354 | Model construction: genotype in, :class:`torch.nn.Module` out | `ModelSummary`, `NasNetwork`, `count_parameters`, `state_dict_bytes`, `summarize_model`, `ModelBuilder` +1 | architectures, exceptions | f:failure_recovery, i:component_integration, perf:performance_guards, p:architecture_properties +5 |
| `models/initialization.py` | 131 | Weight initialisation | `initialize_module`, `zero_initialize_residual_branches`, `initialize_weights` | — | u:models |
| `models/operations.py` | 220 | Primitive PyTorch layers used to realise a genotype | `group_count`, `build_activation`, `build_normalization`, `build_global_pool`, `build_conv_bn_act`, `GlobalPoolFlatten` | architectures, exceptions | u:models |

### `nas_engine/datasets/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `datasets/__init__.py` | 37 | Dataset providers, loaders, and low-fidelity views | — | — | (via package) |
| `datasets/base.py` | 137 | Dataset abstractions shared by every provider | `DatasetBundle`, `DatasetProvider` | exceptions | f:failure_recovery, i:component_integration, u:datasets_and_training, u:evaluation |
| `datasets/cifar10.py` | 193 | CIFAR-10 provider | `Cifar10Provider` | exceptions, utilities | u:public_api |
| `datasets/loaders.py` | 329 | DataLoader construction, including low-fidelity views of a dataset | `ResizedDataset`, `deterministic_subset`, `LoaderSettings`, `FidelityView`, `DataLoaders`, `build_dataloaders` | exceptions, utilities | f:failure_recovery, i:component_integration, i:worker_process, u:datasets_and_training +1 |
| `datasets/registry.py` | 102 | Dataset provider registry | `register_provider`, `available_providers`, `get_provider`, `build_dataset` | exceptions | i:component_integration, i:worker_process, u:datasets_and_training |
| `datasets/synthetic.py` | 210 | Deterministic synthetic image classification data | `SyntheticImageDataset`, `SyntheticDatasetProvider` | exceptions, utilities | u:datasets_and_training, u:public_api |

### `nas_engine/training/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `training/__init__.py` | 52 | Training: optimisers, schedules, metrics, early stopping, checkpoints, and the loop | — | — | (via package) |
| `training/checkpointing.py` | 221 | Training checkpoints: save, load, and version | `TrainingCheckpoint`, `save_checkpoint`, `load_checkpoint` | exceptions | f:failure_recovery, u:datasets_and_training |
| `training/early_stopping.py` | 154 | Early stopping | `MonitorMode`, `EarlyStopping` | exceptions | u:datasets_and_training |
| `training/metrics.py` | 171 | Metric computation and aggregation | `accuracy`, `topk_accuracy`, `MetricAggregator`, `EpochMetrics` | — | u:datasets_and_training |
| `training/optimizers.py` | 195 | Optimiser construction | `OptimizerType`, `OptimizerSettings`, `split_parameter_groups`, `build_optimizer` | exceptions | f:failure_recovery, i:component_integration, u:datasets_and_training, u:evaluation |
| `training/schedulers.py` | 191 | Learning-rate schedule construction | `SchedulerType`, `SchedulerSettings`, `build_scheduler` | exceptions | u:datasets_and_training |
| `training/trainer.py` | 802 | The training loop | `TrainingSettings`, `TrainingOutcome`, `Trainer`, `evaluation_mode` | datasets, exceptions, observability, utilities | f:failure_recovery, i:component_integration, i:worker_process, u:datasets_and_training +1 |

### `nas_engine/objectives/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `objectives/__init__.py` | 62 | Multi-objective comparison: objectives, constraints, scoring, Pareto fronts, ranking | — | — | (via package) |
| `objectives/constraints.py` | 164 | Hard constraints on candidate metrics | `ComparisonOperator`, `MetricConstraint`, `ConstraintSet` | exceptions | u:objectives |
| `objectives/objective.py` | 278 | Objective definitions | `ObjectiveDirection`, `NormalizationStrategy`, `Objective`, `ObjectiveSet`, `default_objectives` | exceptions | perf:performance_guards, p:search_properties, u:objectives |
| `objectives/online.py` | 106 | Online scalarisation: a single fitness value available *during* a search | `uses_stable_scalarization`, `online_objective_value` | — | u:objectives |
| `objectives/pareto.py` | 244 | Pareto dominance and front computation | `ObjectiveVector`, `dominates`, `to_objective_vector`, `pareto_front`, `non_dominated_sort`, `crowding_distance` | exceptions | perf:performance_guards, p:search_properties, r:golden_fixtures, u:objectives |
| `objectives/ranking.py` | 239 | Reproducible candidate ranking | `RankedCandidate`, `RankingResult`, `rank_candidates` | — | i:component_integration, perf:performance_guards, p:search_properties, u:objectives +2 |
| `objectives/scoring.py` | 281 | Weighted scalar scoring with population-relative normalisation | `NormalizerStats`, `compute_stats`, `normalize_value`, `ScoringResult`, `WeightedScorer` | exceptions | p:search_properties, u:objectives |

### `nas_engine/evaluation/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `evaluation/__init__.py` | 41 | Candidate evaluation: budgets, measurement, and results | — | — | (via package) |
| `evaluation/budget.py` | 147 | Training budgets: the resource allocation given to one evaluation | `TrainingBudget` | exceptions | f:failure_recovery, i:component_integration, i:worker_process, perf:performance_guards +6 |
| `evaluation/evaluator.py` | 599 | Candidate evaluation: the bridge between a genotype and a set of measured metrics | `EvaluationSettings`, `EvaluationContext`, `CandidateEvaluator` | architectures, datasets, exceptions, models, observability, training, utilities | f:failure_recovery, i:component_integration, i:worker_process, u:evaluation |
| `evaluation/latency.py` | 252 | Inference latency benchmarking | `LatencyMeasurement`, `measure_latency` | exceptions, training | r:determinism, u:evaluation |
| `evaluation/model_size.py` | 122 | Serialised model size measurement | `ModelSizeMeasurement`, `measure_model_size`, `save_model_weights` | — | u:evaluation |
| `evaluation/result.py` | 303 | Evaluation results and the failure taxonomy that classifies what went wrong | `FailureKind`, `classify_failure`, `EvaluationFailure`, `EvaluationResult` | exceptions, utilities | f:failure_recovery, i:worker_process, r:determinism, u:evaluation +2 |

### `nas_engine/search/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `search/__init__.py` | 38 | Search strategies and the interface every strategy implements | — | — | (via package) |
| `search/evolution.py` | 526 | Regularized evolution, also known as aging evolution | `PopulationMember`, `RegularizedEvolution` | architectures, evaluation, exceptions, observability, search_space, utilities | r:determinism, u:strategies |
| `search/random_search.py` | 237 | Random search | `RandomSearch` | architectures, evaluation, exceptions, observability, search_space | perf:performance_guards, p:search_properties, r:determinism, u:strategies |
| `search/registry.py` | 201 | Search-strategy registry | `register_strategy`, `available_strategies`, `build_strategy` | evaluation, exceptions, search_space | i:component_integration, u:strategies |
| `search/strategy.py` | 279 | The search-strategy contract | `Proposal`, `Observation`, `StrategyStatistics`, `SearchStrategy`, `serialize_spec`, `deserialize_spec` | architectures, evaluation | r:determinism, u:strategies |
| `search/successive_halving.py` | 608 | Successive halving: multi-fidelity resource allocation | `ResourceLadder`, `SuccessiveHalving` | architectures, evaluation, exceptions, observability, search_space, utilities | u:strategies |

### `nas_engine/persistence/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `persistence/__init__.py` | 54 | Persistence: database connection, versioned schema, and the repository seam | — | — | r:documentation_claims |
| `persistence/database.py` | 252 | Database connection management | `Database` | exceptions, observability | f:failure_recovery, i:component_integration, perf:performance_guards, u:persistence |
| `persistence/migrations.py` | 207 | Versioned schema management | `Migration`, `current_version`, `apply_migrations`, `ensure_schema` | exceptions, observability, utilities | perf:performance_guards, u:persistence |
| `persistence/models.py` | 469 | SQLAlchemy ORM models: the persisted shape of a search | `new_id`, `UTCDateTime`, `Base`, `SearchStatus`, `SearchRecord`, `CandidateRecord` +6 | utilities | e:full_searches, f:failure_recovery, u:persistence |
| `persistence/repository.py` | 1397 | The repository: the only place that talks SQL | `SearchSummary`, `CandidateSummary`, `RecoveryReport`, `SearchRepository` | architectures, evaluation, exceptions, observability, orchestration, utilities | i:component_integration, perf:performance_guards, u:persistence |

### `nas_engine/config/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `config/__init__.py` | 72 | Validated configuration: models, loading, and precedence | — | — | (via package) |
| `config/loader.py` | 436 | Configuration loading and the precedence chain | `deep_merge`, `parse_scalar`, `assign_path`, `parse_environment`, `parse_overrides`, `read_yaml` +4 | exceptions, observability | u:config |
| `config/models.py` | 939 | Validated configuration models | `ProjectConfig`, `DatasetConfig`, `SearchSpaceConfig`, `AlgorithmConfig`, `BudgetConfig`, `OptimizerConfig` +13 | architectures, datasets, evaluation, exceptions, objectives, search_space, training, utilities | e:full_searches, f:failure_recovery, i:component_integration, u:cli +1 |

### `nas_engine/orchestration/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `orchestration/__init__.py` | 50 | Orchestration: the candidate lifecycle, execution backends, retries, and the engine | — | — | i:worker_process |
| `orchestration/checkpoint.py` | 249 | Search checkpoints: what has to be saved for a resume to be correct | `EngineState`, `SearchCheckpoint` | exceptions, utilities | u:orchestration |
| `orchestration/engine.py` | 1105 | The search engine: the orchestrator that owns the candidate lifecycle | `SearchEngine` | architectures, config, datasets, evaluation, exceptions, models, objectives, observability, persistence, search, search_space, utilities | e:full_searches, f:failure_recovery, i:component_integration, r:determinism +1 |
| `orchestration/executors.py` | 417 | Execution backends for candidate evaluation | `EvaluationTask`, `EvaluationExecutor`, `SequentialExecutor`, `ProcessPoolExecutorBackend`, `build_executor` | architectures, evaluation, exceptions, observability, utilities | f:failure_recovery, i:worker_process, u:orchestration |
| `orchestration/lifecycle.py` | 269 | The candidate state machine | `CandidateState`, `TrialState`, `can_transition`, `validate_transition`, `CandidateStateMachine` | exceptions | e:full_searches, f:failure_recovery, i:component_integration, perf:performance_guards +4 |
| `orchestration/result.py` | 155 | The object a completed search returns | `StopReason`, `SearchResult` | objectives | e:full_searches, u:orchestration |
| `orchestration/retry.py` | 163 | Retry policy and error classification | `RetryDecision`, `RetryPolicy` | evaluation | f:failure_recovery, u:orchestration |
| `orchestration/worker.py` | 189 | The worker-process entry point for multiprocessing execution | `evaluate_task`, `reset_worker_cache` | architectures, config, datasets, evaluation, models, observability, utilities | i:worker_process |

### `nas_engine/reporting/`

| File | Lines | Purpose | Public symbols | Depends on | Tests |
| --- | ---: | --- | --- | --- | --- |
| `reporting/__init__.py` | 31 | Reports, exports, and figures generated from persisted search results | — | — | (via package) |
| `reporting/exporters.py` | 186 | CSV and JSON exports | `sanitize_cell`, `collect_metric_columns`, `export_candidates_csv`, `export_rows_csv`, `export_json` | exceptions, objectives, utilities | u:reporting |
| `reporting/plots.py` | 479 | Matplotlib figures for search reports | `SURFACE`, `INK_PRIMARY`, `INK_SECONDARY`, `GRID`, `AXIS`, `NEUTRAL_MARK` +10 | exceptions, objectives | u:reporting |
| `reporting/report.py` | 581 | Markdown search reports | `ReportArtifacts`, `ReportGenerator` | architectures, evaluation, exceptions, objectives, observability, orchestration, persistence, utilities | e:full_searches, i:component_integration, r:golden_fixtures, u:reporting |

<!-- END GENERATED SOURCE TABLES -->

---

## Tests: `tests/`

| File | Lines | Purpose |
| --- | ---: | --- |
| `conftest.py` | 256 | Shared fixtures: seeded sampler, tiny spaces, in-memory database, repository, config factory |
| `profiles.py` | 47 | Hypothesis profile scaling via `HYPOTHESIS_SCALE`, so nightly runs deepen without editing tests |
| `fixtures/architectures.json` | 155 | Golden architectures with their pinned hashes, parameter counts, MACs, and shapes |
| `fixtures/pareto_cases.json` | 125 | Hand-checked Pareto fronts, including ties and infeasible members |
| `fixtures/report_structure.json` | 36 | The sections a generated report must contain |
| `fixtures/state_transitions.json` | 34 | The pinned candidate state-transition table |
| `unit/test_architectures.py` | 532 | Spec validation, canonicalisation, hashing, shape inference, cost, summaries |
| `unit/test_search_space.py` | 652 | Space definition, sampling, repair, all twelve mutation operators, four validation layers |
| `unit/test_models.py` | 340 | Operations, blocks, initialisation, builder, exact parameter counting |
| `unit/test_datasets_and_training.py` | 753 | Providers, loaders, fidelity views, optimisers, schedules, metrics, early stopping, checkpoints, the loop |
| `unit/test_objectives.py` | 531 | Objectives, constraints, scoring, Pareto dominance, ranking, online scalarisation |
| `unit/test_evaluation.py` | 465 | Budgets, the evaluator, latency, model size, results, failure classification |
| `unit/test_strategies.py` | 571 | The strategy contract and all three strategies, including state round-trips |
| `unit/test_persistence.py` | 614 | Database, migrations, every repository method, recovery |
| `unit/test_config.py` | 459 | The four-layer precedence chain, deep merge, validation messages |
| `unit/test_orchestration.py` | 339 | State machine, retry policy, checkpoints, executors, results |
| `unit/test_observability.py` | 343 | Logging configuration, redaction, the event vocabulary, context, counters |
| `unit/test_reporting.py` | 236 | Exporters, plots, report generation |
| `unit/test_utilities.py` | 307 | Hashing, JSON, paths, seeding, determinism, timing, environment |
| `unit/test_cli.py` | 257 | Every command's arguments, exit codes, and error paths |
| `unit/test_public_api.py` | 223 | That `nas_engine.__all__` is complete, sorted, and importable |
| `property/test_architecture_properties.py` | 401 | Hashing, canonicalisation, shape, and cost invariants over generated architectures |
| `property/test_search_properties.py` | 451 | Sampling, mutation, validation, Pareto, and scoring invariants |
| `integration/test_component_integration.py` | 294 | Real components wired together without the full engine |
| `integration/test_worker_process.py` | 229 | The worker entry point, its caching, and that nothing escapes it |
| `end_to_end/test_full_searches.py` | 439 | Complete searches with all three strategies, resume, reports, and CLI sequences |
| `regression/test_determinism.py` | 273 | The ten reproducibility guarantees, and the four deliberate non-guarantees |
| `regression/test_golden_fixtures.py` | 182 | Pinned hashes, costs, shapes, transitions, Pareto answers, report structure |
| `regression/test_event_vocabulary.py` | 159 | That no module shadows an `Event` name, and that field names are consistent |
| `regression/test_documentation_claims.py` | 333 | That every test the docs cite exists, that the counts they state are true, and that no module is unreached |
| `performance/test_performance_guards.py` | 266 | That the analytic path stays orders of magnitude cheaper than building models |
| `failure_recovery/test_failure_recovery.py` | 643 | Every failure mode: training, build, resource, corruption, crash, retry exhaustion |

See [test strategy](testing/test-strategy.md) and [test matrix](testing/test-matrix.md).

---

## Configuration: `configs/`

| File                      | Lines | Purpose                                                                                           |
| ------------------------- | ----: | ------------------------------------------------------------------------------------------------- |
| `smoke_test.yaml`         |    93 | Two evaluations on synthetic data. Proves the pipeline in seconds; used by CI and by `make smoke` |
| `random_search.yaml`      |   131 | A realistic random search over CIFAR-10, fully commented                                          |
| `evolution.yaml`          |   106 | Regularized evolution with a population of 16 and multi-objective ranking                         |
| `successive_halving.yaml` |   112 | A three-rung ladder with η = 3                                                                    |

Every one is validated in CI by `nas-engine validate-config`, so a configuration cannot rot
against a schema change.

---

## Examples: `examples/`

| File                     | Lines | Demonstrates                                                                    |
| ------------------------ | ----: | ------------------------------------------------------------------------------- |
| `quickstart.py`          |   134 | The shortest complete search through the Python API                             |
| `custom_search_space.py` |   134 | Defining a space from scratch, and refining a preset                            |
| `custom_objective.py`    |   189 | Registering an objective and reading the Pareto front                           |
| `resume_search.py`       |   131 | Interrupting a search and resuming it, with the state compared before and after |

All four run to completion on CPU with no network access, and `make examples` runs them.

---

## Scripts: `scripts/`

| File                   | Lines | Purpose                                                                                         |
| ---------------------- | ----: | ----------------------------------------------------------------------------------------------- |
| `run_smoke_search.sh`  |    67 | End-to-end shell smoke test: search, report, export, inspect. The container's `smoke` command   |
| `benchmark.py`         |   187 | Times sampling, hashing, validation, cost, and model construction. Backs the performance guards |
| `generate_report.py`   |   100 | Regenerates a report for an existing search without re-running it                               |
| `check_coverage.py`    |    78 | Enforces the line and branch thresholds from `coverage.xml`                                     |
| `check_docs_links.py`  |   175 | Verifies every relative link *and anchor* in the documentation                                  |
| `check_tables.py`      |   402 | Checks and reformats every Markdown table; `--fix` aligns them in place                         |
| `generate_manifest.py` |   293 | Regenerates this page's source tables from the AST; `--check` fails when stale                  |

The checkers exist as files rather than Makefile heredocs because a multi-line Python
program inside a recipe is fragile — shell quoting mangles it, and it cannot be tested or
linted. These are linted and type-checked with everything else, and
`tests/regression/test_documentation_claims.py` imports `check_tables` directly so its
detection logic is itself tested.

`check_tables.py` is the answer to a real failure: a table whose source pipes do not line
up renders correctly but is unreadable in an editor or a diff, and the genuinely broken
cases — a row with the wrong cell count, a column with no header — hide in exactly the same
place. `make docs-fix` reformats; `make docs` fails if anything is left.

---

## Packaging and CI

| File | Lines | Purpose |
| --- | ---: | --- |
| `pyproject.toml` | 227 | Metadata, dependencies, the console script, and the configuration for Ruff, mypy, pytest, and coverage |
| `Makefile` | 174 | 30 targets. `make help` lists them; `make check` runs the gate CI runs |
| `Dockerfile` | 115 | Two-stage build, CPU-only by default, non-root uid 1000, with a healthcheck |
| `docker-entrypoint.sh` | 32 | Dispatches `smoke`, `bash`, `python`, or any `nas-engine` subcommand |
| `docker-compose.yml` | 96 | `doctor`, `smoke`, `search`, and `shell` services with resource caps |
| `.github/workflows/ci.yml` | 214 | Lint, type-check, test on 3.10/3.11/3.12, coverage gate, config validation, examples, Docker build, package verification |
| `.github/workflows/nightly.yml` | 157 | Deeper property testing (`HYPOTHESIS_SCALE=8`), longer searches, multiprocessing |
| `.pre-commit-config.yaml` | 107 | Ruff, ruff-format, mypy, and the docs link check on commit |
| `.dockerignore` | 56 | Keeps the build context small and secrets out of it |
| `.gitignore` | 63 | Excludes artifacts, coverage output, and caches |
| `LICENSE` | 21 | MIT |
| `artifacts/.gitkeep` | 0 | Preserves the default output directory in a fresh clone |

---

## Documentation: `docs/`

| Area            | Pages | Covers                                                                                                          |
| --------------- | ----: | --------------------------------------------------------------------------------------------------------------- |
| Root            |     5 | Index, getting started, glossary, this manifest, the traceability matrix                                        |
| `concepts/`     |    10 | NAS foundations, search spaces, encoding, the three algorithms, objectives, training, reproducibility, pitfalls |
| `architecture/` |     6 | System overview, component design, data flow, persistence, concurrency, security                                |
| `guides/`       |     8 | Running, resuming, defining spaces, adding strategies, datasets, objectives, reports, troubleshooting           |
| `testing/`      |     3 | Test strategy, the test matrix, reproducibility tests                                                           |
| `operations/`   |     4 | Deployment, observability, backup and recovery, the production runbook                                          |
| `adr/`          |     4 | Representation, persistence, the strategy interface, concurrency                                                |

`make docs` checks that every relative link and heading anchor in all of them resolves.

## See also

- [Index](index.md)
- [Traceability matrix](traceability-matrix.md) — requirement to implementation to test.
- [Component design](architecture/component-design.md) — why the layering is what it is.
