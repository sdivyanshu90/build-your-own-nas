# Traceability matrix

Every requirement, where it is implemented, and how it is verified.

The point of this page is to make coverage *checkable* rather than asserted. Where something
is only partly done, or deliberately not done, that is stated in the row rather than left
for a reader to discover.

**Status key**

| Status | Meaning                                                               |
| ------ | --------------------------------------------------------------------- |
| ✅     | Implemented and verified by an automated test                         |
| ⚠️     | Implemented, but verification is manual or partial — the row says why |
| 📄     | Deliberately documented rather than implemented — the row says why    |

Test names are given unqualified; find them with `pytest -k <name>`.

---

## 1. Search-space definition and validation

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 1.1 | Define a search space declaratively | `search_space/space.py` — `SearchSpace`, `BlockChoices`, `StemChoices`, `HeadChoices` | `u:search_space` | ✅ |
| 1.2 | Named presets, refinable rather than replaceable | `search_space/presets.py` — `default_cnn`, `tiny_cnn`, `micro_cnn`; `SearchSpaceConfig.overrides` deep-merges | `test_valid_search_space_override_is_applied` | ✅ |
| 1.3 | Hard constraints on every candidate | `SpaceConstraints` — parameters, MACs, total stride, final resolution, depth | `test_constraint_violation_is_distinguished_from_invalidity` | ✅ |
| 1.4 | Reject an empty or contradictory space at construction | `SpaceConstraints._validate`, `SearchSpace.require_non_empty` | `test_impossible_parameter_ceiling_is_reported_early` | ✅ |
| 1.5 | Report the space's size | `SearchSpace.cardinality_upper_bound`, `log10_cardinality` | `test_cardinality_grows_with_the_choice_sets`, `test_log_cardinality_matches_the_bound` | ✅ |
| 1.6 | Validate an architecture against the space | `search_space/validation.py` — four layers: schema, semantics, membership, constraints | `u:search_space`, `p:search_properties` | ✅ |
| 1.7 | Distinguish *invalid* from *infeasible* | `ValidationReport.only_constraint_violations` drives `PRUNED` vs `FAILED` | `test_a_pruned_candidate_is_distinguished_from_a_failure` | ✅ |
| 1.8 | Static shape inference, no model construction | `architectures/shapes.py` — `infer_shapes`, `conv_output_size` | `test_static_shapes_match_what_pytorch_produces` (property) | ✅ |
| 1.9 | Analytic cost, exact for parameters | `architectures/cost.py` — `compute_cost` | `test_analytic_cost_equals_the_measured_parameter_count` (property) | ✅ |
| 1.10 | Repair rather than discard where possible | `search_space/repair.py` — `repair_architecture` | `test_repair_is_idempotent`, `test_repairing_a_valid_architecture_changes_nothing` (property) | ✅ |

## 2. Architecture encoding

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 2.1 | Deterministic encoding | `architectures/spec.py` — frozen Pydantic models, closed enums | `p:architecture_properties` | ✅ |
| 2.2 | Serialisable to and from JSON | `architectures/canonical.py` | `test_json_round_trip_is_lossless` (property) | ✅ |
| 2.3 | Content-addressed identity | `architectures/hashing.py` — BLAKE2b/128 over canonical JSON | `test_equal_architectures_hash_identically` | ✅ |
| 2.4 | Inactive fields must not affect identity | `to_canonical_dict` resets them; canonicalisation is idempotent | `test_inactive_fields_hold_their_sentinel_values` | ✅ |
| 2.5 | Immutable, with controlled modification | `frozen=True`; `evolve()` re-runs the constructor | `test_blocks_are_immutable`, `test_evolve_reapplies_canonicalisation` | ✅ |
| 2.6 | Imported JSON treated as untrusted | `from_canonical_dict` — `extra="forbid"`, enum and range validation, no execution | `test_rejects_unknown_fields`, `test_rejects_unknown_operations`, `test_rejects_non_object_payloads` | ✅ |
| 2.7 | Encoding must not drift silently | `ARCHITECTURE_SCHEMA_VERSION` + golden fixtures | `test_hashes_are_unchanged`, `test_canonical_form_round_trips_to_the_same_bytes` | ✅ |
| 2.8 | Human-readable summary | `architectures/summary.py` — `compact()`, `to_text()` | `test_compact_line_mentions_the_key_facts` | ✅ |
| 2.9 | Lineage reconstruction | `architectures/lineage.py`, persisted `parent_id` and `mutation` | `test_lineage_is_recorded` (e2e) | ✅ |

**Decision:** [ADR 0001](adr/0001-search-space-representation.md).

## 3. Model generation

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 3.1 | Genotype to executable PyTorch model | `models/builder.py` — `ModelBuilder.build` → `NasNetwork` | `u:models` | ✅ |
| 3.2 | Output shape matches the class count | `ClassifierHead` | `test_output_shape_matches_the_class_count` (property) | ✅ |
| 3.3 | Forward pass produces finite logits | `models/blocks.py`, `models/initialization.py` | `test_forward_pass_produces_finite_logits` (property) | ✅ |
| 3.4 | Exact parameter counting | `count_parameters` separates parameters from buffers | `test_parameter_counting_separates_buffers` | ✅ |
| 3.5 | Deterministic initialisation for a seed | `models/initialization.py` + `seed_everything` | `test_weights_are_reproducible_for_a_given_seed` | ✅ |
| 3.6 | Build failures are actionable, not `KeyError` | `ModelBuildError` with the offending block | `test_error_names_the_offending_block` | ✅ |

## 4. Training and evaluation

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 4.1 | Train a candidate under a budget | `training/trainer.py` — `Trainer.fit` | `u:datasets_and_training` | ✅ |
| 4.2 | Optimisers and schedules | `training/optimizers.py`, `training/schedulers.py` | `test_warmup_ramps_from_a_non_zero_value`, `test_cosine_anneals_towards_the_floor` | ✅ |
| 4.3 | Early stopping | `training/early_stopping.py` | `test_early_stopping_ends_the_run` | ✅ |
| 4.4 | Resumable per-candidate training | `training/checkpointing.py` | `test_resume_continues_from_the_checkpoint` | ✅ |
| 4.5 | A corrupt checkpoint is rejected, not loaded | `load_checkpoint` validates version and contents | `test_a_corrupt_training_checkpoint_is_rejected`, `test_a_truncated_checkpoint_is_rejected` | ✅ |
| 4.6 | Metrics: accuracy, top-k, loss | `training/metrics.py` | `test_topk_is_clamped_to_the_class_count`, `test_topk_is_at_least_top1` | ✅ |
| 4.7 | Multi-fidelity budgets: epochs, data fraction, resolution | `evaluation/budget.py`, `datasets/loaders.py` — `FidelityView` | `test_subsets_are_deterministic`, `test_resolution_fidelity_resizes_every_split` | ✅ |
| 4.8 | A per-candidate wall-clock guard | `budget.max_seconds`, enforced in the trainer | `test_timeout_is_enforced` | ✅ |
| 4.9 | Non-finite loss is detected and classified | `NonFiniteLossError` → `FailureKind.DIVERGENCE` | `test_divergence_is_a_permanent_failure` | ✅ |
| 4.10 | Test split used only at the end | `DatasetBundle.test`; `include_test=False` by default; only `nas-engine evaluate` reads it | `test_the_winner_can_be_scored_on_the_held_out_test_split` | ✅ |

## 5. Search strategies

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 5.1 | A common strategy interface | `search/strategy.py` — `SearchStrategy` | `TestStrategyContract` | ✅ |
| 5.2 | **Random search** | `search/random_search.py` | `TestRandomSearch` (11 tests) | ✅ |
| 5.2a | — uniform sampling from the space | `ArchitectureSampler` | `test_every_sample_is_a_valid_member` (property) | ✅ |
| 5.2b | — duplicate avoidance | `on_duplicate` bookkeeping + engine-level dedup | `test_proposals_are_unique` | ✅ |
| 5.2c | — stops when the space is exhausted | `SearchExhaustedError` handling | `test_exhausted_space_stops_the_search` | ✅ |
| 5.3 | **Regularized (aging) evolution** | `search/evolution.py` | `TestRegularizedEvolution` (16 tests) | ✅ |
| 5.3a | — tournament selection | `_select_parent` | `test_tournament_prefers_fitter_parents` | ✅ |
| 5.3b | — mutation | `search_space/mutation.py`, 12 operators | `u:search_space`, `p:search_properties` | ✅ |
| 5.3c | — aging removes the **oldest** | `deque(maxlen=population_size)` | `test_aging_removes_the_oldest_not_the_worst` | ✅ |
| 5.3d | — failures never enter the population | `observe` filters unsuccessful results | `test_failed_candidates_never_enter_the_population` | ✅ |
| 5.3e | — population survives a resume | `state_dict` serialises every member | `test_state_round_trip_preserves_the_population` | ✅ |
| 5.4 | **Successive halving** | `search/successive_halving.py` | `TestSuccessiveHalving` (11 tests), `TestResourceLadder` (8) | ✅ |
| 5.4a | — geometric resource ladder | `ResourceLadder` | `test_epochs_grow_geometrically`, `test_rung_sizes_shrink_geometrically` | ✅ |
| 5.4b | — equal cost per rung | `ResourceLadder` sizing | `test_each_rung_costs_about_the_same` | ✅ |
| 5.4c | — promotion waits for the whole rung | `_propose_for_rung` reads the *previous* rung's promotions | `test_promotion_waits_for_the_whole_rung` | ✅ |
| 5.4d | — rung is part of candidate identity | `UNIQUE(search_id, architecture_hash, rung)` | `test_the_same_architecture_is_re_evaluated_at_a_higher_rung` | ✅ |
| 5.4e | — a rung of pure failures ends the bracket | bracket termination | `test_all_failures_end_the_bracket` | ✅ |
| 5.5 | Strategies are registered explicitly, not auto-discovered | `search/registry.py` — `register_strategy` | `test_custom_strategies_can_be_registered` | ✅ |
| 5.6 | **Bayesian optimisation** — integration point | [Adding a search strategy](guides/adding-a-search-strategy.md) documents the surrogate/acquisition seam | — | 📄 |
| 5.7 | **RL-NAS** — integration point | Same guide: the controller maps onto `propose`/`observe` | — | 📄 |
| 5.8 | **DARTS** — integration point | Same guide, and [ADR 0003](adr/0003-search-strategy-interface.md): needs a different *evaluation* path, not another strategy | — | 📄 |

Rows 5.6–5.8 are documented rather than implemented **because the specification asked for
exactly that.** ADR 0003 is explicit that DARTS does not fit the interface, rather than
implying a drop-in would work.

**Decision:** [ADR 0003](adr/0003-search-strategy-interface.md).

## 6. Multi-objective optimisation

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 6.1 | Maximise validation accuracy by default | `objectives/objective.py` — `default_objectives` | `test_default_objective_set_is_valid` | ✅ |
| 6.2 | Secondary objectives: parameters, latency, size | Any metric name may be an objective | `u:objectives` | ✅ |
| 6.3 | Pareto dominance and front | `objectives/pareto.py` — `dominates`, `pareto_front` | `test_fronts_match_the_hand_checked_answers` (golden) | ✅ |
| 6.4 | Non-dominated sorting | `non_dominated_sort` | `test_non_dominated_sort_partitions_the_population` (property) | ✅ |
| 6.5 | Crowding distance for tie-breaking | `crowding_distance` | `test_crowding_distance_favours_the_extremes` | ✅ |
| 6.6 | Weighted scalarisation with normalisation | `objectives/scoring.py` — minmax, log, none | `u:objectives` | ✅ |
| 6.7 | Hard constraints separate from objectives | `objectives/constraints.py` — `ConstraintSet` | `test_infeasible_candidates_rank_last` | ✅ |
| 6.8 | A time-stable score for feeding strategies | `objectives/online.py` — `online_objective_value` | `test_value_is_stable_regardless_of_population` | ✅ |
| 6.9 | Ranking is reproducible | `rank_candidates` sorts with explicit tie-breaks | `test_ranking_is_deterministic_under_reordering` | ✅ |

## 7. Orchestration, checkpointing, and recovery

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 7.1 | A validated candidate state machine | `orchestration/lifecycle.py` — `ALLOWED_TRANSITIONS` | `test_the_transition_table_is_unchanged` (golden) | ✅ |
| 7.2 | Illegal transitions raise | `validate_transition` | `test_leaving_a_terminal_state_is_forbidden` | ✅ |
| 7.3 | Checkpoint search state periodically | `orchestration/checkpoint.py`, `persistence.checkpoint_every` | `test_checkpoints_are_append_only` | ✅ |
| 7.4 | Resume continues rather than restarting | `SearchEngine.resume`, strategy `load_state_dict` | `test_resume_reaches_the_same_state_as_an_uninterrupted_run` | ✅ |
| 7.5 | Recover candidates left mid-evaluation | recovery sweep requeues `RUNNING` | `test_a_running_candidate_is_requeued_on_resume`, `test_running_candidates_are_requeued` | ✅ |
| 7.6 | Reconcile a stale checkpoint against the database | `_reconcile_completed` — the database wins | `test_a_crashed_evaluation_is_recovered_on_resume` | ✅ |
| 7.7 | Retry policy with classification and backoff | `orchestration/retry.py` | `TestRetryPolicy` (10 tests) | ✅ |
| 7.8 | Permanent failures are never retried | `EvaluationFailure.retriable` | `test_permanent_failures_are_never_retried` | ✅ |
| 7.9 | Retry exhaustion fails the candidate, not the search | `_finish_failed` | `test_retry_exhaustion_fails_the_candidate` | ✅ |
| 7.10 | A corrupt checkpoint is refused | version and field validation | `test_a_corrupt_search_checkpoint_is_rejected`, `test_rejects_a_future_format` | ✅ |
| 7.11 | Checkpoint retention is bounded | `persistence.keep_checkpoints` | `test_pruning_keeps_the_newest` | ✅ |
| 7.12 | Ctrl-C stops cleanly and checkpoints | `KeyboardInterrupt` → `StopReason.INTERRUPTED` | `test_every_stop_reason_is_described` | ✅ |

## 8. Persistence

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 8.1 | SQLite storage | `persistence/database.py` | `u:persistence` | ✅ |
| 8.2 | SQLAlchemy ORM models | `persistence/models.py` — 8 tables | `u:persistence` | ✅ |
| 8.3 | **No string-interpolated SQL** | Every query goes through the ORM or bound parameters | Enforced by `ruff` rule `S608` across `src/` | ✅ |
| 8.4 | Versioned schema with migrations | `persistence/migrations.py` | `test_ensure_schema_is_idempotent` | ✅ |
| 8.5 | A newer database is refused | `SchemaVersionError` | `test_newer_database_is_refused` | ✅ |
| 8.6 | Repository seam; no ORM objects escape | `persistence/repository.py` returns frozen dataclasses | `u:persistence` | ✅ |
| 8.7 | One method, one transaction | `Database.session()` context manager | `test_transactions_roll_back_on_error`, `test_a_write_failure_rolls_the_transaction_back` | ✅ |
| 8.8 | Concurrent-read safety | `journal_mode=WAL`, `busy_timeout` | `test_in_memory_database_shares_one_connection` | ✅ |
| 8.9 | Referential integrity | `foreign_keys=ON` + cascades | `test_foreign_keys_are_enforced`, `test_deleting_a_search_cascades` | ✅ |
| 8.10 | Timezone-aware timestamps everywhere | `UTCDateTime` type decorator, `utc_now()` | `test_timestamps_are_timezone_aware` | ✅ |
| 8.11 | Stored specs re-validated on read | `get_candidate_spec` → `from_canonical_dict` | `test_specification_is_revalidated_on_read`, `test_corrupt_specification_is_rejected_on_read` | ✅ |
| 8.12 | Full lineage recorded | `parent_id`, `mutation`, `origin`, `generation` | `test_lineage_nodes_carry_parents_and_mutations` | ✅ |

**Decision:** [ADR 0002](adr/0002-persistence-layer.md).

## 9. Concurrency

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 9.1 | Local single-process by default | `SequentialExecutor`, `concurrency.mode: sequential` | `test_sequential_mode_builds_the_inline_backend` | ✅ |
| 9.2 | Optional multiprocessing | `ProcessPoolExecutorBackend` | `test_multiprocessing_produces_the_same_kind_of_result` (e2e) | ✅ |
| 9.3 | `spawn`, for PyTorch safety | `concurrency.start_method: spawn` | `test_multiprocessing_mode_builds_the_pool_backend` | ✅ |
| 9.4 | Picklable plain-data payloads | `EvaluationTask.to_payload` | `test_the_returned_payload_is_plain_data` | ✅ |
| 9.5 | Nothing escapes a worker | `_worker_entrypoint` catches everything | `test_nothing_escapes_the_worker` | ✅ |
| 9.6 | A dead worker costs one candidate | future exception → retriable `WorkerError` | `test_a_dead_worker_becomes_a_retriable_failure` | ✅ |
| 9.7 | **Bounded queues / in-flight work** | `concurrency.max_in_flight` caps dispatch | `test_worker_count_is_validated` | ✅ |
| 9.8 | **Configurable resource limits** | Model size, wall clock, workers, and threads — see the table below | `test_resource_limit_is_reported_before_building`, `test_timeout_is_enforced` | ✅ |
| 9.9 | Per-process evaluator caching | `orchestration/worker.py` | `test_the_evaluator_is_cached_per_configuration` | ✅ |

Requirement 9.8 spans several settings, so they are listed here rather than crammed into
one cell:

| Limit                    | Setting                             | Bounds                                        |
| ------------------------ | ----------------------------------- | --------------------------------------------- |
| Model size               | `evaluation.max_parameters`         | Trainable parameters, checked before building |
| Per-candidate time       | `budget.max_seconds_per_evaluation` | One evaluation's wall clock                   |
| Whole-search time        | `budget.max_seconds`                | The run's wall clock                          |
| In-flight work           | `concurrency.max_in_flight`         | Tasks dispatched at once                      |
| Parallelism              | `concurrency.workers`               | Worker processes                              |
| Thread fan-out           | `hardware.torch_threads`            | Intra-op threads per process                  |
| Container CPU and memory | `docker-compose.yml`                | Enforced by the runtime, not the process      |

**Decision:** [ADR 0004](adr/0004-concurrency-model.md).

## 10. Reproducibility

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 10.1 | Default seed 42 | `ReproducibilityConfig.seed = 42` | `test_defaults_apply_with_no_sources` | ✅ |
| 10.2 | Derived per-component seeds, not one global RNG | `utilities/seeding.py` — `derive_seed`, `SeedBundle` | `test_derivation_is_deterministic`, `test_labels_produce_independent_streams` | ✅ |
| 10.3 | Per-candidate seeding by content | `CandidateEvaluator.candidate_seed` from the architecture hash | `test_results_are_reproducible_across_calls` | ✅ |
| 10.4 | Torch determinism configured and *reported* | `utilities/determinism.py` — `DeterminismReport` with warnings | `test_the_determinism_report_lists_its_caveats` | ✅ |
| 10.5 | Environment captured with every search | `utilities/environment.py` → `searches.environment_json` | `test_configuration_and_environment_round_trip` | ✅ |
| 10.6 | Identical sequential runs agree exactly | — | `test_two_identical_sequential_searches_agree_exactly` | ✅ |
| 10.7 | Resume equals uninterrupted | — | `test_resume_reaches_the_same_state_as_an_uninterrupted_run` | ✅ |
| 10.8 | Latency is *not* claimed reproducible | documented and asserted as a non-goal | `test_latency_is_not_asserted_to_be_reproducible` | ✅ |

Cross-machine, cross-library-version, and multiprocessing whole-search identity are
**deliberately not guaranteed**, with reasons, in
[reproducibility tests](testing/reproducibility-tests.md).

## 11. Reporting

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 11.1 | Markdown comparison report | `reporting/report.py` — `ReportGenerator` | `test_a_generated_report_contains_every_required_section` (golden) | ✅ |
| 11.2 | CSV export | `reporting/exporters.py` — `export_candidates_csv` | `test_csv_export_lists_every_candidate` (e2e) | ✅ |
| 11.3 | JSON export | `export_json` | `test_json_export_is_self_describing` (e2e) | ✅ |
| 11.4 | Plots | `reporting/plots.py` — accuracy/parameters with the front, accuracy/latency, progress, population | `u:reporting` | ✅ |
| 11.5 | CSV injection is neutralised | `sanitize_cell` prefixes formula-leading cells | `test_formula_prefixes_are_neutralised` | ✅ |
| 11.6 | Deterministic output filenames | derived from the search id | `test_report_names_are_deterministic` (e2e) | ✅ |
| 11.7 | The report records the environment and determinism caveats | environment table in `ReportGenerator` | `test_a_generated_report_contains_every_required_section` | ✅ |
| 11.8 | Colour choices are accessible | palette validated for CVD separation and contrast | ⚠️ verified once with the palette validator; not re-checked in CI | ⚠️ |

Row 11.8 is the honest one: the palette passed an automated check (all-pairs CVD ΔE 9.2,
normal-vision 24.0) when it was chosen, but nothing re-runs that check on every commit. The
constants are pinned in `reporting/plots.py`, so a change is at least visible in review.

## 12. Interfaces

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 12.1 | A Python API | `nas_engine/__init__.py` — 51 exported symbols | `u:public_api` | ✅ |
| 12.2 | `__all__` complete, sorted, importable | — | `test_every_exported_name_exists`, `test_exports_are_sorted` | ✅ |
| 12.3 | A CLI | `cli.py` — 13 commands via Typer | `u:cli` (all commands), `e:full_searches` | ✅ |
| 12.4 | Meaningful exit codes | `ExitCode` IntEnum | `test_codes_are_distinct`, `test_interrupt_uses_the_conventional_code` | ✅ |
| 12.5 | `--json` on every query command | `_emit` | `test_json_output_is_parseable_for_every_query_command` (e2e) | ✅ |
| 12.6 | The documented command sequence works | — | `test_the_documented_command_sequence_works` (e2e) | ✅ |
| 12.7 | A diagnostic command | `nas-engine doctor` | `test_reports_every_check` | ✅ |

## 13. Configuration

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 13.1 | YAML configuration | `config/loader.py` — `read_yaml` uses `safe_load` | `u:config` | ✅ |
| 13.2 | Pydantic validation | `config/models.py` — 19 models | `u:config` | ✅ |
| 13.3 | Four-layer precedence: defaults < YAML < env < CLI | `load_config` | `test_file_overrides_defaults`, `test_environment_overrides_the_file`, `test_command_line_overrides_everything` | ✅ |
| 13.4 | Deep merge, not replacement | `deep_merge` | `test_deep_merge_preserves_siblings`, `test_overrides_do_not_erase_sibling_fields` | ✅ |
| 13.5 | Errors name the field, the constraint, and the value | `_format_validation_error` | `test_error_names_the_field_and_value` | ✅ |
| 13.6 | A configuration hash, stored with the search | `SearchConfig.config_hash` | `test_configuration_and_environment_round_trip` | ✅ |
| 13.7 | Configuration change across a resume warns, mismatched strategy blocks | `check_config_compatibility` | `test_a_changed_configuration_warns_rather_than_blocking`, `test_strategy_mismatch_is_fatal` | ✅ |
| 13.8 | Every shipped config validates | — | CI runs `validate-config` over `configs/*.yaml` | ✅ |

## 14. Security

Each row here restates a constraint from the specification verbatim.

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 14.1 | *"Do not execute arbitrary Python from configuration files"* | `yaml.safe_load`; strategies and providers resolve through registries of pre-registered names, never by import path | `test_unknown_strategy_is_reported`, `test_python_object_tags_are_refused` | ✅ |
| 14.2 | *"Do not use `eval`"* | No `eval`/`exec` anywhere | Ruff `S307` + `PGH001`; grep-clean | ✅ |
| 14.3 | *"Do not construct SQL with string interpolation"* | ORM and bound parameters only | Ruff `S608` | ✅ |
| 14.4 | *"Treat imported architecture JSON as untrusted input"* | `from_canonical_dict` — forbid-extra, closed enums, range checks | `test_rejects_unknown_fields`, `test_rejects_malformed_documents` | ✅ |
| 14.5 | *"Validate all paths before writing artifacts"* | `utilities/paths.py` — `resolve_under_root`, `safe_filename`, `UnsafePathError` | `test_rejects_traversal`, `test_rejects_absolute_components`, `test_detects_sibling_directories` | ✅ |
| 14.6 | *"Use bounded queues and configurable resource limits"* | `max_in_flight`; parameter, time, worker, and thread limits; bounded JSON reads | `u:orchestration`, `test_rejects_oversized_payload`, `test_rejects_oversized_file` | ✅ |
| 14.7 | *"Redact known sensitive configuration fields from logs"* | `_redaction_processor` — 8 substrings, case-insensitive, recursive, depth-capped | `test_sensitive_fields_are_redacted_in_events` | ✅ |
| 14.8 | *"Do not use unsafe deserialization for untrusted data"* | No `pickle` for data; `torch.load(weights_only=True)` for checkpoints | `test_a_corrupt_training_checkpoint_is_rejected` | ✅ |
| 14.9 | *"Clearly document the trust boundary"* | [security](architecture/security.md) | — | 📄 |
| 14.10 | CSV formula injection | `sanitize_cell` | `test_formula_prefixes_are_neutralised` | ✅ |

## 15. Testing

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 15.1 | Unit tests | `tests/unit/` | **867** tests | ✅ |
| 15.2 | Integration tests | `tests/integration/` | **36** | ✅ |
| 15.3 | End-to-end tests | `tests/end_to_end/` | **20** | ✅ |
| 15.4 | Property-based tests | `tests/property/` (Hypothesis) | **52** | ✅ |
| 15.5 | Regression tests | `tests/regression/` | **71** | ✅ |
| 15.6 | Failure-recovery tests | `tests/failure_recovery/` | **29** | ✅ |
| 15.7 | Performance guards | `tests/performance/` | **14** | ✅ |
| 15.8 | ≥ 90% line coverage | — | **93.70%**, gated by `scripts/check_coverage.py` | ✅ |
| 15.9 | ≥ 85% branch coverage | — | **86.65%**, gated | ✅ |
| 15.10 | No internet access required | Synthetic dataset by default; CIFAR-10 tests skip without a local copy | Whole suite runs offline | ✅ |
| 15.11 | No CIFAR-10 download required | — | `test_cifar10_never_downloads_without_permission` | ✅ |
| 15.12 | No GPU required | CPU-only by default; CUDA paths skipped | Suite passes on CPU-only CI | ✅ |
| 15.13 | No external tracking services, cloud credentials, or paid APIs | No such dependency exists | — | ✅ |

**Total: 1 089 tests** (1 086 passed, 1 skipped, 2 deselected on this machine; the skip is a
CUDA-absence path that cannot run *because* this host has CUDA).

## 16. Code quality

| # | Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- | :---: |
| 16.1 | Fully typed | Annotations throughout | `mypy --strict`, clean over 134 files | ✅ |
| 16.2 | Ruff clean | `pyproject.toml` rule set incl. `S` (bandit), `D` (docstrings) | `make lint` | ✅ |
| 16.3 | Formatted | `ruff format` | `make format-check` | ✅ |
| 16.4 | No placeholders, `pass` stubs, or TODOs | — | grep-clean; every abstract method has a real implementation | ✅ |
| 16.5 | No broad exception swallowing | Every `except` narrows, re-raises, or classifies. `KeyboardInterrupt`/`SystemExit` always propagate | Ruff `BLE`, `S110` | ✅ |
| 16.6 | Timezone-aware timestamps | `utc_now()`; no naive `datetime.now()` | Ruff `DTZ` | ✅ |
| 16.7 | Dependency injection | Engine, evaluator, and repository take collaborators as arguments | Tests substitute stubs throughout | ✅ |
| 16.8 | Actionable errors | `NasEngineError` with `code`, `details`, `retriable`; messages state the fix | `test_error_lists_the_legal_targets`, and similar | ✅ |
| 16.9 | Docstrings on every public symbol | Google style | Ruff `D` rules | ✅ |

## 17. Packaging and operations

| #    | Requirement         | Implementation                                             | Verification                                          | Status |
| ---- | ------------------- | ---------------------------------------------------------- | ----------------------------------------------------- | :----: |
| 17.1 | `pyproject.toml`    | —                                                          | `make build`, `make verify-package`                   |   ✅   |
| 17.2 | Console entry point | `nas-engine = "nas_engine.cli:main"`                       | `e:full_searches`                                     |   ✅   |
| 17.3 | Docker              | `Dockerfile` — two stages, non-root, CPU-only, healthcheck | ⚠️ built in CI; **not built in this environment**     |   ⚠️   |
| 17.4 | GitHub Actions      | `ci.yml`, `nightly.yml`                                    | ⚠️ syntax reviewed; **not executed here**             |   ⚠️   |
| 17.5 | Makefile            | 31 targets                                                 | `make help`, and every target used during development |   ✅   |
| 17.6 | Pre-commit hooks    | `.pre-commit-config.yaml`                                  | ⚠️ configured; not installed in this environment      |   ⚠️   |

## 18. Documentation

| #     | Requirement                    | Implementation                                               | Verification                  | Status |
| ----- | ------------------------------ | ------------------------------------------------------------ | ----------------------------- | :----: |
| 18.1  | Conceptual documentation       | `docs/concepts/` — 10 pages                                  | `make docs`                   |   ✅   |
| 18.2  | Architecture documentation     | `docs/architecture/` — 6 pages                               | `make docs`                   |   ✅   |
| 18.3  | Task guides                    | `docs/guides/` — 8 pages                                     | `make docs`                   |   ✅   |
| 18.4  | Testing documentation          | `docs/testing/` — 3 pages                                    | `make docs`                   |   ✅   |
| 18.5  | Operations documentation       | `docs/operations/` — 4 pages                                 | `make docs`                   |   ✅   |
| 18.6  | Decision records               | `docs/adr/` — 4 ADRs with alternatives and consequences      | `make docs`                   |   ✅   |
| 18.7  | Glossary                       | `docs/glossary.md`                                           | `make docs`                   |   ✅   |
| 18.8  | Repository manifest            | `docs/repository-manifest.md`, tables generated from the AST | `make docs` runs `--check`    |   ✅   |
| 18.9  | This matrix                    | `docs/traceability-matrix.md`                                | —                             |   ✅   |
| 18.10 | Every link and anchor resolves | —                                                            | `scripts/check_docs_links.py` |   ✅   |
| 18.11 | Runnable examples              | `examples/` — 4 scripts                                      | `make examples`, all exit 0   |   ✅   |

---

## What is not fully verified

Four rows above are ⚠️, and they share a cause: this environment cannot run them.

| Item                         | Why                             | What was done instead                                                |
| ---------------------------- | ------------------------------- | -------------------------------------------------------------------- |
| Docker build (17.3)          | No Docker daemon available here | The Dockerfile was reviewed line by line; CI builds it on every push |
| GitHub Actions (17.4)        | Requires GitHub                 | Workflow YAML reviewed; every command in it was run locally          |
| Pre-commit (17.6)            | Not installed here              | The hooks invoke the same commands as `make check`, which passes     |
| Palette accessibility (11.8) | Validator is not wired into CI  | Verified once when the palette was chosen; constants are pinned      |

One further deviation is worth stating plainly rather than burying:

**Python version.** The specification named Python 3.12. This environment has Python 3.10,
so the project targets `>=3.10` in order that the whole suite could actually be *executed
and verified* rather than merely written. CI matrixes 3.10, 3.11, and 3.12, and the Docker
image is built on 3.12. Nothing in the code uses a 3.11+ feature; the only concession is
avoiding PEP 695 type-parameter syntax.

## See also

- [Repository manifest](repository-manifest.md) — every file, its symbols, and its tests.
- [Test matrix](testing/test-matrix.md) — the same coverage from the tests' point of view.
- [Test strategy](testing/test-strategy.md) — why the suite is shaped this way.
- [Security](architecture/security.md) — the trust boundary in full.
