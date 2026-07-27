# Test matrix

Every requirement mapped to the tests that verify it.

**1 089 tests** across seven categories. The default run (`make test`) executes 1 087 in
about 50 seconds; two are marked `slow`.

| Category                   | Tests | Directory                 |
| -------------------------- | ----: | ------------------------- |
| Unit                       |   867 | `tests/unit/`             |
| Property-based             |    52 | `tests/property/`         |
| Integration                |    36 | `tests/integration/`      |
| End-to-end                 |    20 | `tests/end_to_end/`       |
| Regression and determinism |    71 | `tests/regression/`       |
| Failure-recovery           |    29 | `tests/failure_recovery/` |
| Performance                |    14 | `tests/performance/`      |

Coverage: **93.7% of lines, 86.7% of branches** (gate: 90% / 85%).

---

## Search-space system

| Requirement                         | Verified by                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| Typed, validated search spaces      | `test_search_space.py::TestChoiceValidation`                                                 |
| Duplicate choices removed           | `test_duplicates_are_removed_to_avoid_biasing_sampling`                                      |
| Even kernels rejected               | `test_even_kernels_are_rejected`                                                             |
| A parametric operation is required  | `test_parametric_operations_are_required`                                                    |
| Cardinality estimation              | `TestSpaceIntrospection`                                                                     |
| Infeasible constraints caught early | `test_impossible_parameter_ceiling_is_reported_early`                                        |
| Every preset is valid and samplable | `TestPresets::test_every_preset_is_valid_and_samplable`                                      |
| Deterministic seeded sampling       | `TestSampler::test_sampling_is_reproducible`, `test_sampling_is_a_pure_function_of_the_seed` |
| Duplicate avoidance                 | `test_unique_sampling_avoids_known_hashes`, `test_unique_sampling_never_repeats`             |
| Exhaustion handling                 | `test_exhausted_space_returns_none`                                                          |
| Rejection statistics                | `test_statistics_track_rejections`                                                           |
| Sampler checkpointing               | `test_state_round_trip_continues_the_stream`                                                 |
| Monotonic widths respected          | `test_monotonic_widths_are_respected`                                                        |
| Only the first block downsamples    | `test_only_the_first_block_of_a_stage_downsamples`                                           |
| Sampling ignores the global RNG     | `test_random_module_is_never_used_implicitly`                                                |

## Architecture encoding

| Requirement                    | Verified by                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Deterministic serialisation    | `test_architectures.py::TestCanonicalSerialisation`                            |
| Canonical ordering             | `test_canonical_json_sorts_keys`                                               |
| Stable architecture hashing    | `TestHashing`, `test_equal_architectures_hash_identically`                     |
| Hash collision resistance      | `test_different_architectures_almost_never_collide`                            |
| Architecture equality          | `test_equality_helper_matches_hash_equality`                                   |
| Conditional canonicalisation   | `TestBlockCanonicalisation`, `test_inactive_fields_hold_their_sentinel_values` |
| Canonicalisation is idempotent | `test_canonicalisation_is_idempotent`                                          |
| Round-trip is lossless         | `test_json_round_trip_is_lossless`                                             |
| Schema validation              | `test_unknown_fields_are_rejected`, `test_rejects_unknown_operations`          |
| Immutability                   | `test_blocks_are_immutable`, `test_with_block_leaves_the_original_untouched`   |
| Schema versioning              | `test_rejects_a_future_schema_version`                                         |
| Golden hashes unchanged        | `test_golden_fixtures.py::test_hashes_are_unchanged`                           |

## Shape inference and validation

| Requirement                         | Verified by                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Tensor-shape inference              | `test_architectures.py::TestShapeInference`                            |
| Matches PyTorch exactly             | `test_component_integration.py::TestShapeInferenceMatchesTorch`        |
| Shape incompatibility rejected      | `test_rejects_channel_changing_pooling`                                |
| Unsupported residual rejected       | `test_rejects_residual_across_a_channel_change`, `..._across_a_stride` |
| Impossible downsampling rejected    | `test_search_space.py::test_impossible_downsampling_is_relaxed`        |
| Non-positive channels rejected      | `test_out_of_range_fields_are_rejected`                                |
| Invalid kernel size rejected        | `test_even_kernel_is_rejected`, `test_unsupported_kernel_is_rejected`  |
| Parameter limits enforced           | `test_constraint_violation_is_distinguished_from_invalidity`           |
| Choices outside the schema rejected | `TestValidationLayers::test_membership_detects_a_foreign_operation`    |
| Errors name the offending element   | `test_error_names_the_offending_block`                                 |
| All issues collected before raising | `test_issue_serialises_to_plain_data`                                  |

## Model construction

| Requirement                        | Verified by                                                         |
| ---------------------------------- | ------------------------------------------------------------------- |
| Valid PyTorch models built         | `test_models.py::TestBuilder`                                       |
| Intermediate shapes inferred       | `test_runtime_shapes_match_the_static_trace`                        |
| Residual paths validated           | `test_illegal_residual_is_rejected`                                 |
| Deterministic module graphs        | `test_two_builds_produce_identical_module_ordering`                 |
| Model summary                      | `test_summary_serialises_to_plain_data`                             |
| Trainable and non-trainable counts | `test_parameter_counting_separates_buffers`                         |
| Analytic cost is exact             | `test_analytic_cost_equals_the_measured_parameter_count` (property) |
| Configurable input and class count | `test_output_shape_matches_the_class_count` (property)              |
| No hidden global state             | `test_initialisation_is_reproducible`                               |
| Feature shapes exposed             | `test_runtime_shapes_match_the_static_trace`                        |
| Fails before training when invalid | `test_structurally_invalid_specs_fail_before_allocation`            |
| Weight initialisation              | `TestInitialization` (5 tests)                                      |
| Gradients reach every parameter    | `test_gradients_flow_to_every_trainable_parameter`                  |

## Random search

| Requirement                  | Verified by                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| Seeded candidate generation  | `test_strategies.py::TestRandomSearch::test_is_reproducible` |
| Duplicate avoidance          | `test_proposals_are_unique`                                  |
| Search constraints respected | `test_every_sample_is_a_valid_member` (property)             |
| Budget limits                | `test_proposes_up_to_the_budget`                             |
| Checkpointing                | `test_state_round_trip_continues_the_stream`                 |
| Search resume                | `test_random_search_state_round_trips` (property)            |
| Multi-objective recording    | `test_full_searches.py::TestRandomSearchEndToEnd`            |
| Exhaustion stops the search  | `test_exhausted_space_stops_the_search`                      |

## Regularized evolution

| Requirement                         | Verified by                                                      |
| ----------------------------------- | ---------------------------------------------------------------- |
| Population initialisation           | `TestRegularizedEvolution::test_initial_proposals_are_random`    |
| Tournament sampling                 | `test_tournament_prefers_fitter_parents`                         |
| Parent selection                    | same                                                             |
| Mutation                            | `test_switches_to_mutation_once_the_population_is_seeded`        |
| Candidate validation                | `test_children_stay_inside_the_space` (property)                 |
| Child evaluation                    | `test_full_searches.py::TestEvolutionEndToEnd`                   |
| Population aging                    | **`test_aging_removes_the_oldest_not_the_worst`**                |
| Removal of the oldest               | same                                                             |
| Population capped                   | `test_population_is_capped`                                      |
| Duplicate handling                  | `test_a_duplicate_proposal_is_rejected_not_re_evaluated`         |
| Persistent algorithm state          | `test_state_round_trip_preserves_the_population`                 |
| Deterministic resume                | `test_state_round_trip_continues_the_stream`                     |
| Failures never enter the population | `test_failed_candidates_never_enter_the_population`              |
| Empty-population fallback           | `test_empty_population_falls_back_to_random_sampling`            |
| Mutation never modifies the parent  | `test_the_parent_is_never_modified` (property, all 12 operators) |
| Lineage recorded                    | `test_lineage_is_recorded`                                       |

## Successive halving

| Requirement                      | Verified by                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| Low-fidelity evaluation          | `TestSuccessiveHalving::test_first_rung_proposes_random_candidates` |
| Promotion to larger budgets      | `test_later_rungs_use_larger_budgets`                               |
| Configurable epochs              | `TestResourceLadder::test_epochs_grow_geometrically`                |
| Configurable data fraction       | `test_data_fraction_scaling_caps_at_one`                            |
| Configurable resolution          | `test_resolution_scaling_caps_at_native`                            |
| Equal cost per rung              | `test_each_rung_costs_about_the_same`                               |
| The promotion barrier            | `test_promotion_waits_for_the_whole_rung`                           |
| The best are promoted            | `test_promotes_the_best_candidates`                                 |
| Bracket completion               | `test_bracket_completes`                                            |
| No survivors ends the bracket    | `test_all_failures_end_the_bracket`                                 |
| Re-evaluation is not a duplicate | `test_the_same_architecture_is_re_evaluated_at_a_higher_rung`       |
| State round-trip                 | `test_state_round_trip_preserves_rung_progress`                     |

## Extensibility contract

| Requirement                         | Verified by                                                       |
| ----------------------------------- | ----------------------------------------------------------------- |
| Proposing candidates                | `TestStrategyContract`, all three strategies' tests               |
| Receiving results                   | `test_records_observations`                                       |
| Updating internal state             | `test_population_statistics_are_reported`                         |
| Serialising state                   | Every strategy's `test_state_round_trip_*`                        |
| Restoring state                     | same                                                              |
| Determining completion              | `test_exhausted_space_stops_the_search`, `test_bracket_completes` |
| Strategy statistics                 | `test_statistics_serialise_with_extras`                           |
| Default hooks are no-ops            | `test_default_hooks_are_no_ops`                                   |
| Registry works for every strategy   | `TestRegistry::test_every_strategy_builds_from_the_registry`      |
| Custom strategies can be registered | `test_custom_strategies_can_be_registered`                        |

## Evaluation and training

| Requirement                       | Verified by                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------- |
| Training                          | `test_datasets_and_training.py::TestTrainer::test_training_reduces_the_loss` |
| Validation                        | same                                                                         |
| Testing                           | `test_evaluation.py::test_test_evaluation_reports_test_metrics`              |
| Early stopping                    | `test_early_stopping_ends_the_run`, `TestEarlyStopping` (7 tests)            |
| Gradient clipping                 | Exercised throughout; configured in `TrainingSettings`                       |
| Configurable optimizer            | `TestOptimizers` (7 tests)                                                   |
| Configurable scheduler            | `TestSchedulers` (6 tests)                                                   |
| Mixed precision when supported    | `test_mixed_precision_falls_back_on_cpu`                                     |
| Device selection                  | `test_device_placement_is_honoured`, `TestDeviceResolution`                  |
| Model checkpointing               | `TestTrainingCheckpoints` (8 tests)                                          |
| Training resume                   | `test_resume_continues_from_the_checkpoint`                                  |
| Deterministic mode                | `TestDeterminism`, `tests/regression/test_determinism.py`                    |
| Metric aggregation                | `TestMetrics` (11 tests)                                                     |
| Failure capture                   | `test_failures_are_returned_not_raised`                                      |
| Timeout handling                  | `test_timeout_is_enforced`                                                   |
| Maximum parameter constraints     | `test_parameter_limit_prunes_before_building`                                |
| SGD and AdamW                     | `test_builds_sgd`, `test_builds_adamw`                                       |
| Cross-entropy, accuracy, top-k    | `TestMetrics`                                                                |
| Parameter count                   | `test_analytic_cost_equals_the_measured_parameter_count`                     |
| Serialised model size             | `TestModelSize` (5 tests)                                                    |
| Latency: warm-up, timed, repeated | `TestLatency::test_reports_positive_statistics`                              |
| Latency: median and percentiles   | same                                                                         |
| Latency: device metadata          | `test_carries_the_portability_warning`                                       |
| Latency: portability warning      | same, and `test_latency_is_not_asserted_to_be_reproducible`                  |
| Training and search stay separate | Enforced by `test_the_domain_does_not_import_the_orchestrator`               |

## Multi-objective optimisation

| Requirement                         | Verified by                                                   |
| ----------------------------------- | ------------------------------------------------------------- |
| Weighted scalar scoring             | `test_objectives.py::TestWeightedScoring`                     |
| Hard constraints                    | `TestConstraints` (9 tests)                                   |
| Pareto dominance                    | `TestParetoDominance` (9 tests)                               |
| Pareto-front computation            | `TestParetoFront` (8 tests)                                   |
| Tie-breaking rules                  | `test_ranking_is_deterministic_under_reordering`              |
| Objective direction                 | `test_direction_signs`, `test_minimisation_inverts_the_scale` |
| Missing-metric handling             | `test_missing_required_metric_leaves_the_score_unset`         |
| Metric normalisation                | `TestNormalisation` (10 tests)                                |
| Reproducible ranking                | `test_ranking_is_independent_of_input_order` (property)       |
| Dominance is a strict partial order | `test_dominance_is_asymmetric`, `..._irreflexive` (property)  |
| Front members are never dominated   | `test_front_members_are_never_dominated` (property)           |
| Online scalarisation is time-stable | `test_value_is_stable_regardless_of_population`               |
| Golden Pareto cases                 | `test_golden_fixtures.py::TestParetoFixture`                  |

## Orchestration

| Requirement                               | Verified by                                                    |
| ----------------------------------------- | -------------------------------------------------------------- |
| Candidate requested from the strategy     | `test_component_integration.py::TestEngineAndRepository`       |
| Architecture validated                    | `test_a_pruned_candidate_is_distinguished_from_a_failure`      |
| Canonical representation and hash         | `test_all_candidates_are_unique`                               |
| Duplicate detection                       | `test_a_duplicate_proposal_is_rejected_not_re_evaluated`       |
| Candidate persisted                       | `test_a_search_persists_every_candidate`                       |
| Evaluation queued and run                 | `test_trials_and_metrics_are_recorded`                         |
| Metrics and artifacts persisted           | same                                                           |
| Strategy notified                         | `test_records_observations`                                    |
| Leaderboards and Pareto fronts updated    | `test_ranking_recomputed_from_the_database_matches_the_engine` |
| Search state checkpointed                 | `test_the_engine_writes_checkpoints`                           |
| Stopping conditions                       | `TestRandomSearchEndToEnd`, `test_the_ladder_is_climbed`       |
| Explicit candidate states                 | `test_orchestration.py::TestStateMachine` (10 tests)           |
| Transitions validated                     | `test_only_table_edges_are_permitted` (property, all 64 pairs) |
| Safe recovery after interruption          | `TestInterruptionAndResume`, `TestRecovery`                    |
| Completed vs never-started vs interrupted | `test_persistence.py::TestRecovery`                            |
| Configurable retry with bounded retries   | `TestRetryPolicy` (10 tests)                                   |
| Error classification                      | `TestFailureClassification` (12 parametrised cases)            |

## Concurrency

| Requirement                                    | Verified by                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| Sequential execution                           | The whole suite runs sequentially                                                      |
| Local multiprocessing                          | `test_full_searches.py::test_multiprocessing_produces_the_same_kind_of_result` (slow)  |
| No two workers evaluate the same architecture  | `test_a_claimed_candidate_cannot_be_claimed_again`, `test_identity_is_unique_per_rung` |
| Database writes stay consistent                | `test_a_write_failure_rolls_the_transaction_back`                                      |
| Worker failures do not corrupt state           | `test_a_dead_worker_becomes_a_retriable_failure`                                       |
| RNG isolated per worker                        | `test_worker_bundles_are_isolated`                                                     |
| Logs identify search, candidate, trial, worker | `test_ambient_context_is_attached_to_events`                                           |
| Sequential results are reproducible            | `tests/regression/test_determinism.py`                                                 |
| Worker payload contract                        | `test_worker_process.py` (14 tests)                                                    |
| Nothing escapes the worker                     | `test_nothing_escapes_the_worker`                                                      |

## Persistence

| Requirement                        | Verified by                                                    |
| ---------------------------------- | -------------------------------------------------------------- |
| Search runs, configurations, seeds | `test_persistence.py::TestSearchRecords`                       |
| Architecture specs and hashes      | `TestCandidates::test_specification_is_revalidated_on_read`    |
| Candidate status                   | `test_state_transitions_are_validated`                         |
| Parent-child relationships         | `test_lineage_nodes_carry_parents_and_mutations`               |
| Mutation descriptions              | same                                                           |
| Training budgets                   | `test_completed_trial_stores_metrics_and_artifacts`            |
| Metrics, failures, retry counts    | `TestTrialsAndMetrics`, `test_retry_counter_increments`        |
| Checkpoints                        | `TestCheckpointsAndEvents`                                     |
| Artifact locations                 | `test_completed_trial_stores_metrics_and_artifacts`            |
| Environment metadata               | `test_configuration_and_environment_round_trip`                |
| Strategy state                     | `test_checkpoints_are_append_only`                             |
| Timestamps                         | `test_timestamps_are_timezone_aware`                           |
| Versioned schema management        | `TestMigrations` (4 tests)                                     |
| Creating a new search              | `test_create_and_fetch`                                        |
| Resuming                           | `TestCheckpointAndResume`                                      |
| Inspecting status                  | `test_counts_include_every_state`                              |
| Listing candidates                 | `test_listing_can_filter_and_paginate`                         |
| Retrieving the best                | `test_best_candidate_uses_a_single_metric`                     |
| Exporting                          | `TestArtifactsAndExports`                                      |
| Reconstructing lineage             | `test_lineage_nodes_carry_parents_and_mutations`               |
| Recomputing Pareto fronts          | `test_ranking_recomputed_from_the_database_matches_the_engine` |
| No unsafe deserialisation          | `test_corrupt_specification_is_rejected_on_read`               |
| Cascade deletes                    | `test_deleting_a_search_cascades`                              |

## Configuration

| Requirement                           | Verified by                                            |
| ------------------------------------- | ------------------------------------------------------ |
| Separate sections                     | `test_config.py::TestDefaults`                         |
| Precedence order                      | `TestPrecedence` (6 tests)                             |
| Invalid config fails early            | `TestValidation` (11 tests)                            |
| Errors name field, value, expectation | `test_error_names_the_field_and_value`                 |
| All problems reported together        | `test_all_problems_are_reported_together`              |
| Versioning                            | `test_future_version_is_rejected`, `TestCompatibility` |
| Deep merging                          | `TestMerging` (6 tests)                                |
| YAML safety                           | `test_python_object_tags_are_refused`                  |
| Conversion to domain dataclasses      | `TestConversion` (6 tests)                             |

## Command-line interface

| Requirement                         | Verified by                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| Every command has help and examples | `test_cli.py::test_every_command_has_help_and_examples` (13 commands) |
| Meaningful exit codes               | `TestExitCodes` (6 tests)                                             |
| Human-readable output               | `TestInit`, `TestValidateConfig`, `TestDoctor`                        |
| Machine-readable JSON               | `test_json_output_is_parseable_for_every_query_command`               |
| `doctor` inspects the environment   | `TestDoctor::test_reports_every_check`                                |
| The documented sequence works       | `test_the_documented_command_sequence_works` (11 commands)            |
| Hash-prefix lookup                  | `test_show_candidate_accepts_a_hash_prefix`                           |

## Reporting

| Requirement                           | Verified by                                              |
| ------------------------------------- | -------------------------------------------------------- |
| Markdown report                       | `test_full_searches.py::TestArtifactsAndExports`         |
| JSON export                           | `test_json_export_is_self_describing`                    |
| CSV export                            | `test_csv_export_lists_every_candidate`                  |
| PNG plots                             | `test_reporting.py::TestPlots` (9 tests)                 |
| Deterministic filenames               | `test_report_names_are_deterministic`                    |
| Required sections present and ordered | `test_golden_fixtures.py::TestReportStructureFixture`    |
| Known limitations always included     | `TestKnownLimitations`                                   |
| Formula-injection defence             | `TestSanitisation`                                       |
| Reports work from the database alone  | `test_a_report_can_be_generated_from_the_database_alone` |

## Reproducibility

Covered in detail in [reproducibility tests](reproducibility-tests.md).

## Security

| Control                    | Verified by                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| No unsafe YAML             | `test_python_object_tags_are_refused`                                     |
| Path traversal blocked     | `TestPathValidation` (6 tests)                                            |
| Filename sanitisation      | `test_reduces_hostile_input` (7 hostile inputs)                           |
| Size caps                  | `test_oversized_file_is_refused`, `test_rejects_oversized_payload`        |
| Log redaction              | `TestRedaction` (7 tests)                                                 |
| Environment allow-list     | `test_only_allow_listed_variables_are_captured`                           |
| Resource limits            | `test_parameter_limit_prunes_before_building`, `test_timeout_is_enforced` |
| Recursion bounds           | `test_recursion_is_bounded`, `test_cycles_terminate`                      |
| No `eval`/`exec`           | Ruff `S` rule, project-wide                                               |
| No SQL interpolation       | SQLAlchemy expression language throughout                                 |
| Container runs as non-root | `nightly.yml`                                                             |

## Project-level guarantees

| Guarantee                                   | Verified by                                           |
| ------------------------------------------- | ----------------------------------------------------- |
| Every exported name exists                  | `test_every_exported_name_exists`                     |
| Exports are sorted                          | `test_exports_are_sorted`                             |
| Every module imports cleanly                | `test_every_module_imports_cleanly`                   |
| No circular imports                         | same                                                  |
| The domain does not import the orchestrator | `test_the_domain_does_not_import_the_orchestrator`    |
| No wildcard imports                         | `test_no_module_uses_a_wildcard_import`               |
| No test needs the network                   | `test_no_test_module_imports_a_network_client`        |
| No test needs a GPU                         | `test_no_test_moves_tensors_to_a_gpu`                 |
| No test reads the ambient environment       | `test_no_test_reads_the_process_environment_directly` |
| CIFAR-10 never downloads without permission | `test_cifar10_never_downloads_without_permission`     |
| The quick start works                       | `test_the_documented_quick_start_works`               |

## Documentation and observability guarantees

Prose and log-event names are interfaces too, and neither the type checker nor the test
runner notices when they rot. These guards close that gap.

| Guarantee                                            | Verified by                                                |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| No module shadows an `Event` name with a raw string  | `test_no_module_logs_a_raw_string_owned_by_the_event_enum` |
| Module log names are namespaced                      | `test_module_scoped_log_names_are_namespaced`              |
| The same quantity has the same field name everywhere | `test_no_emit_uses_a_non_canonical_field_name`             |
| Every test the docs cite exists                      | `test_every_cited_test_function_exists`                    |
| Every test class the docs cite exists                | `test_every_cited_test_class_exists`                       |
| Counterexample names in the prose stay fictional     | `test_illustrative_names_really_are_absent`                |
| The documented CLI command count is true             | `test_the_cli_command_count_is_right`                      |
| The documented export count is true                  | `test_the_exported_symbol_count_is_right`                  |
| The documented table count is true                   | `test_the_table_count_is_right`                            |
| The documented mutation-operator count is true       | `test_the_mutation_operator_count_is_right`                |
| The documented page counts are true                  | `test_the_documentation_page_counts_are_right`             |
| Every module is imported by some test                | `test_every_module_is_imported_by_a_test`                  |
| Every documentation table is well-formed             | `test_every_table_is_well_formed`                          |
| The citation scanner is not vacuous                  | `test_the_scan_finds_citations_at_all`                     |
| The source scanner is not vacuous                    | `test_the_scan_actually_reads_the_package`                 |
| The shadowing detector really detects                | `test_a_shadowing_call_would_be_detected`                  |
| The field-name detector really detects               | `test_a_non_canonical_field_would_be_detected`             |

The last row matters more than it looks. Every guard here works by scanning files, and a
scanner whose pattern stops matching passes silently while checking nothing. Each one
therefore asserts that it still finds what it expects to find.

## See also

- [Test strategy](test-strategy.md) — what each category is for.
- [Reproducibility tests](reproducibility-tests.md).
- [Traceability matrix](../traceability-matrix.md) — requirements to implementation.
