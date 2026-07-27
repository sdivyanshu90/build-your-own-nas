# Test strategy

Seven categories, each answering a different question.

## The principles

**No network, no GPU, no downloads, no credentials.** Enforced mechanically in
[`tests/unit/test_public_api.py`](../../tests/unit/test_public_api.py), not by convention:
AST checks reject a test that imports a network client, calls `.cuda()`, or reads
`os.environ` directly.

**Deterministic.** Every seed is explicit. Nothing depends on wall-clock time, dictionary
iteration order, or the developer's shell — `conftest.py` strips `NAS_ENGINE__*` variables
for every test.

**Fast by default.** The whole default suite runs in under a minute. Anything genuinely slow
is marked `slow` and excluded; the nightly workflow includes it.

**Tests document behaviour.** A test name should state a property, not describe a mechanic.
`test_aging_removes_the_oldest_not_the_worst` says what the system guarantees.
`test_deque_maxlen` says how it is implemented.

## The seven categories

| Category         | Question                                    | Count | Runtime |
| ---------------- | ------------------------------------------- | ----: | ------- |
| Unit             | Does this unit behave as specified?         |  ~780 | ~6 s    |
| Property         | Does this invariant hold for *every* input? |   ~52 | ~8 s    |
| Integration      | Do these two components agree?              |   ~36 | ~9 s    |
| End-to-end       | Does the whole thing work?                  |   ~20 | ~28 s   |
| Failure-recovery | Does it survive things going wrong?         |   ~29 | ~8 s    |
| Regression       | Has anything drifted?                       |   ~44 | ~7 s    |
| Performance      | Has anything become accidentally quadratic? |   ~14 | ~1 s    |

---

### Unit tests

`tests/unit/`

The bulk. Each exercises one unit against its specification, with no database, no dataset
beyond a tiny fixture, and no search.

Organised by subject rather than by source file, because a reader looking for "how is
canonicalisation tested" should find one place:

| File                            | Covers                                                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| `test_utilities.py`             | Hashing, canonical JSON, paths, timing, seeding, determinism, environment              |
| `test_architectures.py`         | Genotype, canonicalisation, hashing, shapes, cost, summaries, lineage                  |
| `test_search_space.py`          | Space validation, sampling, repair, mutation, the four validation layers               |
| `test_models.py`                | Layer factories, blocks, initialisation, the builder                                   |
| `test_datasets_and_training.py` | Data, loaders, metrics, optimisers, schedules, early stopping, checkpoints, the loop   |
| `test_evaluation.py`            | Budgets, failure classification, latency, model size, the evaluator                    |
| `test_objectives.py`            | Objectives, constraints, normalisation, scoring, Pareto, ranking, online scalarisation |
| `test_strategies.py`            | The interface and all three strategies                                                 |
| `test_orchestration.py`         | State machine, retry policy, checkpoints, executors, results                           |
| `test_persistence.py`           | Connections, migrations, every repository operation, recovery                          |
| `test_config.py`                | Validation, precedence, merging, YAML safety, compatibility                            |
| `test_cli.py`                   | Help, exit codes, JSON output, doctor, error translation                               |
| `test_reporting.py`             | Exports, formula-injection defence, figures, skip reasons                              |
| `test_observability.py`         | Redaction, event severity, context, counters, the error taxonomy                       |
| `test_public_api.py`            | The public surface, the module graph, and the no-external-dependencies rules           |

---

### Property-based tests

`tests/property/`

A unit test asserts behaviour for the inputs a developer thought of. A property test asserts
an **invariant** for inputs Hypothesis generates, and shrinks any counterexample to a
minimal case.

The invariants tested are the ones the rest of the system silently assumes:

| Invariant                                                       | Why it matters                                                  |
| --------------------------------------------------------------- | --------------------------------------------------------------- |
| Canonicalisation is idempotent                                  | Otherwise hashes depend on how many times an object was rebuilt |
| Inactive conditional fields hold their sentinel                 | Otherwise equal networks hash differently                       |
| Equal canonical forms hash identically                          | The basis of duplicate detection                                |
| Different canonical forms almost never collide                  | Ditto                                                           |
| Serialisation round-trips exactly                               | Stored architectures must reload                                |
| The analytic cost model **equals** the measured parameter count | The parameter objective would otherwise be a lie                |
| Static shapes match what PyTorch produces                       | Pre-flight validation would otherwise be wrong                  |
| Output shape matches the class count                            |                                                                 |
| Every sample is a valid member of its space                     |                                                                 |
| Mutation stays inside the space                                 |                                                                 |
| **Mutation never modifies its parent**                          | Frozen genotypes protect the population                         |
| Repair is idempotent                                            |                                                                 |
| Pareto-front members are never dominated                        |                                                                 |
| Dominance is irreflexive, asymmetric, transitive                |                                                                 |
| Ranking is a total order, independent of input order            |                                                                 |
| Checkpoint round-trips continue the stream, not replay it       |                                                                 |
| The state machine rejects every transition outside its table    |                                                                 |

Two generator strategies:

- **Free generation** for data-only properties — draw arbitrary blocks and assert
  canonicalisation.
- **Constructive generation** for architectures — `buildable_architectures()` builds valid
  architectures *by construction* rather than drawing freely and discarding, which would
  waste most of Hypothesis's budget.

Budgets scale with `HYPOTHESIS_SCALE`:

```bash
HYPOTHESIS_SCALE=10 pytest tests/property
```

Explicit `@settings(max_examples=...)` decorators override a Hypothesis *profile*, so the
counts route through `tests/profiles.py::scaled` instead. The nightly workflow sets the
factor to 8.

**What is deliberately not written:** properties that restate an example. `assert
hash(x) == hash(x)` for one fixed `x` is a unit test wearing a costume.

---

### Integration tests

`tests/integration/`

Each exercises a **seam** — a place where two components must agree — rather than either
component in isolation:

| Seam                                      | Test                                                 |
| ----------------------------------------- | ---------------------------------------------------- |
| Configuration → search space and strategy | `TestConfigurationToComponents`                      |
| Static shape inference ↔ PyTorch          | `TestShapeInferenceMatchesTorch`                     |
| Trainer ↔ evaluator metrics               | `test_evaluator_metrics_match_a_direct_training_run` |
| Strategy → engine → repository            | `TestEngineAndRepository`                            |
| Engine → checkpoint → resume              | `TestCheckpointAndResume`                            |
| Repository → report generator             | `TestReportingFromPersistedResults`                  |
| Engine → executor → worker                | `tests/integration/test_worker_process.py`           |

The shape-inference test is the important one. Static shape inference is a *model* of
PyTorch's behaviour, and a model of a library is only trustworthy if it is continuously
checked against it.

The worker tests call `evaluate_task` **in-process** so its behaviour is observable — the
payload contract, the evaluator cache, seed isolation, and the guarantee that nothing
escapes. The real spawned path gets one slow end-to-end test: a spawned-process test is slow
and its failures are hard to attribute.

---

### End-to-end tests

`tests/end_to_end/`

Complete searches through the public API and the CLI, on synthetic data, in seconds.

| What                                                   | Test                                                          |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| Random search completes and produces a winner          | `TestRandomSearchEndToEnd`                                    |
| Evolution fills and ages a population                  | `TestEvolutionEndToEnd`                                       |
| Lineage is recorded                                    | `test_lineage_is_recorded`                                    |
| Successive halving climbs its ladder                   | `TestSuccessiveHalvingEndToEnd`                               |
| The same architecture is re-evaluated at a higher rung | `test_the_same_architecture_is_re_evaluated_at_a_higher_rung` |
| A crashed evaluation is recovered on resume            | `TestInterruptionAndResume`                                   |
| Report, CSV, JSON, and plots are produced              | `TestArtifactsAndExports`                                     |
| The winner reloads and performs inference              | `TestBestModelReload`                                         |
| The documented CLI sequence works                      | `TestCommandLineEndToEnd`                                     |

That last one runs eleven CLI commands in sequence — every command in the README's
quick-start. If a documented command breaks, this fails.

Marked `slow`: a longer evolution run and the multiprocessing path.

---

### Failure-recovery tests

`tests/failure_recovery/`

Every scenario simulates something going wrong and asserts two things: the failure is
recorded with an understandable status, **and the rest of the search survives**.

| Scenario                    | Assertion                                      |
| --------------------------- | ---------------------------------------------- |
| Training exception          | Captured, classified `TRAINING`, retriable     |
| Divergent loss              | Classified `DIVERGENCE`, **not** retriable     |
| Unbuildable architecture    | Classified `BUILD`, permanent                  |
| Resource-limit violation    | Reported before the model is built             |
| A failing candidate         | The search continues; the others complete      |
| Retriable failure           | Retried, then succeeds                         |
| Retry exhaustion            | Candidate fails with two recorded trials       |
| Permanent failure           | Never retried, even with retries available     |
| Corrupt training checkpoint | Rejected with an actionable message            |
| Truncated checkpoint        | Rejected                                       |
| Missing checkpoint          | Training starts fresh                          |
| Corrupt search checkpoint   | Resume refuses                                 |
| Interrupted evaluation      | Requeued, or failed when retries are exhausted |
| Database write failure      | Transaction rolls back                         |
| Unreachable database        | Reported as `PersistenceError`                 |
| Duplicate proposal          | Rejected, not re-evaluated                     |
| Incompatible config version | Resume refuses                                 |
| Missing artifact file       | Reported clearly                               |
| Dead worker                 | Becomes a retriable failure                    |

Two real bugs were found by writing these:

1. **The worker's failure path could itself raise.** A malformed budget made the exception
   handler's fallback `TrainingBudget.from_dict` throw, so an exception escaped the worker
   after all. Fixed with `_safe_budget`.
2. **A resumed evaluation aborted its own transaction.** Rewriting the same artifact
   violated a uniqueness constraint, which rolled back the whole `complete_trial` — losing
   the metrics alongside it. Fixed by updating rather than inserting.

Neither would have been found by testing the happy path.

---

### Regression tests

`tests/regression/`

Two kinds.

**Golden fixtures** pin values that must not drift silently: canonical architecture JSON and
its hash, parameter counts, MAC counts, output shapes, Pareto outcomes for hand-checked
cases, the state-transition table, and the report's structure.

Fixtures are **intentionally versioned**. Each file carries a `fixture_version`; changing a
golden value requires bumping it and recording why. A failure here is not necessarily a bug
— it is a demand for a deliberate decision:

```text
The architecture hash changed. Every stored hash in every existing database is now wrong.
If this change is intentional, bump fixture_version in tests/fixtures/architectures.json
and document the migration.
```

**Determinism tests** are covered separately in
[reproducibility tests](reproducibility-tests.md).

---

### Performance tests

`tests/performance/`

**Guard rails, not benchmarks.** Each threshold sits roughly an order of magnitude above the
observed time on a modest laptop CPU, so ordinary machine-to-machine variation, CI noise, and
a busy scheduler cannot make them fail.

What they catch is an accidental algorithmic regression: an $O(n)$ operation becoming
$O(n^2)$, a per-call allocation becoming a per-call model build, a query losing its index.

| Guard                                   | Threshold | Observed |
| --------------------------------------- | --------- | -------- |
| Sampling                                | 20 ms     | ~0.2 ms  |
| Validation                              | 10 ms     | ~0.08 ms |
| Hashing                                 | 5 ms      | ~0.04 ms |
| Shape inference                         | 5 ms      | ~0.03 ms |
| The cost model is cheaper than building | 5× margin | ~80×     |
| Pareto front over 200                   | 5 s       | ~60 ms   |
| Ranking 200 candidates                  | 5 s       | ~65 ms   |
| Inserting 50 candidates                 | 20 s      | ~0.3 s   |
| Restoring strategy state                | 1 s       | ~0.2 ms  |

They are environment sensitive by nature. If one fails, the first question is "is this
machine loaded?", not "is the code broken". For numbers worth quoting, use
`scripts/benchmark.py`.

---

## Fixtures

`tests/conftest.py` provides:

| Fixture                                      | Scope    | Purpose                                                   |
| -------------------------------------------- | -------- | --------------------------------------------------------- |
| `default_space`, `tiny_space`, `micro_space` | session  | The three presets                                         |
| `sampler`, `sample_spec`                     | function | A seeded sampler and one architecture                     |
| `manual_spec`                                | function | A **hand-written** architecture with known structure      |
| `synthetic_bundle`                           | session  | A 96-example dataset                                      |
| `database`, `repository`                     | function | In-memory database with the schema applied                |
| `file_database`                              | function | File-backed, for tests needing persistence across handles |
| `smoke_config`, `config_factory`             | function | Minimal, fast configurations                              |

`manual_spec` is deliberately hand-written rather than sampled, so tests asserting exact
parameter counts do not silently change when the sampler changes.

`synthetic_bundle` is session-scoped because generating it costs milliseconds that every
test would otherwise pay.

---

## Coverage

The gate is **90% of lines and 85% of branches**, checked separately by
`scripts/check_coverage.py` because `coverage.py` can only fail on one combined number.

Current: **93% line, 88% branch**.

Deliberately uncovered:

| Code                         | Why                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| CUDA-only paths              | No GPU in CI. Guarded with `# pragma: no cover`                                           |
| Apple MPS paths              | Same                                                                                      |
| Enum exhaustiveness guards   | mypy proves them unreachable; they exist for values arriving from outside the type system |
| `torchvision` import failure | Requires uninstalling an optional dependency                                              |

Coverage is a **floor, not a target**. A line executed by a test that asserts nothing is
covered and untested. The suite is not padded to raise the number.

---

## Running tests

```bash
make test                # the default suite, excludes `slow`
make test-unit           # one category
make test-property
make test-integration
make test-e2e            # includes the slow ones
make test-recovery
make test-regression
make test-performance
make test-all            # everything
make coverage            # with the gate

pytest tests/unit/test_architectures.py::TestHashing -v
pytest -k canonicalisation
pytest -m slow
HYPOTHESIS_SCALE=10 pytest tests/property
```

## Writing a new test

1. **Pick the category.** One unit → unit. An invariant → property. Two components → 
   integration. A whole search → end-to-end.
2. **Name it as a property**, not a mechanic.
3. **Use the fixtures.** Do not build a dataset or a database by hand.
4. **Seed everything.** No unseeded randomness.
5. **Assert one thing** — or a few closely related things.
6. **Mark it `slow`** if it takes more than a second or two.
7. **No network, no GPU, no credentials.** The guards will catch you.

## See also

- [Test matrix](test-matrix.md) — every requirement mapped to its tests.
- [Reproducibility tests](reproducibility-tests.md) — what is asserted deterministic.
