# Component design

What each package is responsible for, what it may depend on, and where the public API
boundary sits.

## The layering rule

Packages are arranged in five layers. **A package may only depend on packages below it.**

| Layer           | Packages                                                | May import      |
| --------------- | ------------------------------------------------------- | --------------- |
| 5 — Interface   | `cli`                                                   | anything        |
| 4 — Application | `config`, `orchestration`, `reporting`                  | layers 1–3      |
| 3 — Execution   | `models`, `datasets`, `training`, `evaluation`          | layers 1–2      |
| 2 — Domain      | `architectures`, `search_space`, `objectives`, `search` | layer 1         |
| 1 — Foundation  | `utilities`, `exceptions`, `observability`              | each other only |

This is enforced mechanically. `test_the_domain_does_not_import_the_orchestrator` walks the
AST of every module in the leaf packages and fails if one imports from a higher layer.

Why it matters: the layering is what makes each package testable in isolation. The
architecture genotype can be exercised without PyTorch. The objectives can be exercised
with hand-written numbers. A search strategy can be exercised without a dataset, a database,
or a GPU — which is the difference between a test suite that runs in five seconds and one
that runs in five minutes.

## The one deliberate exception

`persistence` imports `orchestration.lifecycle`.

The candidate state machine is a *domain* concept that both the repository (which stores
states) and the engine (which transitions them) need. Putting it in `persistence` would
make the engine depend on the database for a domain rule; duplicating it would let the two
copies drift.

So it lives in `orchestration/lifecycle.py`, a leaf module that imports nothing but
`exceptions`. `persistence → orchestration.lifecycle → exceptions` is acyclic, and the
module docstring records the constraint so a future change does not break it.

---

## Package by package

### `exceptions` — the error taxonomy

**Responsibility.** One exception hierarchy for the whole project.

Every deliberate failure derives from `NasEngineError`, which carries three things:

| Field       | Purpose                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `code`      | Stable machine-readable identifier, so callers never string-match messages |
| `details`   | Structured, JSON-serialisable context — field names, values, limits        |
| `retriable` | A class-level property driving the retry policy                            |

The `retriable` flag is the load-bearing part. It lets the orchestrator classify a failure
without knowing which component raised it:

```python
class NonFiniteLossError(TrainingError):
    retriable = False    # same seed, same architecture, same divergence

class EvaluationTimeoutError(NasEngineError):
    retriable = True     # depends on machine load
```

Messages state what was received, what was expected, and how to fix it:

```text
constraints.max_parameters=10 is below the minimum possible parameter count of roughly
322 for this space; no candidate can ever be feasible. Raise max_parameters or narrow
stage_channels.
```

---

### `utilities` — cross-cutting helpers

**Responsibility.** Seeding, environment capture, safe paths, stable hashing, canonical
JSON, monotonic timing, determinism configuration.

**Depends on.** `exceptions` only. Nothing here may import a domain package, which is what
keeps the dependency graph rooted.

| Module        | Provides                                                                    |
| ------------- | --------------------------------------------------------------------------- |
| `seeding`     | Seed derivation, seed bundles, RNG-state serialisation, worker isolation    |
| `determinism` | PyTorch determinism configuration and an honest report of what was achieved |
| `environment` | The environment snapshot persisted with every search                        |
| `hashing`     | BLAKE2b content hashing — never Python's randomised `hash()`                |
| `json_io`     | Canonical JSON, atomic writes, size-bounded reads                           |
| `paths`       | Filename sanitisation and path-traversal defence                            |
| `timing`      | Monotonic durations and timezone-aware UTC timestamps                       |

---

### `observability` — logging, events, counters

**Responsibility.** Structured event logging with a closed vocabulary, ambient identifier
context, in-process counters.

Event names are a *public interface* — dashboards and log queries are built on them — so
they are declared once as an enumeration rather than typed as free strings at call sites.
Adding an event means adding a member; renaming one is a breaking change.

Ambient context (`search_id`, `candidate_id`, `trial_id`, `worker_id`) travels through
`contextvars` rather than through every function signature. `contextvars` rather than
thread-locals because entering a context returns a token that restores the exact previous
value, making nesting safe.

A redaction processor walks every event dictionary and replaces the value of any key whose
name suggests a secret. Defence in depth: the framework never logs credentials, but user
configuration is untrusted.

---

### `architectures` — the genotype

**Responsibility.** The vocabulary the whole system speaks: specification, canonical form,
hashing, shape inference, cost model, summaries, lineage.

**Depends on.** `utilities`, `exceptions`. **Not PyTorch.** An architecture is pure data.

**Public.** `ArchitectureSpec`, `BlockSpec`, `StageSpec`, `StemSpec`, `HeadSpec`, the four
enums, `architecture_hash`, `summarise`.

The design is covered in [architecture encoding](../concepts/architecture-encoding.md) and
[ADR 0001](../adr/0001-search-space-representation.md).

---

### `search_space` — what may be searched

**Responsibility.** Space definition and validation, seeded sampling, structural repair,
mutation operators, four-layer validation.

**Depends on.** `architectures`.

The four validation layers are kept distinct because each answers a different question and
each has a different remedy:

| Layer      | Question                               | Where               | Failure means                                        |
| ---------- | -------------------------------------- | ------------------- | ---------------------------------------------------- |
| Schema     | Are the types and ranges valid?        | Pydantic            | The document is malformed                            |
| Semantic   | Do the tensors line up?                | `infer_shapes`      | The network cannot be built                          |
| Membership | Is every choice offered by this space? | `check_membership`  | The candidate came from a different space            |
| Constraint | Is it within the resource budget?      | `check_constraints` | Buildable but too expensive → `PRUNED`, not `FAILED` |

All findings are collected before raising, so a user fixing an imported architecture sees
every problem at once instead of one per attempt.

---

### `models` — the phenotype

**Responsibility.** Turning a genotype into a runnable `nn.Module`, and measuring it.

**Depends on.** `architectures`, PyTorch.

Four guarantees:

- **Fail before allocating.** Shapes are validated statically first, so an impossible
  architecture raises a precise error naming the offending block rather than a
  `RuntimeError` from deep inside cuDNN.
- **No hidden global state.** The builder takes everything as arguments and returns a fresh
  module. Nothing is cached behind the caller's back.
- **Deterministic module order.** Submodules are registered in genotype order, so
  `state_dict()` keys are stable and a checkpoint saved by one process loads in another.
- **Introspectable.** The module carries its specification, its shape trace, and its
  parameter counts.

`ModelBuilder` is a class rather than a bare function so that build-time policy
(initialisation, device, dtype) is configured once and *injected* wherever models are
needed. The evaluator receives a builder, not a hard-coded call — which is what makes the
evaluator testable with a stub builder that raises on demand.

---

### `datasets` — data providers and loaders

**Responsibility.** Dataset providers, split management, DataLoader construction,
low-fidelity views.

`DatasetProvider` is a `typing.Protocol` rather than an abstract base class, so user code
can supply a dataset without importing from this package — structural typing keeps the
dependency arrow pointing the right way.

The three splits have three jobs, and the separation is
[enforced structurally](../concepts/training-and-evaluation.md#validation-leakage):
`include_test` defaults to `False`, so leaking the test split requires deliberately passing
a flag.

Registration is explicit. Nothing is auto-discovered by importing a module named in
configuration — that would be arbitrary code execution.

---

### `training` — the training loop

**Responsibility.** Optimisers, schedules, metrics, early stopping, checkpoints, and the
loop itself.

**Knows nothing about** search strategies, candidates, databases, or objectives. That
boundary is a core design rule: a search strategy containing training code could not be
unit-tested without a GPU, and a trainer that knew about candidates could not be reused to
train the final winner.

`TrainingSettings` is a plain dataclass, deliberately **not** a Pydantic model. The training
package must not depend on the configuration framework; the config layer converts at the
boundary. That keeps the trainer usable from a plain script.

---

### `evaluation` — candidate measurement

**Responsibility.** Budgets, latency benchmarking, model-size measurement, the candidate
evaluator, and the result and failure types.

`CandidateEvaluator` is where a genotype becomes a set of measured metrics. Dependencies are
injected — dataset, loader settings, training settings, model builder — so tests can supply
a stub without patching module globals.

The evaluator **never raises** for a candidate-level problem. Every exception is caught,
classified, and returned inside the result. `KeyboardInterrupt` and `SystemExit` do
propagate, because they mean the operator wants the process to stop.

---

### `objectives` — multi-objective comparison

**Responsibility.** Objectives, constraints, normalisation, scoring, Pareto fronts, ranking.

Everything operates on plain `Mapping[str, float]` metrics, so the package has no dependency
on PyTorch, the evaluator, or persistence — and its logic is tested exhaustively with
hand-written numbers.

The `online` module exists because of a subtlety worth restating: population-relative
normalisation cannot be used *during* a search, because a value computed at evaluation 10 is
not comparable with one computed at evaluation 200. See
[multi-objective optimisation](../concepts/multi-objective-optimization.md#online-versus-final-scoring).

---

### `search` — strategies

**Responsibility.** The strategy interface and its three implementations.

**Depends on.** `search_space` (to sample and mutate) and `evaluation` (only for the budget
and result types). **Not** on the engine, persistence, or training code.

This is the project's central extension point. See
[ADR 0003](../adr/0003-search-strategy-interface.md) and
[adding a search strategy](../guides/adding-a-search-strategy.md).

---

### `persistence` — storage

**Responsibility.** Connection management, versioned schema, and the repository seam.

Nothing outside this package constructs SQL or holds a session. Two rules make that
enforceable:

- **Detached read models.** Every query returns a frozen dataclass, never an ORM instance.
  ORM objects are bound to the session that loaded them; touching a lazily-loaded attribute
  after the session closes raises `DetachedInstanceError` far from the cause.
- **One method, one transaction.** A method either fully applies or fully rolls back.

See [persistence](persistence.md) and [ADR 0002](../adr/0002-persistence-layer.md).

---

### `config` — validated configuration

**Responsibility.** The configuration models, the four-layer precedence chain, and
conversion to the domain's plain dataclasses.

This package sits at the boundary between untrusted input and the domain. It validates
once and hands the domain already-checked values, which is why no domain module needs a
Pydantic import or a defensive range check.

`extra="forbid"` on every model means a typo is an error rather than a silently ignored key
— a single setting that prevents an entire class of "why did my configuration have no
effect" confusion.

---

### `orchestration` — the engine

**Responsibility.** The candidate lifecycle, execution backends, retry policy, checkpoints,
and the search loop.

The top of the graph. It imports from every other package, and nothing imports from it
except `persistence`'s use of `lifecycle`.

---

### `reporting` — output

**Responsibility.** Markdown reports, CSV and JSON exports, matplotlib figures.

Reads **only** through the repository, so a report can be produced from a database file
long after the run that created it, on a different machine, without the engine.

---

### `cli` — the command line

**Responsibility.** Argument parsing, output formatting, exit codes.

A thin shell. Every command loads configuration, calls into the library, and formats the
result. No domain logic lives here — which is why the Python API can do everything the CLI
can.

Imports are deferred inside command functions so that `nas-engine --help` does not pay for
importing PyTorch.

---

## The public API boundary

**Public** — everything exported from `nas_engine/__init__.py`. Semantic versioning applies:
names will not be removed or given incompatible signatures within a major version.

That covers `SearchConfig`, `SearchEngine`, `SearchResult`, the architecture genotype types,
the search-space types, `SearchStrategy`, `SearchRepository` and its read models, the
objective types, and the exception taxonomy.

**Internal** — everything else. Submodules may be renamed, split, or rewritten in a minor
release. In particular, no guarantee is made about:

- the database schema or the ORM models — use `SearchRepository`, which *is* public;
- the exact contents of a strategy's `state_dict`;
- module paths of helpers not re-exported at the top level.

**Extension points that are public:**

| Extension         | Interface          | Registration                       |
| ----------------- | ------------------ | ---------------------------------- |
| A search strategy | `SearchStrategy`   | `register_strategy(name, factory)` |
| A dataset         | `DatasetProvider`  | `register_provider(name, factory)` |
| An objective      | `Objective`        | Construct directly                 |
| A constraint      | `MetricConstraint` | Construct directly                 |

Two tests keep the boundary honest: `test_every_exported_name_exists` and
`test_exports_are_sorted`.

## Design patterns, and why each is here

| Pattern                  | Where                           | The problem it solves                                                        |
| ------------------------ | ------------------------------- | ---------------------------------------------------------------------------- |
| **Strategy**             | `SearchStrategy`                | Adding a search algorithm without touching the engine                        |
| **Repository**           | `SearchRepository`              | Confining SQL to one seam so the schema can change independently             |
| **Dependency inversion** | Engine ↔ strategy               | Neither depends on the other; both depend on the interface                   |
| **Dependency injection** | Engine, evaluator               | Testing without a database, a dataset, or a GPU                              |
| **State machine**        | `CandidateStateMachine`         | Making the recovery story expressible and checkable                          |
| **Registry**             | Strategies, datasets            | Selection by name from configuration, without importing code named in a file |
| **Factory**              | `ModelBuilder`, layer factories | One place that knows how to build each thing                                 |
| **Value objects**        | Genotypes, budgets, results     | Immutable, comparable, hashable, serialisable                                |
| **Context manager**      | Sessions, stopwatch, eval mode  | Resources released even on the exception path                                |

Patterns that were deliberately *not* used:

- **No observer/event bus.** The engine calls the strategy directly. A bus would decouple
  things that are not independent and make the control flow harder to follow.
- **No plugin auto-discovery.** Importing a module named in a configuration file is
  arbitrary code execution.
- **No abstract base class for datasets.** A `Protocol` keeps the dependency arrow pointing
  the right way.

## See also

- [System overview](system-overview.md) — the diagrams.
- [Data flow](data-flow.md) — what moves between these components.
- [Adding a search strategy](../guides/adding-a-search-strategy.md) — the main extension
  point in practice.
