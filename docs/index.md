# nas-engine documentation

Neural Architecture Search, implemented from first principles and explained.

This documentation has two jobs. The **concepts** pages teach how NAS works — the
mathematics, the algorithms, and the failure modes — well enough that you could implement
it yourself. The **architecture**, **guides**, **testing**, and **operations** pages
explain how *this* implementation works, why each design was chosen, and what was given
up.

Every page cross-references the source and the tests. Where a claim is enforced by a test,
the test is named.

---

## Start here

| If you want to…                           | Read                                               |
| ----------------------------------------- | -------------------------------------------------- |
| Run your first search in five minutes     | [Getting started](getting-started.md)              |
| Understand what NAS is and why it is hard | [NAS foundations](concepts/nas-foundations.md)     |
| See how the system fits together          | [System overview](architecture/system-overview.md) |
| Look up a term                            | [Glossary](glossary.md)                            |
| Find a specific file                      | [Repository manifest](repository-manifest.md)      |
| Check what is verified and how            | [Traceability matrix](traceability-matrix.md)      |

## Concepts

The theory, with equations, and its connection to the code.

| Page | Covers |
| --- | --- |
| [NAS foundations](concepts/nas-foundations.md) | Search as optimisation; the three components; bi-level optimisation; exploration versus exploitation; budget allocation |
| [Search spaces](concepts/search-spaces.md) | What a space is, cardinality, conditional choices, constraints, bias, and how to design one |
| [Architecture encoding](concepts/architecture-encoding.md) | Genotype versus phenotype, canonical form, hashing, equality, duplicate detection |
| [Random search](concepts/random-search.md) | The baseline, why it is strong, and how to implement it correctly |
| [Regularized evolution](concepts/regularized-evolution.md) | Aging evolution, tournaments, mutation, and why age beats fitness for removal |
| [Successive halving](concepts/successive-halving.md) | Multi-fidelity allocation, the rank-correlation assumption, and its failure modes |
| [Multi-objective optimisation](concepts/multi-objective-optimization.md) | Pareto dominance, fronts, crowding, scalarisation, and the risks of weighted scores |
| [Training and evaluation](concepts/training-and-evaluation.md) | Convolution, normalisation, residuals, optimisation, and every measurement the evaluator takes |
| [Reproducibility](concepts/reproducibility.md) | Reproducibility versus determinism versus statistical repeatability; seeding; what cannot be promised |
| [Common pitfalls](concepts/common-pitfalls.md) | Ten NAS misconceptions, stated precisely and corrected |

## Architecture

How this implementation is built.

| Page                                                 | Covers                                                                        |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| [System overview](architecture/system-overview.md)   | Components, the domain model, the search lifecycle, diagrams                  |
| [Component design](architecture/component-design.md) | Every package's responsibility, the dependency graph, the public API boundary |
| [Data flow](architecture/data-flow.md)               | What moves between components, in what form, and where it is validated        |
| [Persistence](architecture/persistence.md)           | The data model, the repository pattern, transactions, schema evolution        |
| [Concurrency](architecture/concurrency.md)           | Execution modes, worker isolation, what concurrency does and does not change  |
| [Security](architecture/security.md)                 | The trust boundary, the threat model, and every mitigation                    |

## Guides

Task-oriented instructions.

| Page                                                           | Covers                                                                                 |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Running a search](guides/running-a-search.md)                 | Configuration, budgets, choosing a strategy, monitoring                                |
| [Resuming a search](guides/resuming-a-search.md)               | Interruption, recovery, what is restored and what is not                               |
| [Defining search spaces](guides/defining-search-spaces.md)     | Building a space, constraints, sizing, common mistakes                                 |
| [Adding an operation](guides/adding-an-operation.md)           | The five files to touch, and the hash-invalidation consequence                         |
| [Adding a search strategy](guides/adding-a-search-strategy.md) | The interface, a worked example, and sketches for Bayesian optimisation, RL, and DARTS |
| [Custom datasets](guides/custom-datasets.md)                   | Implementing and registering a provider                                                |
| [Interpreting results](guides/interpreting-results.md)         | Reading a report, reading a Pareto front, and knowing when a difference is real        |
| [Troubleshooting](guides/troubleshooting.md)                   | Every error this project raises, and what to do about it                               |

## Testing

| Page                                                      | Covers                                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| [Test strategy](testing/test-strategy.md)                 | The seven test categories and what each is for                     |
| [Test matrix](testing/test-matrix.md)                     | Every requirement mapped to the tests that verify it               |
| [Reproducibility tests](testing/reproducibility-tests.md) | What is asserted to be deterministic, and what is deliberately not |

## Operations

| Page                                                     | Covers                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------- |
| [Deployment](operations/deployment.md)                   | Local, container, and scheduled execution                      |
| [Observability](operations/observability.md)             | Structured events, log fields, counters, and what to alert on  |
| [Backup and recovery](operations/backup-and-recovery.md) | What to back up, how to restore, and how to verify a restore   |
| [Production runbook](operations/production-runbook.md)   | Pre-flight checks, monitoring, and the symptom-to-action table |

## Decisions

Architecture decision records, each with the alternatives considered and the trade-off
accepted.

| ADR                                             | Decision                                       |
| ----------------------------------------------- | ---------------------------------------------- |
| [0001](adr/0001-search-space-representation.md) | How architectures are represented              |
| [0002](adr/0002-persistence-layer.md)           | SQLite, SQLAlchemy, and hand-rolled migrations |
| [0003](adr/0003-search-strategy-interface.md)   | The propose/observe strategy contract          |
| [0004](adr/0004-concurrency-model.md)           | Batched process-pool execution                 |

## Reading order

For someone new to both NAS and this codebase:

1. [Getting started](getting-started.md) — run something.
2. [NAS foundations](concepts/nas-foundations.md) — understand the problem.
3. [Search spaces](concepts/search-spaces.md) and
   [Architecture encoding](concepts/architecture-encoding.md) — the two ideas everything
   else rests on.
4. [Random search](concepts/random-search.md) — the simplest complete algorithm.
5. [System overview](architecture/system-overview.md) — how the pieces connect.
6. [Common pitfalls](concepts/common-pitfalls.md) — before you trust any result.

For someone extending the project:

1. [Component design](architecture/component-design.md)
2. [ADR 0003](adr/0003-search-strategy-interface.md)
3. [Adding a search strategy](guides/adding-a-search-strategy.md)
4. [Test strategy](testing/test-strategy.md)
