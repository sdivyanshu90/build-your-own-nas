# Neural Architecture Search from Scratch

A working, tested, documented Neural Architecture Search framework built on PyTorch — and
an explanation of how NAS actually works, written for an engineer who knows Python and the
basics of machine learning.

Every part of the search is implemented here rather than delegated to an AutoML library:
the search space, the encoding, the hashing, the sampler, the mutation operators, the
search algorithms, the scheduler, the candidate state machine, the multi-objective
comparison, the checkpoint and resume logic, and the persistence layer.

```python
from nas_engine import SearchConfig, SearchEngine

config = SearchConfig.from_yaml("configs/random_search.yaml")
engine = SearchEngine(config)
result = engine.run()
print(result.summary())
```

```console
$ nas-engine search --config configs/evolution.yaml --report
$ nas-engine pareto
$ nas-engine best
```

---

## What it does

| Capability                                       | Code                            | Documentation                                                    |
| ------------------------------------------------ | ------------------------------- | ---------------------------------------------------------------- |
| Typed, validated, extensible search spaces       | `src/nas_engine/search_space/`  | [Search spaces](docs/concepts/search-spaces.md)                  |
| Deterministic architecture encoding and hashing  | `src/nas_engine/architectures/` | [Architecture encoding](docs/concepts/architecture-encoding.md)  |
| PyTorch model construction with shape validation | `src/nas_engine/models/`        | [Component design](docs/architecture/component-design.md)        |
| Random search, regularized evolution, halving    | `src/nas_engine/search/`        | [NAS foundations](docs/concepts/nas-foundations.md)              |
| Training, early stopping, checkpoints            | `src/nas_engine/training/`      | [Training](docs/concepts/training-and-evaluation.md)             |
| Multi-objective ranking and Pareto fronts        | `src/nas_engine/objectives/`    | [Multi-objective](docs/concepts/multi-objective-optimization.md) |
| Candidate lifecycle, retries, resume             | `src/nas_engine/orchestration/` | [System overview](docs/architecture/system-overview.md)          |
| SQLite persistence with versioned migrations     | `src/nas_engine/persistence/`   | [Persistence](docs/architecture/persistence.md)                  |
| Markdown reports, CSV and JSON exports, plots    | `src/nas_engine/reporting/`     | [Interpreting results](docs/guides/interpreting-results.md)      |
| A CLI and a Python API                           | `src/nas_engine/cli.py`         | [Getting started](docs/getting-started.md)                       |

## Install

Requires Python 3.10 or newer (3.12 is the reference version).

```bash
git clone https://github.com/example/neural-architecture-search
cd neural-architecture-search
make install          # editable install with development dependencies
make check            # format, lint, strict types, tests
make smoke            # a real end-to-end search, about 30 seconds on one CPU core
```

CIFAR-10 support is an optional extra, because it pulls in `torchvision`:

```bash
pip install -e ".[dev,cifar]"
```

## Five minutes

```bash
# 1. Is the environment sound?
nas-engine doctor

# 2. Create and check a configuration
nas-engine init --output configs/my-search.yaml
nas-engine validate-config --config configs/my-search.yaml

# 3. Run a search
nas-engine search --config configs/my-search.yaml --report

# 4. Look at the results
nas-engine status
nas-engine best
nas-engine pareto
nas-engine list-candidates --limit 20
nas-engine show-candidate <hash-prefix>

# 5. Score the winner on the held-out test split (exactly once)
nas-engine evaluate

# 6. Export
nas-engine export --format csv --output results.csv
```

Interrupted? Nothing is lost:

```bash
nas-engine resume --config configs/my-search.yaml
```

[docs/getting-started.md](docs/getting-started.md) walks through this in detail.

## Documentation

The documentation is a first-class deliverable, not an afterthought. It explains *why*
each design was chosen, what the alternatives were, and what was given up.

**Start here** — [docs/index.md](docs/index.md)

| Section                                                   | Pages | Contents                                                                |
| --------------------------------------------------------- | ----: | ----------------------------------------------------------------------- |
| [Concepts](docs/concepts/nas-foundations.md)              |    10 | The mathematics of NAS, search spaces, encoding, the three algorithms   |
| [Architecture](docs/architecture/system-overview.md)      |     6 | Overview, components, data flow, persistence, concurrency, security     |
| [Guides](docs/guides/running-a-search.md)                 |     8 | Running, resuming, extending, custom datasets, results, troubleshooting |
| [Testing](docs/testing/test-strategy.md)                  |     3 | Test strategy, the full test matrix, reproducibility testing            |
| [Operations](docs/operations/production-runbook.md)       |     4 | Deployment, observability, backup and recovery, the runbook             |
| [Decisions](docs/adr/0001-search-space-representation.md) |     4 | Architecture decision records, with alternatives and consequences       |
| [Glossary](docs/glossary.md)                              |     1 | Every term used in the codebase                                         |

## What this project will not tell you

NAS is surrounded by claims that do not survive contact with the details. The
documentation is explicit about the limits, and so is every generated report:

- **NAS does not find a globally optimal architecture.** It finds the best architecture it
  happened to evaluate, inside one search space, under one training recipe.
- **Validation accuracy from a short run is a noisy estimate.** With a validation split of
  *n* examples the standard error is roughly `sqrt(p(1-p)/n)`. Differences smaller than a
  few standard errors are not evidence.
- **A search selects on validation accuracy, so the winner's validation number is
  optimistically biased.** Only the held-out test split gives an unbiased estimate, and it
  must be used once.
- **Latency is machine-specific.** Numbers measured here are comparable between candidates
  on the same machine during the same run, and nowhere else.
- **A fair comparison between NAS methods includes the cost of the search itself.**
- **Search-space design frequently matters more than the search algorithm.**

[docs/concepts/common-pitfalls.md](docs/concepts/common-pitfalls.md) treats each of these
properly.

## Repository layout

```text
src/nas_engine/
├── architectures/   genotypes: specification, canonical form, hashing, shapes, cost
├── search_space/    what may be searched: choices, sampling, repair, mutation, validation
├── models/          phenotypes: PyTorch modules built from genotypes
├── datasets/        synthetic and CIFAR-10 providers, loaders, fidelity views
├── training/        optimisers, schedules, metrics, early stopping, checkpoints, the loop
├── evaluation/      budgets, latency, model size, the candidate evaluator, results
├── objectives/      objectives, constraints, scoring, Pareto fronts, ranking
├── search/          the strategy interface and the three implementations
├── orchestration/   the engine, the state machine, executors, retries, checkpoints
├── persistence/     database, versioned schema, the repository seam
├── reporting/       Markdown reports, CSV and JSON exports, figures
├── observability/   structured logging, events, context, counters
├── utilities/       seeding, environment capture, safe paths, hashing, timing
├── config/          validated configuration models and the precedence chain
├── exceptions.py    the error taxonomy
└── cli.py           the command-line interface

tests/               unit · property · integration · end-to-end · regression ·
                     performance · failure-recovery
docs/                concepts · architecture · guides · testing · operations · ADRs
configs/             ready-to-run configurations for each strategy
examples/            four runnable examples
scripts/             smoke search, benchmarks, report generation, CI helpers
```

[docs/repository-manifest.md](docs/repository-manifest.md) documents every file's purpose,
public symbols, dependencies, and tests.

## Development

```bash
make help              # every target
make format            # Ruff format and autofix
make lint              # Ruff lint
make typecheck         # mypy, strict
make test              # the default suite (excludes `slow`)
make test-all          # everything
make coverage          # coverage with the 90% line / 85% branch gate
make smoke             # a real search through the CLI
make examples          # run all four examples
make build             # wheel and sdist
make verify-package    # install the wheel into a clean virtualenv and run it
make docker-build      # container image
```

Tests never touch the network, never need a GPU, and never download CIFAR-10. That is
enforced mechanically in `tests/unit/test_public_api.py`.

## Containers

```bash
docker build -t nas-engine .
docker run --rm nas-engine doctor
docker run --rm -v "$PWD/artifacts:/data/artifacts" nas-engine smoke
docker compose run --rm search
```

The image runs as a non-root user, is CPU-only by default, and mounts `/data` for results.
GPU instructions are at the bottom of the [Dockerfile](Dockerfile).

## Status and stability

The public API is everything exported from `nas_engine/__init__.py`; it follows semantic
versioning. Everything else is internal and may change in a minor release. The boundary is
documented in that module and in
[docs/architecture/component-design.md](docs/architecture/component-design.md).

## Licence

MIT. See [LICENSE](LICENSE).
