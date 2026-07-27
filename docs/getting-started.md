# Getting started

From a clean checkout to a finished search with a report, in about five minutes on one CPU
core. No GPU, no network access, no dataset download.

## 1. Install

```bash
git clone https://github.com/sdivyanshu90/build-your-own-nas.git
cd build-your-own-nas
make install
```

That installs the package in editable mode with the development dependencies. Python 3.10
or newer is required; 3.12 is the reference version and the one CI treats as primary.

Verify the installation:

```bash
nas-engine doctor
```

```text
                                nas-engine doctor
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check              ┃ status ┃ detail                                            ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ python version     │ PASS   │ 3.12.3 (recommended)                              │
│ pytorch            │ PASS   │ 2.3.0                                             │
│ cuda               │ WARN   │ not available; searches will run on CPU           │
│ configuration      │ PASS   │ valid, hash 8ff22b54ce1c66b10831405e41181f8a      │
│ database           │ PASS   │ sqlite+pysqlite:///…/nas.db reachable             │
│ search space       │ PASS   │ 'default_cnn', about 1e21.2 architectures         │
└────────────────────┴────────┴───────────────────────────────────────────────────┘
```

`doctor` exits non-zero if any check fails, so it is usable as a pre-flight gate in a
script. A `WARN` is informational — no CUDA simply means the search runs on CPU.

## 2. Run the smoke search

```bash
make smoke
```

This runs [`scripts/run_smoke_search.sh`](../scripts/run_smoke_search.sh), which executes
a four-candidate random search on synthetic data and then exercises every inspection
command. It is the fastest way to confirm the whole system works, and it is what CI uses
as its end-to-end gate.

## 3. Understand what just happened

The smoke run used [`configs/smoke_test.yaml`](../configs/smoke_test.yaml). Every
configuration has the same thirteen sections:

```yaml
project: # name, description, where output goes
dataset: # which data, batch size, provider options
search_space: # which space to search
algorithm: # which strategy, and its parameters
budget: # how much compute the search may spend
training: # optimiser, schedule, clipping, early stopping
evaluation: # what to measure and store
objectives: # what to optimise, and any hard constraints
persistence: # database and artifact locations, checkpoint cadence
logging: # level, format, optional file
hardware: # device selection and thread limits
concurrency: # sequential or multiprocessing
reproducibility: # seed and determinism
retry: # retry policy for failed evaluations
```

Check a configuration before spending compute on it:

```bash
nas-engine validate-config --config configs/random_search.yaml
```

An invalid configuration names every offending field, the value it received, and the range
it expected:

```text
configuration from configs/broken.yaml is invalid:
  - budget.max_evaluations: Input should be greater than or equal to 1 (received 0)
  - epocs: Extra inputs are not permitted (received 3)
Fix the listed fields, or run 'nas-engine validate-config --config <file>' to check a
file before using it.
```

That second line is why `extra="forbid"` is set on every configuration model: a typo would
otherwise be silently ignored, and the default would apply while you believed your setting
had taken effect.

## 4. Run a real search

```bash
nas-engine search --config configs/random_search.yaml --report
```

Output:

```text
Search 7492f071596c4b7c8fe56a7e3d7f25af finished: the evaluation budget was fully spent
  status            : completed
  duration          : 41.3s
  proposed          : 20
  evaluated         : 20
  duplicates        : 0
  invalid           : 0
  pruned            : 3
  failed            : 0
  Pareto front size : 4

  best candidate    : 385cfb98d1da7ad7b5c1e202c730221f
    validation acc  : 0.6562
    parameters      : 112,650
    score           : 0.8830
```

Read the counters:

- **proposed** — architectures the strategy suggested.
- **duplicates** — proposals the engine rejected because that exact architecture had
  already been evaluated. A large number means the space is nearly exhausted.
- **invalid** — proposals that failed structural validation. Should normally be zero; a
  non-zero count means the sampler and the validator disagree, which is a bug.
- **pruned** — architectures that were structurally fine but exceeded a resource
  constraint. Nothing went wrong; the constraint did its job.
- **failed** — evaluations that raised. Investigate with
  `nas-engine list-candidates --state failed`.

## 5. Inspect the results

```bash
nas-engine status              # counts by lifecycle state
nas-engine best                # the winner, with its full layer table
nas-engine pareto              # the trade-off front
nas-engine list-candidates     # the leaderboard
```

`nas-engine show-candidate <prefix>` accepts an architecture-hash prefix and prints the
architecture and the full trial history, including any failures and retries:

```text
Architecture 385cfb98d1da7ad7b5c1e202c730221f
  input           : 3x32x32 (10 classes)
  stages          : 3
  blocks          : 6
  total stride    : 4 (final feature map 64x8x8)
  trainable params: 112,650
  MACs (per image): 18,241,536

  layer                    kind                      input         output
  ------------------------ ---------------- -------------- --------------
  stem                     conv                    3x32x32       16x32x32
  stages.0.blocks.0        conv                   16x32x32       32x16x16
  …
```

Every command accepts `--json` for machine-readable output:

```bash
nas-engine best --json | jq '.best.metrics'
```

## 6. Use the winner

Score it once on the held-out test split:

```bash
nas-engine evaluate
```

The test split is used _only_ here. Running this repeatedly on the same search
reintroduces the selection bias it exists to avoid — see
[common pitfalls](concepts/common-pitfalls.md#4-reusing-the-test-set-during-search-causes-leakage).

From Python, rebuild the model and run inference:

```python
from nas_engine import SearchConfig, SearchEngine
import torch

config = SearchConfig.from_yaml("configs/random_search.yaml")
engine = SearchEngine(config)
spec, model = engine.load_best_model()

model.eval()
with torch.no_grad():
    logits = model(torch.randn(8, spec.input_channels, spec.input_size, spec.input_size))
print(logits.shape)   # torch.Size([8, 10])
engine.close()
```

## 7. Read the report

`--report` wrote a Markdown report, a JSON export, a CSV export, and three figures into
`artifacts/random_search/reports/`. The report contains the configuration, the environment
that produced it, the leaderboard, the Pareto front, the winner's architecture, the
figures, the winner's lineage, and — always — the known limitations.

Regenerate a report at any time, including for a search that ran weeks ago:

```bash
nas-engine report --config configs/random_search.yaml
```

[Interpreting results](guides/interpreting-results.md) explains how to read it, and in
particular how to tell whether a difference between two candidates is real.

## 8. Interrupt and resume

Press Ctrl-C during a search. Then:

```bash
nas-engine resume --config configs/random_search.yaml
```

The engine finds candidates the crash left mid-evaluation, returns them to the queue,
restores the strategy's exact random-generator position from its checkpoint, and continues.
Nothing is replayed and nothing is lost. [Resuming a search](guides/resuming-a-search.md)
explains what is restored and what is not.

## 9. Try the other strategies

```bash
nas-engine search --config configs/evolution.yaml --report
nas-engine search --config configs/successive_halving.yaml --report
```

Compare them **at equal total compute**, not at equal evaluation count — successive halving
performs more evaluations at lower fidelity, which is the whole point. See
[NAS foundations](concepts/nas-foundations.md#11-comparing-methods-fairly).

## 10. Use real data

The shipped configurations use synthetic data so nothing needs downloading. To use
CIFAR-10:

```bash
pip install -e ".[cifar]"
```

Then replace the dataset block (the commented-out version is already in
[`configs/random_search.yaml`](../configs/random_search.yaml)):

```yaml
dataset:
  provider: cifar10
  batch_size: 128
  num_workers: 4
  options:
    root: data/cifar10
    download: true # opt in explicitly; nothing downloads without this
    validation_samples: 5000
    augment: true
    seed: 42
```

`download: true` is required. No code path in this project reaches the network unless a
human sets that flag.

## Where to go next

- [Running a search](guides/running-a-search.md) — configuration in depth.
- [NAS foundations](concepts/nas-foundations.md) — the theory.
- [Defining search spaces](guides/defining-search-spaces.md) — the highest-leverage thing
  you can change.
- [Common pitfalls](concepts/common-pitfalls.md) — read before trusting a result.
- [Troubleshooting](guides/troubleshooting.md) — when something goes wrong.

Four runnable examples are in [`examples/`](../examples): `quickstart.py`,
`custom_search_space.py`, `custom_objective.py`, and `resume_search.py`. Run them all with
`make examples`.
