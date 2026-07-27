# Running a search

Configuration in depth, choosing a strategy, sizing a budget, and monitoring progress.

## The configuration

Every configuration has the same thirteen sections. Start from a shipped one rather than
from scratch:

```bash
cp configs/random_search.yaml configs/my-search.yaml
nas-engine validate-config --config configs/my-search.yaml
```

Or scaffold one:

```bash
nas-engine init --output configs/my-search.yaml --strategy regularized_evolution
```

### Precedence

Four layers, later winning:

1. Built-in defaults — every field has one, so an empty file works.
2. The YAML file.
3. Environment variables — `NAS_ENGINE__SECTION__FIELD`.
4. Command-line overrides — `--set section.field=value`.

```bash
nas-engine search --config configs/my-search.yaml \
  --set budget.max_evaluations=50 \
  --set hardware.device=cpu
```

```bash
NAS_ENGINE__BUDGET__MAX_EVALUATIONS=50 nas-engine search --config configs/my-search.yaml
```

Merging is **deep**, so overriding `training.optimizer.learning_rate` leaves
`training.optimizer.weight_decay` alone. Values are parsed with YAML rules, so `3` is an
integer, `true` a boolean, and `[1, 2]` a list.

Use environment variables for deployment-specific values (device, worker count, paths) and
`--set` for one-off experiments. Anything you want to reproduce belongs in the file.

---

## Section by section

### `project`

```yaml
project:
  name: my-search              # persisted; `resume` finds the latest search with this name
  description: What and why.   # appears in the report
  output_dir: artifacts/mine   # database, artifacts, and reports go here
```

Use a distinct `name` per experiment. `nas-engine resume` without `--search-id` finds the
most recent search with a matching name, so reusing a name across unrelated experiments
makes resume ambiguous.

### `dataset`

```yaml
dataset:
  provider: synthetic     # or cifar10, or your own registered provider
  batch_size: 64
  num_workers: 0
  pin_memory: false
  drop_last: false
  options:                # passed straight to the provider
    num_classes: 10
    input_size: 32
    train_samples: 4096
    validation_samples: 1024
    noise_scale: 0.8
    seed: 42
```

`num_workers: 0` is the right default here: for small datasets, spawning worker processes
costs more than it saves and adds nondeterminism. Raise it for CIFAR-10 with augmentation.

**The validation split size determines your measurement resolution.** With 1 000 examples
the standard error is about 1.1 percentage points, so differences below ~2 points are noise.
Size it for the differences you need to detect. See
[common pitfalls](../concepts/common-pitfalls.md#3-validation-accuracy-is-a-noisy-estimate).

### `search_space`

```yaml
search_space:
  preset: default_cnn
  overrides:                     # merged over the preset
    stage_channels: [16, 32, 64]
    constraints:
      max_parameters: 500000
```

`nas-engine validate-config` prints the resulting space, including its size:

```text
Search space 'default_cnn' (schema v1)
  stages          : [2, 3]
  stage widths    : [16, 32, 64] (monotonic)
  operations      : ['conv', 'dw_sep_conv', 'identity', 'max_pool', 'avg_pool']
  approx. size    : 1e18.4 architectures (upper bound)
```

See [defining search spaces](defining-search-spaces.md).

### `algorithm`

```yaml
algorithm:
  name: random_search       # or regularized_evolution, successive_halving
  params: {}                # strategy-specific
```

### `budget`

```yaml
budget:
  max_evaluations: 20               # total candidates
  max_seconds: null                 # wall-clock limit for the whole search
  epochs: 5                         # per candidate (rung 0 for successive halving)
  train_fraction: 1.0
  resolution: null                  # null means the dataset's native size
  max_seconds_per_evaluation: 900   # a per-candidate guard
```

Always set `max_seconds_per_evaluation`. Without it, one pathological architecture can
consume the entire wall-clock budget.

### `training`

```yaml
training:
  optimizer:
    name: adamw
    learning_rate: 0.002
    weight_decay: 0.0001
    decay_normalization: false    # exempt norm and bias parameters
  scheduler:
    name: cosine
    warmup_steps: 20
    min_lr_factor: 0.01
  gradient_clip_norm: 5.0
  label_smoothing: 0.05
  early_stopping_patience: 0      # 0 disables
  early_stopping_min_delta: 0.0
  mixed_precision: false          # CUDA only
  topk: 5
  restore_best_weights: true
  zero_init_residual: true
```

**One recipe applies to every candidate.** That is the point: the search compares
architectures, not hyperparameters. It also means the ranking is a ranking *under this
recipe* — see
[common pitfalls](../concepts/common-pitfalls.md#9-a-discovered-architecture-may-overfit-the-dataset-and-the-training-recipe).

AdamW is the default because it is forgiving of architectural variation. With SGD, the
learning rate is likely to suit some architectures much better than others, and the search
will partly be ranking learning-rate compatibility.

### `evaluation`

```yaml
evaluation:
  measure_latency: true
  latency_batch_size: 1
  latency_warmup_iterations: 3
  latency_timed_iterations: 10
  latency_repeats: 5
  measure_model_size: true
  save_weights: true
  save_training_checkpoints: false   # resumable per-candidate training; costs disk
  max_parameters: 2000000            # a hard ceiling, checked before building
```

Set `measure_latency: false` when running with multiple workers — contention makes the
numbers meaningless.

Set `save_training_checkpoints: true` for long per-candidate training, where losing an
interrupted candidate's progress is expensive.

### `objectives`

```yaml
objectives:
  objectives:
    - metric: validation_accuracy
      direction: maximize
      weight: 1.0
      normalization: minmax
    - metric: trainable_parameters
      direction: minimize
      weight: 0.2
      normalization: log
  constraints:
    - metric: trainable_parameters
      operator: le
      threshold: 2000000
```

See [multi-objective optimisation](../concepts/multi-objective-optimization.md), and note
in particular that with `minmax`/`log` normalisation the secondary objectives shape the
*final ranking* but not the evolutionary trajectory.

### `persistence`, `logging`, `hardware`, `concurrency`, `reproducibility`, `retry`

```yaml
persistence:
  database_path: nas.db        # relative to output_dir
  artifact_dir: candidates
  report_dir: reports
  checkpoint_every: 1          # checkpoint after every N completed evaluations
  keep_checkpoints: 5

logging:
  level: INFO
  format: console              # or json
  file: null                   # optional, relative to output_dir

hardware:
  device: auto                 # auto | cpu | cuda | cuda:0 | mps
  torch_threads: null

concurrency:
  mode: sequential             # or multiprocessing
  workers: 1
  start_method: spawn

reproducibility:
  seed: 42
  deterministic: true
  warn_only: true

retry:
  max_retries: 1
  retry_on_timeout: true
  retry_on_resource_error: true
  backoff_seconds: 0.0
```

`device: auto` prefers CUDA, then Apple MPS, then CPU. An explicitly requested accelerator
that is unavailable is an **error**, not a silent CPU fallback: a run that quietly takes a
hundred times longer than expected is worse than one that refuses to start.

---

## Choosing a strategy

| Situation                              | Strategy                | Why                                                  |
| -------------------------------------- | ----------------------- | ---------------------------------------------------- |
| First run on a new space               | `random_search`         | The baseline. You need it to interpret anything else |
| Budget under ~20 evaluations           | `random_search`         | Evolution's initialisation would eat the budget      |
| Budget 30–200, large space             | `regularized_evolution` | Enough to fill a population and run generations      |
| Expensive evaluations, many candidates | `successive_halving`    | Buys many more samples from the same compute         |
| Small space (under a few thousand)     | `random_search`         | Adaptive search has no room to help                  |
| Very noisy evaluation                  | `random_search`         | Adaptive methods fit the noise                       |

**Always run random search first, at the budget you plan to spend.** Then run the fancier
method at *the same total compute*. If it does not clearly win, the extra machinery is not
earning its keep.

---

## Sizing the budget

Estimate the cost first:

```text
total ≈ max_evaluations × epochs × (seconds per epoch)
```

Measure one epoch:

```bash
nas-engine search --config configs/my-search.yaml \
  --set budget.max_evaluations=1 --set budget.epochs=1 \
  --set project.output_dir=/tmp/probe
```

Then read `training_seconds` from the result and multiply.

For successive halving the arithmetic differs — `budget.epochs` is the **rung-0** budget:

```text
total ≈ num_rungs × initial_candidates × epochs_at_rung_0 × (seconds per epoch)
```

### Evaluations versus epochs

Given a fixed total, spending it on more candidates or on more epochs each is a real
trade-off:

- **More candidates, fewer epochs** — better coverage, noisier ranking, biased towards
  fast-converging architectures.
- **Fewer candidates, more epochs** — trustworthy measurements, very little of the space
  seen.

A reasonable starting point is enough epochs for the *worst* architecture in the space to
get past its initial transient — often 3–5 on small images — and then as many candidates as
the budget allows.

---

## Monitoring

Watch the log:

```bash
nas-engine search --config configs/my-search.yaml
```

```text
2026-07-27T09:06:14.617837Z [info  ] search.started        concurrency=sequential device=cpu
                                     max_evaluations=20 search_id=99365970… seed=42 strategy=random_search
2026-07-27T09:06:14.628258Z [info  ] candidate.proposed    architecture_hash=385cfb98… mutation=None
                                     origin=random rung=0 search_id=99365970…
2026-07-27T09:06:14.650858Z [info  ] evaluation.started    attempt=0 budget=1 epochs, limit 120s
                                     candidate_id=80a89eb1… trial_id=b2131bc0…
2026-07-27T09:06:16.160590Z [info  ] evaluator.completed   duration_seconds=1.509 trainable_parameters=5564.0
                                     validation_accuracy=0.2604 worker_id=main
2026-07-27T09:06:16.189411Z [info  ] evaluation.completed  duration_seconds=1.509 objective_value=0.2604
                                     validation_accuracy=0.2604 worker_id=main
2026-07-27T09:06:16.245307Z [info  ] checkpoint.saved      search_id=99365970… sequence=1
```

Real lines are one per row and carry every context field; they are wrapped and trimmed here
to fit the page. `evaluator.completed` is the evaluator's own inner record of the attempt —
see [observability](../operations/observability.md#module-diagnostics-are-a-separate-namespace)
for why it is a separate name.

Inspect from another terminal while it runs — WAL mode makes concurrent reads work:

```bash
watch -n 10 'nas-engine status --config configs/my-search.yaml'
nas-engine list-candidates --config configs/my-search.yaml --limit 10
```

For machine consumption:

```bash
nas-engine search --config configs/my-search.yaml --set logging.format=json 2> run.jsonl
jq -c 'select(.event == "evaluation.completed") | {hash: .architecture_hash, acc: .validation_accuracy}' run.jsonl
```

### What to watch for

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `duplicates` climbing | The space is nearly exhausted | Widen the space, or stop |
| `pruned` climbing | A search-space constraint is too tight | Raise `search_space.overrides.constraints.max_parameters` or `.max_multiply_accumulates` |
| `failed` non-zero | Real failures | `nas-engine list-candidates --state failed` |
| Every accuracy near chance | Learning rate wrong, or too few epochs | Raise `budget.epochs`; check with one candidate |
| All accuracies identical | Insufficient budget to discriminate | Raise `budget.epochs` or the validation split |
| Very slow evaluations | Architectures too large | Lower `search_space.overrides.constraints.max_parameters` |

---

## After the search

```bash
nas-engine status
nas-engine best
nas-engine pareto
nas-engine report --config configs/my-search.yaml
nas-engine evaluate                                # the test split, once
nas-engine export --format csv --output results.csv
```

See [interpreting results](interpreting-results.md).

---

## Running several searches

Comparing strategies:

```bash
for strategy in random_search regularized_evolution successive_halving; do
  nas-engine search --config configs/my-search.yaml \
    --set algorithm.name=$strategy \
    --set project.name=compare-$strategy \
    --set project.output_dir=artifacts/compare-$strategy
done
```

Give each a distinct `output_dir` so they get separate databases; a shared one works too
but makes `resume` ambiguous.

Measuring statistical repeatability — which is what actually matters:

```bash
for seed in 1 2 3 4 5; do
  nas-engine search --config configs/my-search.yaml \
    --set reproducibility.seed=$seed \
    --set project.name=seed-$seed \
    --set project.output_dir=artifacts/seed-$seed
done
```

Compare **distributions**, not single best values. If five seeds of one method span three
percentage points, a two-point difference between methods means nothing.

---

## Parallel execution

```yaml
concurrency:
  mode: multiprocessing
  workers: 4
  start_method: spawn
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nas-engine search --config configs/my-search.yaml
```

Set the thread variables. Otherwise each worker spawns its own BLAS thread pool and *N*
workers × *M* threads oversubscribe the machine badly.

Also set `evaluation.measure_latency: false` — contention makes latency meaningless — and
understand that [multiprocessing gives up bit-level reproducibility](../architecture/concurrency.md#what-concurrency-changes-the-honest-answer).

---

## From Python

```python
from nas_engine import SearchConfig, SearchEngine

config = SearchConfig.from_yaml("configs/my-search.yaml")
engine = SearchEngine(config)
try:
    result = engine.run()
    print(result.summary())
    for candidate in result.pareto_front:
        print(candidate.architecture_hash, candidate.metrics)
finally:
    engine.close()
```

Every collaborator can be injected, which is what makes the engine testable and scriptable:

```python
engine = SearchEngine(
    config,
    dataset=my_bundle,             # skip the provider registry
    database=Database.in_memory(), # nothing touches disk
    strategy=MyStrategy(...),      # skip the strategy registry
    configure_process=False,       # do not reconfigure global logging
)
```

## See also

- [Resuming a search](resuming-a-search.md)
- [Interpreting results](interpreting-results.md)
- [Troubleshooting](troubleshooting.md)
- [Production runbook](../operations/production-runbook.md)
