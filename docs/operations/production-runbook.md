# Production runbook

Procedures for running, watching, and rescuing a real search.

Each entry is written to be followed under pressure: symptom, diagnosis, action.

---

## Starting a search

```bash
# 1. Pre-flight
nas-engine doctor --config configs/production.yaml
nas-engine validate-config --config configs/production.yaml
df -h artifacts/

# 2. Prove the pipeline on a tiny budget first
nas-engine search --config configs/production.yaml \
  --set budget.max_evaluations=2 \
  --set project.output_dir=artifacts/preflight
rm -rf artifacts/preflight

# 3. Run
nas-engine search --config configs/production.yaml --report
```

`validate-config` prints the resolved configuration and the search space, including
`log10_cardinality` — how many architectures the space contains. If that number is smaller
than your evaluation budget, the search will exhaust the space and spend the rest of the
budget rejecting duplicates.

Step 2 costs a minute and catches the failures that would otherwise surface an hour in: a
missing dataset, an unwritable directory, a search space where every sample violates a
constraint. Multiply its wall-clock by `budget.max_evaluations` for a rough total.

---

## Checking on a running search

```bash
nas-engine status --config configs/production.yaml
```

```text
    Search 7492f071596c4b7c8fe56a7e3d7f25af
 name         smoke_test
 strategy     random_search
 status       completed
 seed         42
 config hash  2c9fac85083782f0881c77f5d70f1478
 created      2026-07-27T07:57:43.758482+00:00
 started      2026-07-27T07:57:43.771076+00:00
 completed    2026-07-27T07:58:13.835642+00:00
 duration     30.1s
 checkpoints  3
 Candidates by state
┏━━━━━━━━━━━┳━━━━━━━┓
┃ state     ┃ count ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ proposed  │     0 │
│ validated │     0 │
│ queued    │     0 │
│ running   │     0 │
│ completed │     4 │
│ failed    │     0 │
│ pruned    │     0 │
│ cancelled │     0 │
└───────────┴───────┘
```

Read it as: is `completed` advancing, is `failed` small, is `pruned` explicable? On a
running search, `running` should equal your worker count and `queued` should be non-zero —
if both are zero while the search is still going, something is stuck.

The `config hash` is worth noting: it identifies the exact configuration this search ran
under, so two searches with the same hash are comparable and two with different hashes are
not.

```bash
watch -n 60 'nas-engine status --config configs/production.yaml'
nas-engine best --config configs/production.yaml
nas-engine pareto --config configs/production.yaml
```

---

## Stopping a search

### Cleanly

**Ctrl-C once.** The engine finishes the in-flight evaluation, writes a checkpoint, and
exits with `search.interrupted`. Resume later with no loss.

Ctrl-C twice forces an immediate exit. The in-flight evaluation is lost; the recovery sweep
cleans it up on resume.

### Under a scheduler

Set `budget.max_seconds` below the wall-clock limit so the search stops itself. A search
killed by the scheduler still recovers, but a clean stop wastes nothing.

### Permanently

Just stop it. The database holds every completed result; run `nas-engine report` to write
the artefacts for what did finish. A partial search is still a result.

---

## Resuming

```bash
nas-engine resume --config configs/production.yaml
```

The engine loads the latest checkpoint, sweeps `RUNNING` candidates back to `QUEUED`,
reconciles against the database, and continues. Resuming twice is harmless.

Without `--search-id`, `resume` picks the most recent search whose project name matches the
configuration. When one directory holds several, name the one you mean:

```bash
sqlite3 artifacts/nas.db \
  "SELECT id, name, strategy, status, created_at FROM searches ORDER BY created_at DESC LIMIT 10;"

nas-engine resume --config configs/production.yaml --search-id 7492f071596c4b7c
```

`--search-id` works the same way on `status`, `best`, `pareto`, `list-candidates`,
`show-candidate`, `export`, `report`, and `evaluate`.

---

## Symptom: the search is slower than expected

**Diagnose**

```bash
nas-engine status --config configs/production.yaml       # rate of completion
nas-engine list-candidates --config configs/production.yaml --limit 20
top -b -n 1 | head -20                                   # is anything running?
```

**Common causes, in the order worth checking**

| Cause                          | Check                                | Fix                                                      |
| ------------------------------ | ------------------------------------ | -------------------------------------------------------- |
| Thread oversubscription        | `nproc` vs `workers`                 | `OMP_NUM_THREADS=1` per worker                           |
| Data loading is the bottleneck | GPU idle, CPU pegged                 | Raise `dataset.num_workers`, enable `dataset.pin_memory` |
| Batch size too small           | Very fast steps, low utilisation     | Raise `dataset.batch_size`                               |
| Models larger than expected    | `params` column of `list-candidates` | Tighten the space or the constraints                     |
| Weight saving dominates        | Large `.pt` files, many candidates   | `evaluation.save_weights: false`                         |

Thread oversubscription is the most common and the least obvious: four workers each
spawning `nproc` BLAS threads produces four times the threads the machine has, and
everything slows down. **One rule: `workers × OMP_NUM_THREADS ≤ physical cores.**

**Act**

```bash
# Ctrl-C, then resume with corrected settings.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nas-engine resume --config configs/production.yaml \
    --set concurrency.workers=4
```

Resume honours the new settings. The completed work is kept.

---

## Symptom: many candidates are failing

**Diagnose**

```bash
nas-engine list-candidates --config configs/production.yaml --state failed --limit 50
nas-engine show-candidate 385cfb98 --config configs/production.yaml
```

`show-candidate` prints the architecture *and* every trial attempt with its error message,
which is usually enough on its own.

Every failed attempt stores a structured failure record in `trials.error_json`, with a
coarse `kind`, a stable `code`, and the message. Group by kind to see the shape of the
problem:

```sql
SELECT json_extract(t.error_json, '$.kind')    AS kind,
       json_extract(t.error_json, '$.code')    AS code,
       COUNT(*)                                AS n,
       MIN(json_extract(t.error_json, '$.message')) AS example
FROM trials t
JOIN candidates c ON c.id = t.candidate_id
WHERE c.search_id = '7492f071596c4b7c8fe56a7e3d7f25af'
  AND t.status IN ('failed', 'timeout')
GROUP BY kind, code
ORDER BY n DESC;
```

`trials` has no `search_id` of its own — a trial belongs to a candidate, and the candidate
belongs to the search. The join is what keeps that relationship honest.

**Act by failure kind**

| `kind`        | Meaning                                       | Retriable | Action                                                     |
| ------------- | --------------------------------------------- | --------- | ---------------------------------------------------------- |
| `resource`    | Out of memory                                 | yes       | Lower `dataset.batch_size` or `concurrency.workers`        |
| `timeout`     | Exceeded `budget.max_seconds_per_evaluation`  | yes       | Raise the limit, or lower `budget.epochs`                  |
| `divergence`  | The loss became non-finite                    | no        | Lower the learning rate; set `training.gradient_clip_norm` |
| `build`       | The model could not be constructed            | no        | A search-space bug. Reproduce it — see below               |
| `validation`  | The architecture was invalid                  | no        | A search-space or mutation bug                             |
| `constraint`  | Violated a `search_space` resource constraint | no        | Expected; see "everything is being pruned" below           |
| `training`    | Training failed for a recoverable reason      | yes       | Read the message; often data or device                     |
| `persistence` | The result could not be stored                | yes       | Disk full, or permissions on the output directory          |
| `worker`      | The worker process died                       | yes       | Usually the OOM killer. Lower `concurrency.workers`        |
| `unknown`     | Anything not otherwise classified             | no        | Read the message and traceback; reproduce sequentially     |

The `retriable` column is not advice — it is the `retriable` attribute of the exception
that was raised, and it is what the retry policy acts on. The reasoning is uniform: a
`divergence` is deterministic, so retrying it burns budget to reach the same NaN, while an
out-of-memory may well succeed once another worker finishes and frees memory.

Two settings can *withhold* a retry that the policy would otherwise allow —
`retry.retry_on_timeout` and `retry.retry_on_resource_error`, both `true` by default. They
only ever subtract. Nothing in the configuration can make a non-retriable failure retriable,
because that judgement belongs to the code that knows why it failed.

One wrinkle worth knowing: `resource` covers two different things. A genuine
out-of-memory is retriable, but exceeding `evaluation.max_parameters` raises a
`ResourceLimitError` that is not — it is a deterministic arithmetic check, and it would
fail identically every time.

A single failing candidate is normal — a search space is *supposed* to contain
architectures that do not work. **Above 20%, the recipe is wrong, not the architectures.**

Reproduce one in isolation. Pull the spec out of the database and run it through the
evaluator directly, sequentially, at `DEBUG`:

```python
from pathlib import Path

from nas_engine import ModelBuilder, SearchEngine
from nas_engine.config.loader import load_config
from nas_engine.evaluation.evaluator import EvaluationContext

config = load_config(Path("configs/production.yaml"))
engine = SearchEngine(config)
try:
    record = engine.repository.get_candidate("<candidate-id>")
    spec = engine.repository.get_candidate_spec(record.id)

    bundle = engine.dataset
    # Does it even build? A construction failure stops here, with a real traceback.
    ModelBuilder(
        input_shape=(bundle.input_channels, bundle.input_size, bundle.input_size),
        num_classes=bundle.num_classes,
    ).build(spec)

    # Does it train? In-process, so nothing swallows the traceback.
    result = engine.evaluator.evaluate(
        spec,
        config.budget.build_budget(),
        EvaluationContext(candidate_id=record.id, trial_id="manual"),
    )
    print(result.succeeded, result.metrics, result.failure)
finally:
    engine.close()
```

`evaluate` does not raise for candidate-level problems — it classifies them and returns
them on the result, which is why `result.failure` is where the diagnosis lives. Running
in-process and sequentially gives you a readable failure instead of interleaved worker
output.

---

## Symptom: everything is being pruned

```text
completed 3   pruned 41
```

**Diagnose**

The pruning reason is on the event, so ask the database what it said:

```sql
SELECT json_extract(payload_json, '$.reason') AS reason, COUNT(*)
FROM search_events
WHERE search_id = '…' AND event = 'candidate.pruned'
GROUP BY reason ORDER BY 2 DESC;
```

Then compare what the space actually produces against the constraint that is rejecting it:

```python
from nas_engine.architectures.cost import compute_cost
from nas_engine.search_space import ArchitectureSampler, get_preset

sampler = ArchitectureSampler(get_preset("default_cnn"), seed=0)
sizes = sorted(compute_cost(sampler.sample()).trainable_parameters for _ in range(200))
print(f"p5={sizes[10]:,}  median={sizes[100]:,}  p95={sizes[190]:,}")
```

Usually `search_space.overrides.constraints.max_parameters` sits below the 5th percentile of
what the space produces.

**Act**

Either raise the constraint or shrink the space (fewer stages, fewer blocks, narrower
widths). Raising the constraint is usually right: a constraint that excludes 90% of the
space is not guiding the search, it is wasting it.

### Three different limits, three different outcomes

They are easy to confuse, and they fail at different times:

| Setting                                | When it applies                | Result                                     |
| -------------------------------------- | ------------------------------ | ------------------------------------------ |
| `search_space.overrides.constraints.*` | At proposal, analytically      | `PRUNED` — costs microseconds              |
| `evaluation.max_parameters`            | At evaluation, before training | `FAILED` with `ResourceLimitError`         |
| `objectives.constraints`               | At ranking, after completion   | Marked infeasible; excluded from the front |

**Prefer the search-space constraint.** It rejects before any work is done, and a pruned
candidate is a normal outcome rather than a failure. `evaluation.max_parameters` is a
backstop against building something that will not fit in memory; a search that trips it
regularly has its limits in the wrong layer.

---

## Symptom: everything is a duplicate

```text
candidate.duplicate rate 78%
```

The space is nearly exhausted, or the strategy has converged onto a small region.

**Act**

- Random search: the space is too small. Enlarge it, or stop — you have seen most of it.
- Evolution: raise `algorithm.params.population_size`, lower
  `algorithm.params.tournament_size` (less selection pressure means more exploration), or
  raise `algorithm.params.mutation_attempts` so a proposal tries harder to find a novel
  child before falling back to a random draw.

The engine already handles duplicates correctly — it does not re-evaluate them. This is a
signal about the search, not a bug.

---

## Symptom: the search appears stalled

No `evaluation.completed` for far longer than an evaluation should take.

**Diagnose**

```bash
nas-engine status --config configs/production.yaml    # are any RUNNING?
ps aux | grep nas-engine
py-spy dump --pid <pid>                               # if py-spy is available
```

**Act**

If a worker is genuinely wedged, Ctrl-C twice and resume — the recovery sweep requeues the
interrupted candidate.

Then prevent it: set `budget.max_seconds_per_evaluation`. Without it a pathological
architecture can run indefinitely. **With it, the same architecture fails in bounded time
and the search continues.** This is the single most valuable setting for an unattended run.

---

## Symptom: accuracy is not improving

The search runs, nothing fails, and the best value plateaus early.

**Check, in this order**

1. **Is the budget enough to distinguish candidates?** Two epochs on CIFAR-10 measures
   noise. Look at the accuracy spread — if every candidate is within a point of every
   other, the budget is too small to rank them.
2. **Is the space diverse enough?** `nas-engine validate-config --json` reports
   `search_space.log10_cardinality`. A space of a few hundred points is exhausted quickly.
3. **Is the strategy exploring?** For evolution, check the lineage depth — a shallow tree
   means mutations are not being accepted.
4. **Is the recipe the limit rather than the architecture?** Train the best-found and a
   random architecture at full budget. If they land in the same place, architecture is not
   what is limiting accuracy, and no search will fix that.

Point 4 is the one people skip, and it is the one that most often explains a disappointing
search.

---

## Symptom: results are not reproducible

**Check**

Every search records the environment it ran in. Compare the two:

```bash
sqlite3 -json artifacts/nas.db \
  "SELECT environment_json FROM searches WHERE id = '7492f071596c4b7c';" | jq -r '.[0].environment_json' | jq
```

The report's environment table shows the same snapshot, rendered. Look at the PyTorch
version, the device, the thread counts, and the determinism report's `warnings`.

**Expected differences**

- **Different machine** — results will differ in the last decimals. This is arithmetic, not
  a bug.
- **Multiprocessing** — individual candidates match, proposal order can differ.
- **Different PyTorch** — kernels change.

**Unexpected**

Same machine, same versions, sequential, same seed, different results. That is a real bug.
File it with the two environment snapshots and the configuration.

See [reproducibility tests](../testing/reproducibility-tests.md) for exactly what is
guaranteed.

---

## Symptom: `database is locked`

Two processes are writing the same database.

```bash
lsof artifacts/nas.db
fuser -v artifacts/nas.db
```

One search per database file. To run several at once, give each its own `output_dir`.

Reading while a search runs is fine — WAL allows concurrent readers. `nas-engine status`
during a search is safe.

---

## Symptom: disk full

See [backup and recovery](backup-and-recovery.md#the-disk-filled-up).

The short version: delete `reports/` (regenerable), delete non-Pareto weights, `VACUUM` the
database, then lower `persistence.keep_checkpoints` so it does not happen again.

---

## After the search

```bash
nas-engine report --config configs/production.yaml
nas-engine export --config configs/production.yaml --format json --output results.json
nas-engine best --config configs/production.yaml
```

Then back up the database. The report can always be regenerated; the database cannot.

**Export the winning genotype**, so the architecture outlives the search directory:

```python
import json
from pathlib import Path

from nas_engine import SearchEngine, rank_candidates
from nas_engine.architectures.canonical import to_canonical_json
from nas_engine.config.loader import load_config

config = load_config(Path("configs/production.yaml"))
engine = SearchEngine(config)
try:
    search_id = engine.repository.find_latest_search(name=config.project.name).id
    ranking = rank_candidates(
        engine.repository.completed_metrics(search_id),
        config.objectives.build_objectives(),
        constraints=config.objectives.build_constraints(),
    )
    spec = engine.repository.get_candidate_spec(ranking.best.candidate_id)
    Path("best.json").write_text(to_canonical_json(spec))
finally:
    engine.close()
```

Canonical JSON, not pickle: it is readable, diffable, and safe to hand to another process.

**Retrain the winner properly.** A search budget is a ranking signal, not a final result.
The accuracy in the report is what that architecture achieved in five epochs, and it is not
the number to publish.

```python
from pathlib import Path

from nas_engine import ModelBuilder
from nas_engine.architectures.canonical import from_canonical_json

spec = from_canonical_json(Path("best.json").read_text())
model = ModelBuilder(input_shape=(3, 32, 32), num_classes=10).build(spec)
# ... train to convergence with your production recipe
```

`from_canonical_json` validates: unknown fields, out-of-range values, and enum members
outside the closed vocabulary are all rejected. A `best.json` from an untrusted source
cannot execute anything.

**Validate on the held-out test set once**, with `nas-engine evaluate`:

```bash
nas-engine evaluate --config configs/production.yaml
```

Every accuracy in the search is a *validation* accuracy, and the search selected on it — so
it is optimistically biased. The test set is the unbiased estimate, and it stops being
unbiased the moment you use it to choose anything. Run this once, at the end, and do not
let its number feed back into a configuration change.

---

## Escalation

| Situation                                        | Response                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| A search failed and resume also fails            | Capture `doctor` output, the config, and the last 100 log lines; restore from backup |
| Database corruption                              | [Recovery procedure](backup-and-recovery.md#the-database-is-corrupt)                 |
| Reproducibility broken on identical environments | File a bug with both environment snapshots                                           |
| Results look implausible                         | Check for label leakage and for a validation split that overlaps training            |

The last row deserves attention. An implausibly good result is more often a leak than a
discovery — and it is much cheaper to check than to publish.

## See also

- [Deployment](deployment.md)
- [Observability](observability.md)
- [Backup and recovery](backup-and-recovery.md)
- [Troubleshooting](../guides/troubleshooting.md) — every exception, with its cause.
- [Common pitfalls](../concepts/common-pitfalls.md) — the conceptual failures behind several
  of these symptoms.
