# Troubleshooting

Every error this project raises, what causes it, and what to do.

## Start here

```bash
nas-engine doctor --config configs/my-search.yaml
```

Checks Python and package versions, device availability, configuration validity, directory
permissions, database access, and the search space. Exits non-zero if anything fails, so it
works as a pre-flight gate in a script.

For more detail:

```bash
nas-engine search --config configs/my-search.yaml --set logging.level=DEBUG
```

---

## Configuration errors

### `configuration file not found`

```text
configuration file not found: configs/typo.yaml. Create one with 'nas-engine init', or
pass --config with a valid path.
```

Check the path. Paths are relative to the current directory, not to the configuration file.

### `Extra inputs are not permitted`

```text
configuration from configs/my.yaml is invalid:
  - epocs: Extra inputs are not permitted (received 3)
```

A typo. Unknown keys are rejected deliberately: silently ignoring them would leave the
default in place while you believed your setting had taken effect. Compare against a shipped
configuration for the correct spelling.

### `Input should be greater than or equal to 1`

```text
  - budget.max_evaluations: Input should be greater than or equal to 1 (received 0)
```

Out of range. The message names the field, the value, and the bound.

### `mixed_precision requires a CUDA device`

Set `training.mixed_precision: false`, or `hardware.device: auto`. Mixed precision with loss
scaling is CUDA-only; on CPU the trainer logs a fallback rather than failing, but the
*configuration* combination is rejected because it is almost always a mistake.

### `hardware.device='cuda' was requested but CUDA is not available`

Deliberately an error rather than a silent CPU fallback: a run that quietly takes a hundred
times longer than expected is worse than one that refuses to start. Use `device: auto`.

### `search_space.preset=... is not a known preset`

The message lists the available ones. Register a custom space in `PRESETS` before use — see
[defining search spaces](defining-search-spaces.md).

### `override ... is not of the form key.path=value`

```bash
nas-engine search --set budget.max_evaluations 20      # wrong: missing '='
nas-engine search --set budget.max_evaluations=20      # right
```

### `the stored configuration is version N but this build supports at most version M`

The database was written by a newer nas-engine. Upgrade, or use a different database file.

---

## Search-space errors

### `failed to sample a valid architecture in N attempts`

```text
Most common rejection reasons: [('constraint:multiply_accumulates', 200)].
Relax the search-space constraints (max_parameters, max_multiply_accumulates,
min_final_resolution) or widen the choice sets.
```

The named constraint is rejecting everything. Either raise it, or narrow the choice sets so
infeasible candidates are not generated in the first place — the second is better, because
it stops wasting draws.

Diagnose interactively:

```python
from nas_engine import get_preset
from nas_engine.search_space import ArchitectureSampler

space = get_preset("default_cnn")
sampler = ArchitectureSampler(space, seed=1)
for _ in range(50):
    sampler.try_sample()
print(sampler.statistics.to_dict())
```

### `constraints.max_parameters=N is below the minimum possible parameter count`

Caught before the search starts. Raise the ceiling or narrow `stage_channels`.

### `no stage width satisfies the monotonic-width rule`

A corrupt search space — `stage_channels` is empty after de-duplication. Check the
override.

### `no valid mutation found for architecture ... in N attempts`

```text
Rejections by operator: {'kernel_size': 8, 'operation': 12, ...}.
Widen the search space or relax its constraints.
```

Usually a space with one choice per dimension: there is nothing to mutate *to*. The
rejection counts say which operators had nothing to offer.

---

## Architecture errors

### `operation 'max_pool' cannot change the channel count`

```text
stages.1.blocks.0: operation 'max_pool' cannot change the channel count, but
out_channels=64 while the block receives 32 channels. Set out_channels=32, or use 'conv' /
'dw_sep_conv' to change width.
```

Only convolutions can change width. If you built the architecture by hand, use
`repair_architecture` to fix it automatically. If this came from the sampler, it is a bug —
please report it.

### `use_residual=True requires the block input and output to have identical shapes`

Only identity shortcuts are supported. Set `stride=1` and match the channel counts, or set
`use_residual=False`.

### `... reduces the feature map below 1x1`

Too many stride-2 blocks for the input size. Reduce `stage_strides`, reduce `num_stages`, or
increase `input_size`. Set `min_final_resolution: 2` to prevent it.

### `architecture payload failed validation`

Imported JSON is invalid. The message names every offending field. Common causes: an
operation name not in the enumeration, a value out of range, or an extra key.

### `architecture schema_version=2 is newer than the supported version 1`

The architecture was written by a newer nas-engine. Upgrade.

---

## Training and evaluation errors

### `loss became non-finite`

```text
loss became non-finite (nan) at epoch 0, step 12. This architecture is numerically unstable
under the current recipe; lower the learning rate or enable gradient clipping.
```

**Not retried** — the same seed and architecture would diverge again.

| Cause                                | Fix                                        |
| ------------------------------------ | ------------------------------------------ |
| Learning rate too high               | Lower `training.optimizer.learning_rate`   |
| Gradient clipping off                | Set `training.gradient_clip_norm: 5.0`     |
| Very deep, unnormalised architecture | Remove `none` from `block.normalizations`  |
| A pathological architecture          | Expected occasionally; the search moves on |

One divergence in a hundred candidates is normal. Half of them diverging means the recipe is
wrong.

### `training exceeded its Ns budget`

Retriable. Either the architecture is genuinely slow, or the machine is loaded. Raise
`budget.max_seconds_per_evaluation`, lower `budget.epochs`, or lower
`search_space.constraints.max_parameters`.

### `the training loader yields no batches`

`batch_size` exceeds the training split with `drop_last: true`. Lower the batch size or set
`drop_last: false`.

### `model has no trainable parameters`

The architecture is entirely pooling and identity operations. Set
`search_space.constraints.min_parameters` to exclude them.

### `architecture has N trainable parameters which exceeds the evaluation limit`

`evaluation.max_parameters` is enforced *before* the model is built, from the analytic cost
model. Raise it, or tighten the search space so such candidates are not proposed.

---

## Checkpoint errors

### `checkpoint at ... could not be read`

```text
the file is corrupt or truncated. Delete it to restart this candidate from scratch.
```

Usually an interrupted write. Deleting the file loses only that candidate's training
progress.

### `checkpoint belongs to architecture X but Y was expected`

A file was moved or renamed. Delete it; the candidate retrains from scratch.

### `this checkpoint was written by the 'X' strategy but the resume is configured to use 'Y'`

Strategy state is not interchangeable. Restore the original `algorithm.name`, or start a new
search.

### `checkpoint was written with population_size=X but the current configuration uses Y`

Changing the population size across a resume would change the aging schedule and invalidate
the comparison. Restore the original value, or start a new search.

### `search checkpoint is missing required fields`

A corrupt row. Delete the most recent checkpoint and resume — checkpoints are append-only,
so an earlier one is still there:

```sql
DELETE FROM checkpoints WHERE search_id = '…'
  AND sequence = (SELECT MAX(sequence) FROM checkpoints WHERE search_id = '…');
```

---

## Persistence errors

### `database is locked`

Another process holds the write lock. The busy timeout is 30 seconds, so this means either a
very long transaction or a stale lock from a crashed process.

```bash
lsof artifacts/my-search/nas.db      # who has it open?
```

If nothing does, remove the WAL sidecar files:

```bash
rm -f artifacts/my-search/nas.db-wal artifacts/my-search/nas.db-shm
```

### `unable to open database file`

The directory does not exist or is not writable. `nas-engine doctor` checks this.

### `the database is at schema version N but this build supports at most version M`

Upgrade nas-engine, or point at a different database. Downgrading a schema is not supported
and would lose data.

### `candidate ... already exists in search ...`

Not an error in normal operation — the engine treats it as a duplicate and moves on. Seeing
it as an *exception* means something bypassed the engine.

### `no search found with id ...`

Check with `nas-engine status --json`, or list them:

```python
for summary in repository.list_searches():
    print(summary.id, summary.name, summary.status)
```

---

## Orchestration errors

### `cannot move a candidate from 'completed' to 'queued'`

A state-machine violation. `completed` is terminal. If this happens during normal operation
it is a bug — please report it with the log.

### `no search was found to resume`

Nothing in the database yet. Start one with `nas-engine search`.

### `worker process failed while evaluating candidate ...`

Retriable. Common causes: the worker was OOM-killed; the configuration is invalid in a way
only the worker discovers; a dependency is missing in the worker environment.

Reproduce sequentially, where the error is not truncated by the process boundary:

```bash
nas-engine search --config configs/my-search.yaml --set concurrency.mode=sequential
```

### `could not start a process pool with N workers`

```text
Set concurrency.mode='sequential' to run in-process.
```

Usually a resource limit. Lower `workers`.

---

## Results that look wrong

### Every candidate scores at chance

| Cause                | Check                              | Fix                                       |
| -------------------- | ---------------------------------- | ----------------------------------------- |
| Learning rate wrong  | Does the training loss fall?       | Adjust `training.optimizer.learning_rate` |
| Too few epochs       | Is `epochs` 1?                     | Raise it                                  |
| Labels not learnable | Does a linear probe beat chance?   | Fix the dataset                           |
| Wrong class count    | Does `num_classes` match the data? | Fix it                                    |

Isolate it with one candidate:

```bash
nas-engine search --config configs/my-search.yaml \
  --set budget.max_evaluations=1 --set budget.epochs=10 --set logging.level=DEBUG
```

If a single candidate cannot learn with ten epochs, the problem is the data or the recipe,
not the search.

### Every candidate scores identically

The training budget is too small to discriminate, or the space is too narrow. Raise `epochs`
first.

### The Pareto front has one member

One candidate dominates on every objective. Either that is genuinely true, or the secondary
objectives have no spread — check with:

```python
costs = [c.metrics["trainable_parameters"] for c in result.ranked]
print(min(costs), max(costs))
```

Less than a factor of two means the parameter objective has nothing to do.

### Accuracy is much lower than expected

- Is `epochs` high enough? NAS budgets are deliberately short.
- Is the validation split large enough to measure accurately?
- Is augmentation on for CIFAR-10? Without it, models overfit quickly.
- Is the architecture actually reasonable? Check `nas-engine best`.

### Latency numbers are wildly inconsistent

- Multiple workers contending. Set `concurrency.mode: sequential` for latency measurements.
- Other load on the machine.
- Too few repeats. Raise `evaluation.latency_repeats`.

Check the p99-to-median ratio: a large gap means contention, not a slow model.

### Duplicates keep appearing

The space is nearly exhausted. Widen it, or accept that further budget buys nothing.

```python
space = get_preset("default_cnn")
print(f"upper bound: 1e{space.log10_cardinality():.1f}")
```

---

## Performance problems

### The search is very slow

Where the time goes, in order:

1. **Training.** Lower `epochs`, lower `max_parameters`, or use successive halving.
2. **Latency measurement.** Lower `latency_repeats`, or set `measure_latency: false`.
3. **Model construction.** Only matters for very large architectures.
4. **Persistence.** Rarely the bottleneck. Measure with `scripts/benchmark.py`.

### Memory grows over the run

- `save_training_checkpoints: true` writes one file per candidate per rung. Disk, not
  memory, but it adds up.
- Strategy state grows with the number of seen hashes. Bounded and small.
- If *process* memory grows, that is a leak — please report it with the configuration.

### The database file is large

Checkpoints dominate: each holds the strategy state, including every seen hash. Lower
`persistence.keep_checkpoints`, or prune:

```sql
DELETE FROM checkpoints WHERE search_id = '…' AND sequence < (
    SELECT MAX(sequence) - 5 FROM checkpoints WHERE search_id = '…'
);
VACUUM;
```

---

## Getting help

Include:

1. `nas-engine doctor --json`
2. The configuration (redact anything sensitive)
3. The full error, including the structured `details`
4. `nas-engine status --json` if a search exists
5. What you expected instead

Every error carries a machine-readable `code` and structured `details` — include them, they
are the most useful part.

## See also

- [Running a search](running-a-search.md)
- [Resuming a search](resuming-a-search.md)
- [Production runbook](../operations/production-runbook.md)
- [Interpreting results](interpreting-results.md)
