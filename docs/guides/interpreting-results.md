# Interpreting results

Reading a report, reading a Pareto front, and knowing when a difference is real.

## The report

`nas-engine report` produces four things in `<output_dir>/reports/`:

| File                         | Contents                                            |
| ---------------------------- | --------------------------------------------------- |
| `<search_id>_report.md`      | The human-readable report                           |
| `<search_id>_results.json`   | Everything, nested — for programmatic use           |
| `<search_id>_candidates.csv` | One row per candidate — for a spreadsheet or pandas |
| `plots/<search_id>_*.png`    | Three or four figures                               |

Filenames derive from the search id, never a timestamp, so regenerating overwrites in place.

## Reading it in order

### Headline

```markdown
The best architecture found is `385cfb98d1da7ad7b5c1e202c730221f`, reaching a validation
accuracy of **0.6562** with **112,650** trainable parameters.

This number is a *selected* validation score and is therefore optimistically biased; see
Known limitations.
```

That second sentence is not boilerplate. The search picked this candidate *because* it
scored highest on validation, so the score includes whatever luck contributed. The unbiased
number comes from `nas-engine evaluate`, used once.

### Search statistics

```markdown
| Metric | Value |
| Planned evaluations | 20 |
| Candidates proposed | 23 |
| Completed | 20 |
| Failed | 0 |
| Pruned (constraint) | 3 |
| Unique architectures ranked | 20 |
| Pareto-front size | 4 |
```

| Row                           | What a surprising value means                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Proposed ≫ completed          | Many rejections. Check pruned and failed                                                                                |
| Pruned high                   | A constraint is too tight. Raise it or narrow the space                                                                 |
| Failed non-zero               | Real failures. `nas-engine list-candidates --state failed`                                                              |
| Pareto front = 1              | One candidate dominates everything. Either it is genuinely best on all axes, or the secondary objectives have no spread |
| Pareto front ≈ all candidates | The objectives are nearly independent, or the population is tiny                                                        |

### Configuration and environment

The configuration excerpt shows the sections that change what the search *means*. The
environment section shows what produced the numbers:

```markdown
| PyTorch | 2.3.0 |
| CUDA available | False |
| Devices | cpu |
| Torch threads | 4 |
| Determinism caveat | intra-op parallelism is enabled (4 threads); reduction order in some CPU kernels depends on the thread count |
```

Latency numbers are only comparable against other numbers from this same environment.

### Leaderboard

```markdown
| Rank | Architecture | Accuracy | Parameters | Latency (ms) | Score | Front |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `385cfb98d1da` | 0.6562 | 112,650 | 0.412 | 0.8830 | 0 |
| 1 | `4712175a11a3` | 0.6484 | 89,204 | 0.388 | 0.8641 | 0 |
| 2 | `82f0f1b32eec` | 0.5938 | 12,940 | 0.201 | 0.7102 | 0 |
| 3 | `1ff67f67a59b` | 0.5000 | 1,756 | 0.094 | 0.1667 | 1 |
```

The **Front** column is the one to read first. Front 0 is the Pareto front: those candidates
are not dominated by anything. Front 1 and above are dominated — some front-0 candidate
beats them on *every* objective.

The **Score** column depends on your weights and is population-relative, so it is only
comparable within this table. See
[multi-objective optimisation](../concepts/multi-objective-optimization.md#the-population-relative-trap).

### Pareto front

Every member is a real option; the choice between them requires a preference the search
cannot supply. From the table above:

- `385cfb98` — the most accurate, at 112 650 parameters.
- `82f0f1b3` — 6 points less accurate, at **9× fewer** parameters.

Neither is "better". On a server, the first. On a microcontroller, the second — and the
first is not even installable.

### Best architecture

```markdown
| Layer | Kind | Input | Output |
| `stem` | conv | `3x32x32` | `16x32x32` |
| `stages.0.blocks.0` | conv | `16x32x32` | `32x16x16` |
| `stages.0.blocks.1` | dw_sep_conv | `32x16x16` | `32x16x16` |
| `stages.1.blocks.0` | conv | `32x16x16` | `64x8x8` |
| `head.pool` | global_avg | `64x8x8` | `64x1x1` |
| `head.classifier` | linear | `64x1x1` | `10x1x1` |
```

Worth checking:

- **Does the resolution reduce sensibly?** A total stride that leaves a 1×1 feature map
  discarded all spatial structure before pooling.
- **Do the widths grow?** They should, if `monotonic_widths` is on.
- **Are there operations you did not expect?** A winner made mostly of identity blocks means
  the search found that depth does not help — informative, and often a sign the task is too
  easy.

### Figures

**Accuracy versus model size** (log x-axis). Points on the Pareto front are joined; the best
is circled. The shape of the front is the useful part: a steep front means accuracy is cheap
to buy with parameters; a flat one means it is not.

**Accuracy versus latency.** Same reading, with the machine-specificity caveat printed
beneath it.

**Search progress.** Individual evaluations as points, the running best as a line. A curve
that rises steeply then flattens means additional budget is not buying much. A curve still
climbing at the end means the search was cut short.

### Lineage

For evolutionary searches, the winner's ancestry:

```text
a1b2c3d4 (objective=0.5312)
    └── e5f6a7b8 [kernel_size s0b1: 3 -> 5] (objective=0.5938)
        └── 385cfb98 [stage_width s1: 32 -> 64] (objective=0.6562)
```

Which mutations mattered. If every improvement came from one operator, the others are not
earning their place in the operator set.

### Known limitations

Always present, always the same seven statements. They are properties of NAS, not of your
run, and omitting them would make the headline look stronger than it is.

---

## Is this difference real?

The single most important question, and the one most often skipped.

### Finite-sample noise

$$
\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}
$$

With $n = 1000$ and $p = 0.65$:

$$
\mathrm{SE} = \sqrt{\frac{0.65 \times 0.35}{1000}} = 0.0151
$$

So candidate 0 at 0.6562 and candidate 1 at 0.6484 differ by 0.0078 — **half a standard
error**. That is not evidence of anything.

For a difference to be meaningful at roughly 95% confidence you want at least $2\sqrt{2}
\times \mathrm{SE} \approx 0.043$ — four percentage points on this split.

### Training noise

On top of that, the same architecture trained twice with different seeds reaches different
accuracies. On short budgets this is often *larger* than the sampling noise.

Measure it:

```bash
for seed in 1 2 3; do
  nas-engine search --config configs/my-search.yaml \
    --set budget.max_evaluations=1 --set reproducibility.seed=$seed \
    --set project.output_dir=/tmp/noise-$seed
done
```

The spread across those three runs is the resolution of your measurement. Differences
smaller than it mean nothing, regardless of the sampling arithmetic.

### A practical rule

**Treat the top few candidates as tied** unless the gap between them exceeds both noise
sources. When they are tied on accuracy, choose on the secondary objectives — that is what
the Pareto front is for, and it is a much more defensible basis for a decision than a
difference you cannot measure.

---

## From the command line

```bash
nas-engine best                    # the winner, with its layer table
nas-engine pareto                  # the trade-off front
nas-engine list-candidates --limit 20
nas-engine show-candidate 385cfb98 # one candidate, with its full trial history
nas-engine status                  # counts by lifecycle state
```

Machine-readable:

```bash
nas-engine best --json | jq '.best.metrics'
nas-engine pareto --json | jq '.pareto_front[] | {hash: .architecture_hash, metrics: .metrics}'
nas-engine list-candidates --json | jq '[.candidates[] | .metrics.validation_accuracy]'
```

## From the CSV

```python
import pandas as pd

df = pd.read_csv("artifacts/my-search/reports/<search_id>_candidates.csv")

# the accuracy/size trade-off among front members
front = df[df.on_pareto_front == 1].sort_values("trainable_parameters")
print(front[["architecture_hash", "validation_accuracy", "trainable_parameters"]])

# is accuracy actually related to size in this space?
print(df[["validation_accuracy", "trainable_parameters"]].corr(method="spearman"))

# how much of the space did the search cover?
print(f"{df.architecture_hash.nunique()} unique architectures")
```

That correlation is worth computing. If accuracy and size are uncorrelated in your space,
the parameter objective is free — you can have small models at no accuracy cost. If they
are strongly correlated, every parameter you save costs accuracy, and the front's slope
tells you the exchange rate.

---

## Re-ranking without re-running

The ranking is recomputed from persisted metrics, so you can change your mind about the
objectives and regenerate:

```bash
nas-engine report --config configs/my-search.yaml \
  --set objectives.objectives[1].weight=2.0     # note: list indexing is not supported
```

List indexing is not supported by `--set`, so edit the configuration file and regenerate:

```bash
nas-engine report --config configs/size-focused.yaml --search-id <id>
```

Or in Python, which is more flexible:

```python
from nas_engine import SearchRepository, rank_candidates
from nas_engine.objectives import Objective, ObjectiveDirection, ObjectiveSet
from nas_engine.persistence import Database, ensure_schema

database = Database("sqlite+pysqlite:///artifacts/my-search/nas.db")
ensure_schema(database)
repository = SearchRepository(database)
population = repository.completed_metrics("<search_id>")

size_first = ObjectiveSet((
    Objective(metric="trainable_parameters", direction=ObjectiveDirection.MINIMIZE, weight=2.0),
    Objective(metric="validation_accuracy", direction=ObjectiveDirection.MAXIMIZE, weight=1.0),
))
for candidate in rank_candidates(population, size_first).ranked[:5]:
    print(candidate.architecture_hash[:12], candidate.metrics)
database.dispose()
```

[`examples/custom_objective.py`](../../examples/custom_objective.py) ranks one set of
results four different ways and shows the winner change.

---

## The final test evaluation

```bash
nas-engine evaluate
```

```text
Test evaluation of 385cfb98d1da7ad7b5c1e202c730221f
 test_accuracy       0.6354
 test_examples     1024.0000
 test_loss           1.0912
 test_topk_accuracy  0.9531
```

**Expect the test accuracy to be lower than validation.** The gap is the selection bias, and
seeing it is healthy. A large gap — several standard errors — means the search overfitted
the validation split, which happens with a small split and many candidates.

Use this **once**. Running it repeatedly and picking the best result reintroduces exactly
the bias it exists to measure.

---

## A checklist

Before believing a result:

- [ ] Does the accuracy difference exceed the validation split's standard error?
- [ ] Does it exceed the training-noise spread across seeds?
- [ ] Did random search on the same space at the same budget do worse?
- [ ] Is the Pareto front more than one candidate?
- [ ] Did the progress curve flatten, or was the search cut short?
- [ ] Are the failed and pruned counts explained?
- [ ] Was the test split used exactly once?
- [ ] Is the winner's architecture sensible when you look at its layer table?

## See also

- [Common pitfalls](../concepts/common-pitfalls.md) — the misreadings this page guards
  against.
- [Multi-objective optimisation](../concepts/multi-objective-optimization.md) — what a
  front means.
- [Troubleshooting](troubleshooting.md) — when the numbers look wrong.
