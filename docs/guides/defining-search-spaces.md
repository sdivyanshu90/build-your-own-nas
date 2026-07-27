# Defining search spaces

The highest-leverage thing you can change.

## Three ways

### 1. Use a preset

```yaml
search_space:
  preset: default_cnn
```

### 2. Override a preset

```yaml
search_space:
  preset: default_cnn
  overrides:
    stage_channels: [16, 32, 64]
    num_stages: [3, 4]
    constraints:
      max_parameters: 500000
      min_final_resolution: 4
```

Overrides are merged over the preset's fields and the result is fully validated, so a typo
or an out-of-range value fails immediately.

### 3. Build one in Python

```python
from nas_engine.search_space import SearchSpace
from nas_engine.search_space.space import BlockChoices, HeadChoices, SpaceConstraints, StemChoices
from nas_engine.architectures.types import ActivationType, NormalizationType, OperationType, PoolingType

space = SearchSpace(
    name="my_space",
    input_channels=3, input_size=32, num_classes=10,
    num_stages=(2, 3, 4),
    blocks_per_stage=(1, 2),
    stage_channels=(16, 24, 32, 48),
    stage_strides=(1, 2),
    monotonic_widths=True,
    block=BlockChoices(
        operations=(OperationType.DW_SEP_CONV, OperationType.IDENTITY, OperationType.AVG_POOL),
        kernel_sizes=(3, 5),
        expansion_ratios=(1.0, 3.0, 6.0),
        normalizations=(NormalizationType.BATCH, NormalizationType.GROUP),
        activations=(ActivationType.RELU6, ActivationType.HARDSWISH),
        allow_residual=True,
    ),
    stem=StemChoices(out_channels=(8, 16), kernel_sizes=(3,), strides=(1, 2)),
    head=HeadChoices(poolings=(PoolingType.AVG,), hidden_units=(0, 96), dropouts=(0.0, 0.2)),
    constraints=SpaceConstraints(
        max_parameters=300_000,
        min_parameters=2_000,
        max_multiply_accumulates=60_000_000,
        min_final_resolution=2,
        max_depth=8,
    ),
)
```

To use it from configuration, register it as a preset:

```python
from nas_engine.search_space.presets import PRESETS
PRESETS["my_space"] = lambda **kwargs: space.model_copy(update=kwargs)
```

[`examples/custom_search_space.py`](../../examples/custom_search_space.py) is a complete
worked version of this.

---

## The workflow

### Step 1: inspect what you built

```python
print(space.describe())
```

```text
Search space 'my_space' (schema v1)
  input           : 3x32x32 -> 10 classes
  stages          : [2, 3, 4]
  blocks per stage: [1, 2]
  stage widths    : [16, 24, 32, 48] (monotonic)
  operations      : ['dw_sep_conv', 'identity', 'avg_pool']
  approx. size    : 1e18.6 architectures (upper bound)
```

Or from the CLI:

```bash
nas-engine validate-config --config configs/my-search.yaml
```

### Step 2: check it is feasible

```python
space.require_non_empty()
```

Catches the most common misconfiguration — a parameter ceiling below the minimum any
architecture can reach:

```text
constraints.max_parameters=10 is below the minimum possible parameter count of roughly 322
for this space; no candidate can ever be feasible. Raise max_parameters or narrow
stage_channels.
```

### Step 3: sample from it

```python
from nas_engine.architectures import summarise
from nas_engine.search_space import ArchitectureSampler, validate_architecture

sampler = ArchitectureSampler(space, seed=42)
for _ in range(10):
    spec = sampler.sample()
    validate_architecture(spec, space)
    print(summarise(spec).compact())
```

```text
a1b2c3d4 | 3 stages | 5 blocks | 0.12M params | 8.4M MACs | stride 4
e5f6a7b8 | 2 stages | 3 blocks | 0.04M params | 3.1M MACs | stride 2
```

### Step 4: check the acceptance rate

```python
print(sampler.statistics.to_dict())
```

```python
{'attempts': 12, 'accepted': 10, 'rejected': 2, 'acceptance_rate': 0.83,
 'rejection_reasons': {'constraint:multiply_accumulates': 2}}
```

**Below about 50%, a constraint is fighting the sampler.** The rejection reasons say which
one. Either relax it, or narrow the choice sets so infeasible candidates are not generated
in the first place.

### Step 5: check the spread

```python
from nas_engine.architectures.cost import compute_cost

costs = [compute_cost(sampler.sample()).trainable_parameters for _ in range(50)]
print(f"min {min(costs):,}  median {sorted(costs)[25]:,}  max {max(costs):,}")
```

```text
min 3,142  median 48,910  max 287,554
```

Two orders of magnitude is healthy: the search has something to trade off. If everything is
within a factor of two, the parameter objective has nothing to do.

---

## Design principles

### Start small

A space of $10^6$ that you understand beats $10^{30}$ that you do not. Widen once you can
see which dimensions matter — which you learn by looking at the top candidates from a first
run and asking what they have in common.

### Include a known-good architecture

If a hand-designed baseline is not a member of your space, the search cannot match it, and
you will not know whether a poor result is the space or the algorithm.

Check by constructing the baseline as an `ArchitectureSpec` and validating it:

```python
from nas_engine.search_space import check_architecture

report = check_architecture(my_baseline, space)
print(report.summary())      # "architecture is valid", or every reason it is not
```

### Set `min_parameters`

Without it, degenerate all-pooling networks train fast, score near chance, and clutter the
Pareto front's low-parameter end. They are technically non-dominated and completely useless.

### Prefer few dimensions with many values

Random search covers `stage_channels: [8, 16, 24, 32, 48, 64]` far better than six binary
flags. High-dimensional binary spaces are where random search actually struggles.

### Watch the duplicate count

A high duplicate count in a search means the space is nearly exhausted and more budget will
buy nothing:

```bash
nas-engine status --json | jq '.counts'
```

---

## Common mistakes

### The space is too large to sample usefully

**Symptom.** Twenty samples from $10^{30}$, all wildly different, no pattern in the results.

**Fix.** Narrow the choice sets. You cannot cover a space that large with any budget you
have; a smaller space you can actually explore gives more usable information.

### Constraints fight the sampler

**Symptom.** Acceptance rate below 30%, or `SearchSpaceError: failed to sample a valid
architecture in 200 attempts`.

**Fix.** Read the rejection reasons. Usually the choice sets can produce architectures the
constraints reject — for instance `stage_channels` up to 512 with `max_parameters: 100000`.
Narrow the choices rather than raising the ceiling, so the budget is not spent generating
candidates that will be thrown away.

### Every candidate scores the same

**Symptom.** Zero spread in validation accuracy.

**Causes, in order of likelihood.**

1. The training budget is too small to discriminate. Raise `epochs`.
2. The space is too narrow — every member is essentially the same network.
3. The task is too easy or too hard. For the synthetic dataset, tune `noise_scale`.

### Monotonic widths exclude what you wanted

**Symptom.** Hourglass or autoencoder shapes never appear.

**Fix.** `monotonic_widths: false`. Understand what you are giving up: the rule removes a
large region that is empirically unproductive for classification, so turning it off makes
the space larger without necessarily making it better.

### The final feature map collapses to 1×1

**Symptom.** Poor accuracy across the board; `total_stride` equal to or above `input_size`.

**Fix.** Set `min_final_resolution: 2` or higher. A 1×1 feature map discards all spatial
structure before pooling, which is legal but usually accidental.

---

## Recipes

### A mobile-style space

Small, deployable, latency-aware.

```yaml
search_space:
  preset: default_cnn
  overrides:
    stage_channels: [8, 16, 24, 32]
    blocks_per_stage: [1, 2]
    block:
      operations: [dw_sep_conv, identity, avg_pool]
      kernel_sizes: [3, 5]
      expansion_ratios: [1.0, 3.0, 6.0]
      activations: [relu6, hardswish]
      normalizations: [batch, group]
    constraints:
      max_parameters: 300000
      max_multiply_accumulates: 30000000
```

Group normalisation is included because a deployed model may run at batch size 1, where
batch statistics are unusable.

### A depth-focused space

Fixed width; the search decides only how deep and with what operations.

```yaml
search_space:
  preset: default_cnn
  overrides:
    stage_channels: [32]
    num_stages: [2, 3, 4]
    blocks_per_stage: [1, 2, 3, 4]
```

Small enough to explore thoroughly, and it answers one clean question.

### An operation-comparison space

Everything fixed except the operation.

```yaml
search_space:
  preset: default_cnn
  overrides:
    num_stages: [3]
    blocks_per_stage: [2]
    stage_channels: [32]
    stage_strides: [2]
    block:
      kernel_sizes: [3]
      expansion_ratios: [1.0]
      normalizations: [batch]
      activations: [relu]
```

A few hundred members — small enough to enumerate almost exhaustively, so the comparison is
close to a controlled experiment.

---

## Validating an imported architecture

Architectures from another source are untrusted input. Validate before using:

```python
from nas_engine.architectures.canonical import from_canonical_json
from nas_engine.search_space import check_architecture

spec = from_canonical_json(Path("candidate.json").read_text())   # rejects anything malformed
report = check_architecture(spec, space)
if not report.is_valid:
    for issue in report.issues:
        print(f"{issue.category}: {issue}")
```

Membership can be skipped when the architecture is deliberately from a different space and
only needs to be buildable:

```python
report = check_architecture(spec, space, check_space_membership=False)
```

## See also

- [Search spaces](../concepts/search-spaces.md) — the concepts.
- [Adding an operation](adding-an-operation.md) — extending the vocabulary.
- [ADR 0001](../adr/0001-search-space-representation.md) — why this representation.
