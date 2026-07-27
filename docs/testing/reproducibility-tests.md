# Reproducibility tests

What is asserted to be deterministic, what is deliberately not, and why.

## The guarantees

For **sequential CPU execution**, given the same configuration, seed, dataset, library
versions, and machine:

| Guarantee                      | Test                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| Candidate proposal order       | `test_proposal_order_is_reproducible`                        |
| Architecture hashes            | `test_two_identical_sequential_searches_agree_exactly`       |
| Mutation decisions             | `test_mutation_decisions_are_reproducible`                   |
| Search-strategy state          | `test_strategy_state_is_reproducible`                        |
| Evolution's population         | `test_evolution_population_is_reproducible`                  |
| Initial weights                | `test_weights_are_reproducible_for_a_given_seed`             |
| Measured metrics               | `test_metrics_are_reproducible_within_one_machine`           |
| Final candidate ranking        | `test_two_identical_sequential_searches_agree_exactly`       |
| Pareto front membership        | same                                                         |
| Resume reaching the same state | `test_resume_reaches_the_same_state_as_an_uninterrupted_run` |

All in [`tests/regression/test_determinism.py`](../../tests/regression/test_determinism.py).

## What each test actually asserts

### Proposal order

```python
def run() -> list[str]:
    sampler = ArchitectureSampler(space, seed=2024)
    return [architecture_hash(sampler.sample()) for _ in range(15)]

assert run() == run()
```

Fifteen consecutive draws produce an identical hash sequence. Catches any accidental
dependency on the global RNG, on dictionary iteration order, or on `PYTHONHASHSEED`.

### Mutation decisions

```python
def run() -> list[tuple[str, str]]:
    mutator = MutationOperator(space, seed=99)
    current, trace = parent, []
    for _ in range(10):
        result = mutator.mutate(current)
        trace.append((result.operator, result.description))
        current = result.child
    return trace

assert run() == run()
```

Both the *operator chosen* and the *change it made* are identical across runs. A weaker test
asserting only the resulting hashes would miss an operator-selection change that happened to
produce the same child.

### Evolution's population

Twelve generations with a deterministic pseudo-fitness, no training at all. Isolates the
selection and aging logic from the noise of real evaluation, which is what makes this test
useful rather than flaky.

### Weights

```python
seed_everything(31)
first = {k: v.clone() for k, v in build_model(spec).state_dict().items()}
seed_everything(31)
second = build_model(spec).state_dict()
assert all(torch.equal(first[k], second[k]) for k in first)
```

Bit-identical initialisation for the same seed — and a companion test asserts that a
*different* seed produces different weights, so the first test cannot pass trivially.

### A whole search

```python
first  = run(tmp_path / "a")     # seed 4242
second = run(tmp_path / "b")     # seed 4242
assert first["hashes"] == second["hashes"]
assert first["accuracies"] == second["accuracies"]      # exact float equality
assert first["parameters"] == second["parameters"]
assert first["best"] == second["best"]
assert first["pareto"] == second["pareto"]
```

Exact float equality on the accuracies. On a single machine with a fixed library set, this
holds. It would not hold across machines, which is exactly why the test title says *within
one machine*.

### Resume

The most valuable test in the file:

```python
whole  = run(max_evaluations=4)                          # uninterrupted
first  = run(max_evaluations=2)                          # split, part 1
second = resume(first.search_id, max_evaluations=4)      # split, part 2
assert sorted(hashes(second)) == sorted(hashes(whole))
```

An uninterrupted four-evaluation run and the same run split in two reach **the same set of
architectures**. That only holds if the strategy's generator state is checkpointed and
restored exactly — re-seeding would make the second half replay the first.

---

## What is deliberately not asserted

Each of these is a decision, not a gap.

### Latency values

Latency depends on machine load, which nobody controls. Asserting a value would produce a
test that fails when CI is busy — and a flaky test in the determinism suite would train
people to ignore determinism failures.

`test_latency_is_not_asserted_to_be_reproducible` documents the decision by asserting the
warning text exists, so a future change that adds a latency equality assertion has the
reasoning available.

### Bit-identical results across machines

Different CPUs use different SIMD widths; different GPUs use different reduction trees;
different BLAS builds use different blocking. Floating-point addition is not associative, so
the results differ in the last bits. No amount of seeding changes that.

The environment snapshot exists precisely so this is visible rather than mysterious.

### Identical results under multiprocessing

Individual candidate results **are** identical — each candidate's seed comes from its
architecture hash. What varies is the *order* in which observations arrive, and an adaptive
strategy that sees a different sequence can propose differently.

The tests assert bit-identity only for sequential runs. That is an accurate statement of
what is true.

### Results across library versions

A PyTorch release can change a kernel's default algorithm. The environment snapshot records
the version.

---

## The determinism report

`configure_determinism` reports honestly what it achieved:

```python
report = configure_determinism(enabled=True, warn_only=True)
report.to_dict()
```

```json
{
  "requested": true,
  "deterministic_algorithms": true,
  "cudnn_deterministic": true,
  "cudnn_benchmark": false,
  "cublas_workspace_config": ":4096:8",
  "warnings": [
    "intra-op parallelism is enabled (4 threads); reduction order in some CPU kernels depends on the thread count"
  ]
}
```

The report is persisted with the search and printed in the report's environment table, so a
result carries its own determinism caveats.

`test_the_determinism_report_lists_its_caveats` asserts the warnings field exists and is
populated.

---

## Running them

```bash
make test-regression
pytest tests/regression/test_determinism.py -v
pytest -m determinism
```

They take about six seconds, because they run several complete searches. That is worth
paying on every commit: a determinism regression is exactly the kind of bug that goes
unnoticed for months.

## Verifying reproducibility yourself

Two runs with the same seed:

```bash
for run in a b; do
  nas-engine search --config configs/random_search.yaml \
    --set project.output_dir=artifacts/repro-$run \
    --set reproducibility.seed=42
  nas-engine export --config configs/random_search.yaml \
    --set project.output_dir=artifacts/repro-$run \
    --format csv --output artifacts/repro-$run.csv
done

diff <(cut -d, -f3,9 artifacts/repro-a.csv) <(cut -d, -f3,9 artifacts/repro-b.csv) \
  && echo "identical"
```

The candidate ids differ (they are UUIDs), so compare the architecture hashes and the
metrics.

## Measuring statistical repeatability

Reproducibility is about repeating one run. **Statistical repeatability** is about whether
the *conclusion* survives a different seed — and that is what actually matters
scientifically.

```bash
for seed in 1 2 3 4 5; do
  nas-engine search --config configs/evolution.yaml \
    --set reproducibility.seed=$seed \
    --set project.output_dir=artifacts/seed-$seed
done
```

Compare distributions, not single best values. A useful rule: **the spread across seeds is a
lower bound on the difference you can detect.** If five seeds of the same method span three
percentage points, a two-point difference between methods means nothing.

The framework supports this; it does not do it for you, and nothing about a single seeded
run tells you whether its conclusion is robust.

## See also

- [Reproducibility](../concepts/reproducibility.md) — the seeding scheme in full.
- [Concurrency](../architecture/concurrency.md) — what parallelism costs.
- [Common pitfalls](../concepts/common-pitfalls.md#10-reproducibility-requires-more-than-setting-one-random-seed).
