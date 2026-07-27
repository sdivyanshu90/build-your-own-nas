# Reproducibility

What this project promises, what it does not, and why the distinction matters.

## Three different words

They are used interchangeably in conversation and mean different things.

| Term | Definition | Promised here? |
| --- | --- | --- |
| **Reproducibility** | Same code, same configuration, same seed, same environment → same *decisions*: the same architectures proposed, in the same order, ranked the same way | **Yes**, for sequential execution |
| **Determinism** | Same inputs → bit-identical floating-point outputs | **On one machine**, with one library set, in sequential mode. Never across machines |
| **Statistical repeatability** | Different seeds → conclusions that agree in distribution: "evolution beats random search on this space" holds up | The framework enables measuring this; it cannot promise it |

Most published NAS work that claims "reproducible" means the first. Almost nothing achieves
the second across hardware. The third is the one that actually matters scientifically, and
it requires running with several seeds — which the framework supports but does not do for
you.

## Why one seed is not enough

`random.seed(42)` seeds exactly one generator. A NAS run touches at least six independent
sources of randomness:

| Source                          | Controlled by                            |
| ------------------------------- | ---------------------------------------- |
| Python `random`                 | `random.seed`                            |
| NumPy's legacy global generator | `numpy.random.seed`                      |
| PyTorch CPU RNG                 | `torch.manual_seed`                      |
| PyTorch CUDA RNGs (all devices) | `torch.cuda.manual_seed_all`             |
| DataLoader worker processes     | `worker_init_fn` plus a seeded generator |
| Search-strategy sampling        | A private `random.Random`                |

Miss one and the run is irreproducible in a way that is difficult to notice: the numbers
look plausible, they are just different every time.

## Derived seeds, not a shared generator

The obvious approach — one global generator that everything draws from — is wrong, and
subtly so.

If the sampler, the mutation operator, and weight initialisation all drew from one stream,
then the values any one of them received would depend on how often *every other component*
had drawn. Adding a single unrelated `random.random()` call anywhere would shift the entire
downstream stream. That is an invisible coupling, and it breaks reproducibility as soon as
unrelated code changes.

Instead, each component receives a seed **derived** from the run's master seed and a stable
text label:

```python
def derive_seed(master_seed: int, label: str) -> int:
    digest = stable_hash(f"{master_seed}:{label}", digest_bytes=8)
    return int(digest, 16) % 2**32
```

Three properties make this work:

- **Deterministic** — the same pair always yields the same seed, in any process.
- **Independent** — labels differing by one character produce unrelated seeds, so component
  streams do not correlate.
- **Order-free** — a component's seed does not depend on when it was created.

The labels form a small hierarchy:

```text
master_seed (from configuration)
├── "strategy"            → the search strategy's selection generator
├── "sampler"             → architecture sampling
├── "mutation"            → mutation operators
├── "data"                → dataset shuffling and splitting
├── "training"            → weight initialisation
├── "loaders"             → DataLoader shuffling
├── "worker:{n}"          → per-worker isolation
└── "eval:{hash}:{rung}"  → per-candidate weight initialisation
```

## Seeding by architecture hash

That last label is the important one.

If every candidate drew from one shared stream, the weights a candidate received would
depend on how many candidates were evaluated *before* it. Under multiprocessing that order
is nondeterministic, so the same search would produce different results on every run — and
worse, a candidate's measured accuracy would depend on its position in the queue.

Deriving from `(master_seed, architecture_hash, rung)` makes each candidate's initial
weights a pure function of **what it is**, not **when it ran**:

```python
def candidate_seed(self, spec, budget) -> int:
    return derive_seed(self._seed, f"eval:{architecture_hash(spec)}:{budget.rung}")
```

Consequences:

- Two searches that happen to propose the same architecture train it identically.
- Concurrency does not change any individual candidate's result.
- Re-running one candidate reproduces its measurement exactly.

Asserted in
[`tests/unit/test_evaluation.py`](../../tests/unit/test_evaluation.py):
`test_candidate_seed_depends_only_on_identity` and
`test_repeated_evaluation_is_reproducible`.

## Resume continues, it does not replay

A resumed search must continue the *same* random stream, not restart it. Re-seeding from
the master seed would replay proposals already evaluated — the search would appear to run
and produce nothing but duplicates.

So the full Mersenne Twister state is checkpointed:

```python
def rng_state_to_json(rng: random.Random) -> dict[str, Any]:
    version, keys, gauss_next = rng.getstate()
    return {"version": version, "keys": list(keys), "gauss_next": gauss_next}
```

`getstate()` returns a 625-element integer tuple; JSON has no tuples, so the structure is
flattened and reassembled on load.

A determinism test proves the result: an uninterrupted four-evaluation run and the same run
split in two with a resume between reach **the same set of architectures**
(`test_resume_reaches_the_same_state_as_an_uninterrupted_run`).

Note the deliberate asymmetry: **RNG state is checkpointed; global RNG state is not
restored.** Restoring PyTorch's global RNG state across processes and versions is fragile,
so per-candidate seeds are re-derived from the persisted master seed instead — which is
version independent.

## Determinism in PyTorch

Seeding fixes the *inputs* to the generators. It does not fix the *order of floating-point
reductions* inside kernels.

Many GPU kernels — and some threaded CPU kernels — accumulate partial sums in a
nondeterministic order. Floating-point addition is not associative:

$$
(a + b) + c \ne a + (b + c)
$$

at the bit level. Two runs with identical seeds can therefore differ in the last few bits,
which occasionally flips an `argmax` and changes reported accuracy.

`configure_determinism` requests deterministic kernel selection and reports honestly what
was achieved:

```python
report = configure_determinism(enabled=True, warn_only=True)
report.to_dict()
# {'requested': True, 'deterministic_algorithms': True,
#  'cudnn_deterministic': True, 'cudnn_benchmark': False,
#  'cublas_workspace_config': ':4096:8',
#  'warnings': ['intra-op parallelism is enabled (4 threads); reduction order in some
#                CPU kernels depends on the thread count']}
```

What it does:

| Setting                                    | Effect                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| `torch.use_deterministic_algorithms(True)` | Selects deterministic kernels where they exist                          |
| `cudnn.deterministic = True`               | Forces deterministic cuDNN algorithms                                   |
| `cudnn.benchmark = False`                  | Disables autotuning, which picks different algorithms run to run        |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8`          | Fixes cuBLAS GEMM workspaces; must be set before the first CUDA context |

`warn_only=True` is the default because several common layers have no deterministic CUDA
kernel, and hard-failing would make the framework unusable on GPU. Determinism *tests* set
it to `False`, where any nondeterministic operation should be an error.

## What is never promised

**Bit-for-bit reproducibility across hardware.** Different CPUs use different SIMD widths.
Different GPUs use different reduction trees. Different BLAS builds use different blocking.
The results differ in the last bits, and no amount of seeding changes that.

**Reproducibility across library versions.** A PyTorch release can change a kernel's
default algorithm. The environment snapshot exists precisely so that this is visible.

**Identical results under multiprocessing.** See below.

**Identical latency measurements.** Latency depends on machine load, which is not under
anyone's control.

## Concurrency: what changes and what does not

| What                         | Sequential | Multiprocessing                       |
| ---------------------------- | ---------- | ------------------------------------- |
| Individual candidate results | Identical  | **Identical**                         |
| Order of proposals           | Identical  | Identical for non-adaptive strategies |
| Order of *observations*      | Identical  | **Varies**                            |
| Final ranking                | Identical  | Varies for adaptive strategies        |
| Latency measurements         | Repeatable | Noisier and systematically slower     |

Because each candidate's seed comes from its architecture hash, **concurrency does not
change any individual candidate's result**. What it changes is the order in which results
arrive — and an adaptive strategy that sees a different observation sequence can make
different subsequent proposals.

The honest summary: **sequential execution is reproducible; multiprocessing is repeatable
in distribution but not identical run to run.** The determinism tests only assert
bit-identity for sequential runs. See [concurrency](../architecture/concurrency.md).

## Environment capture

Every search persists a snapshot of what produced it:

```json
{
  "python_version": "3.12.3",
  "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
  "machine": "x86_64",
  "cpu_count": 8,
  "torch_version": "2.3.0",
  "torch_threads": 4,
  "accelerator": {"cuda_available": false, "cuda_version": null, "device_names": []},
  "package_version": "0.1.0",
  "git_commit": "a1b2c3d…",
  "environment_variables": {"OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "42"},
  "determinism": {"requested": true, "warnings": [...]}
}
```

Only allow-listed environment variables are captured — the ones that materially affect
numerical results or device selection, none of which carries credentials. See
[security](../architecture/security.md).

The snapshot appears in every report, because accuracy numbers depend on library versions
and latency numbers depend on the machine. A result without its environment is not
interpretable.

## Configuration hashing

The configuration is hashed with the same stable BLAKE2b used for architectures. The hash
is stored with the search and checked on resume:

```text
the configuration changed since this checkpoint was written
(cbb0e09da5c0b560 -> c4cc5bd2f95bdd89); results from before and after the resume may not
be comparable
```

A warning, not an error — adjusting the log level or the device between segments is
legitimate. Changes to the strategy, the space, the seeding, or the objectives are reported
*first*, because those invalidate the comparison between the two halves of the run.

## Achieving statistical repeatability

Reproducibility is about repeating one run. Statistical repeatability is about whether the
*conclusion* survives a different seed — and that is what actually matters.

To measure it:

```bash
for seed in 1 2 3 4 5; do
  nas-engine search --config configs/evolution.yaml \
    --set reproducibility.seed=$seed \
    --set project.name=evolution-seed-$seed
done
```

Then compare the distributions, not the single best values. If evolution's median across
five seeds beats random search's median across five seeds, that is a result. If evolution's
single best beats random search's single best, that is an anecdote.

A useful rule: **the spread across seeds is a lower bound on the difference you can
detect.** If five seeds of the same method span three percentage points, a two-point
difference between methods means nothing.

## The determinism test suite

[`tests/regression/test_determinism.py`](../../tests/regression/test_determinism.py)
asserts, for sequential CPU execution:

| Property                                              | Test                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| Proposal order is reproducible                        | `test_proposal_order_is_reproducible`                        |
| Mutation decisions are reproducible                   | `test_mutation_decisions_are_reproducible`                   |
| Strategy state is reproducible                        | `test_strategy_state_is_reproducible`                        |
| Evolution's population is reproducible                | `test_evolution_population_is_reproducible`                  |
| Weights are reproducible for a given seed             | `test_weights_are_reproducible_for_a_given_seed`             |
| Two identical searches agree exactly                  | `test_two_identical_sequential_searches_agree_exactly`       |
| A different seed explores differently                 | `test_a_different_seed_explores_differently`                 |
| Resume reaches the same state as an uninterrupted run | `test_resume_reaches_the_same_state_as_an_uninterrupted_run` |
| Metrics are reproducible on one machine               | `test_metrics_are_reproducible_within_one_machine`           |

And documents what is deliberately *not* asserted: latency values, cross-machine bit
equality, and multiprocessing ordering.

## Where this lives

| Concern                     | File                                                                        |
| --------------------------- | --------------------------------------------------------------------------- |
| Seed derivation and bundles | [`utilities/seeding.py`](../../src/nas_engine/utilities/seeding.py)         |
| PyTorch determinism         | [`utilities/determinism.py`](../../src/nas_engine/utilities/determinism.py) |
| Environment capture         | [`utilities/environment.py`](../../src/nas_engine/utilities/environment.py) |
| Stable hashing              | [`utilities/hashing.py`](../../src/nas_engine/utilities/hashing.py)         |
| Per-candidate seeding       | [`evaluation/evaluator.py`](../../src/nas_engine/evaluation/evaluator.py)   |
| Worker seeding              | [`orchestration/worker.py`](../../src/nas_engine/orchestration/worker.py)   |

## See also

- [Reproducibility tests](../testing/reproducibility-tests.md) — the full list of
  guarantees.
- [Concurrency](../architecture/concurrency.md) — what parallelism costs.
- [Common pitfalls](common-pitfalls.md#10-reproducibility-requires-more-than-setting-one-random-seed).
