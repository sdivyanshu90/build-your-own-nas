# ADR 0004 — Batched process-pool execution, spawned, with plain-data payloads and a single writer

**Status:** Accepted
**Date:** 2026-07-27
**Supersedes:** none

## Context

A search is embarrassingly parallel in principle: candidates are independent, and each takes
seconds to hours. In practice four things constrain how that parallelism can be taken.

**Python's GIL.** Training is mostly in C — PyTorch releases the GIL for the heavy
operations — but the Python-level loop, the data pipeline, and the bookkeeping are not free.
Threads help less than the arithmetic suggests, and they contend on exactly the parts that
are hardest to profile.

**PyTorch and `fork` do not mix.** A forked child inherits the parent's memory, including
CUDA context and thread-pool state. CUDA is explicitly unsafe after fork. Even on CPU, a
forked child that inherits a parent's OpenMP thread pool can deadlock. This is not a rare
edge case; it is the default failure mode of naive `multiprocessing` with PyTorch.

**SQLite has one writer.** [ADR 0002](0002-persistence-layer.md) accepted that. Workers
therefore cannot write results themselves.

**Adaptive strategies are sequential by nature.** Evolution proposes based on what it has
observed. Running 8 candidates in parallel means the strategy proposes 8 before seeing any
result, which is *less informed* than proposing one at a time. Parallelism trades sample
efficiency for wall-clock time, and the trade is real.

## Decision

**Sequential by default. Optional multiprocessing via `ProcessPoolExecutor` with the
`spawn` start method, dispatched in batches, with plain-data payloads and all writes in the
parent.**

### The `EvaluationExecutor` interface

```python
class EvaluationExecutor(ABC):
    @property
    def max_in_flight(self) -> int: ...
    @property
    def mode(self) -> str: ...
    @abstractmethod
    def run_batch(self, tasks: list[EvaluationTask]) -> list[EvaluationResult]: ...
    def shutdown(self) -> None: ...
```

Two implementations: `SequentialExecutor` runs in-process; `ProcessPoolExecutorBackend`
dispatches to a pool. **The engine's loop is identical either way** — it asks the strategy
for up to `max_in_flight` proposals, hands the batch to the executor, and processes the
results. `max_in_flight` is 1 for the sequential backend, so the same code path handles
both.

That is what makes concurrency testable: every orchestration test runs sequentially, in
process, with real assertions and real tracebacks.

### `spawn`, not `fork`

A spawned child starts a fresh interpreter and imports the modules it needs. It inherits
nothing.

The cost is real — roughly a second of interpreter startup per worker, and every payload
must be picklable. The benefit is that the whole class of fork-related corruption does not
exist. When a search takes minutes to hours, a one-second startup is not the thing to
optimise.

The pool is created **lazily**, on the first non-empty batch, so a search that finishes
without dispatching anything never pays for it.

### Payloads are plain data

A task crosses the boundary as a dictionary: canonical architecture JSON, the budget as a
dict, the configuration as a mapping, and a seed. No `ArchitectureSpec`, no
`SearchConfig`, no ORM object, no evaluator.

The worker rebuilds what it needs from that payload and **validates it as untrusted input**.
Each worker process caches its evaluator keyed by the configuration, so building the dataset
and the evaluator is paid once per process, not once per task.

### Nothing escapes the worker

`_worker_entrypoint` catches every exception — including a malformed payload, and including
a failure while *constructing the failure record* — and returns a serialised failed result.
A worker never raises across the boundary.

If the process itself dies (OOM killer, segfault), the future raises in the parent, and the
executor converts that into a **retriable `WorkerError`** for that task only. The rest of the
batch is unaffected.

### Determinism is per candidate, not per stream

Each candidate's seed is derived from its **architecture hash**, not from a shared counter.
So a candidate produces identical weights and identical metrics regardless of which worker
ran it or in what order.

What parallelism does change is the *order results arrive*, and an adaptive strategy that
sees a different order can propose differently. This is stated plainly rather than papered
over: **bit-identical whole-search reproducibility is guaranteed for sequential runs only.**

## Alternatives considered

### Threads

*Rejected.* The GIL limits the speedup, and PyTorch's intra-op thread pool already uses the
cores — two Python threads each running a training loop contend for the same BLAS threads
and typically run slower than one. Threads would also share the SQLite connection and the
strategy object, requiring locks around both.

The one thing threads would buy — shared memory for the dataset — is achieved instead by
each worker loading it once and caching it.

### `fork` (the Linux default)

*Rejected.* Faster startup, shared copy-on-write memory. Also: unsafe with CUDA,
deadlock-prone with OpenMP, and it inherits the parent's SQLite connections and file
descriptors, which is a correctness hazard rather than a performance one. `spawn` is
configurable (`concurrency.start_method`) for anyone who has measured that they need it and
understands the risk, but it is not the default.

### `torch.multiprocessing` with shared-memory tensors

*Rejected.* It exists to share *tensors* between processes, which matters when passing large
batches. Here, each worker loads its own dataset once, and the per-task payload is a few
kilobytes of JSON. There is nothing to share.

### A distributed queue (Celery, Ray, Dask)

*Rejected.* They solve multi-host scheduling, which this project explicitly does not
attempt. Each brings a broker or a scheduler, a serialisation format, a deployment story,
and a debugging surface. For a single-host search over hundreds of candidates, that is a
large amount of machinery for no gain.

The `EvaluationExecutor` interface is the seam: a distributed backend implements
`run_batch` and nothing else changes. The interface was designed with that in mind, which
is different from claiming it is supported.

### Fully asynchronous dispatch (a continuously-topped-up pool)

Keep the pool saturated: as each result returns, immediately dispatch a replacement.

*Rejected*, and this is the most substantive trade-off in this ADR. Async dispatch has
strictly better worker utilisation — with batching, the whole batch waits for its slowest
member, so a batch of 8 where one candidate takes 5× the others leaves 7 workers idle.

Against that:

- **Successive halving needs the barrier.** It cannot propose rung *r+1* until every rung-*r*
  candidate has reported. Async dispatch would need an explicit synchronisation mechanism to
  express what batching gives for free.
- **Checkpointing gets harder.** With batches, there is a clean point between batches where
  the state is consistent. With continuous dispatch, a checkpoint must capture a set of
  in-flight tasks and reconcile them on resume.
- **Reasoning gets harder.** "Propose a batch, run it, observe it" is a loop anyone can hold
  in their head.

**The cost is accepted and it is quantifiable:** with a batch of *n* and evaluation times
*t₁…tₙ*, utilisation is `mean(t) / max(t)`. For roughly uniform training times — the normal
case when candidates come from one search space at one budget — that is close to 1. It gets
bad when the space contains architectures with wildly different costs, which is worth
knowing before choosing a batch size.

### Workers writing to the database directly

*Rejected.* SQLite has one writer. Even with WAL and a busy timeout, concurrent writers
serialise and can time out. More importantly, the parent must see every result to update
strategy state and check the budget — so the result has to come back regardless. Writing in
the parent means one transaction boundary, one place where state transitions happen, and one
place to reason about crash consistency.

### Doing nothing — sequential only

*Rejected*, but it is the default. Sequential is the right choice for most runs of this
project: reproducible, debuggable, and fast enough when evaluations take seconds. Multi-hour
evaluations on many cores are where the pool earns its complexity.

## Consequences

### Good

- The engine loop is identical in both modes; concurrency is not a second code path.
- Every orchestration and recovery test runs sequentially, in process, with real
  tracebacks.
- A dead worker costs one candidate, retriably, not the search.
- No fork-related corruption is possible, by construction.
- Per-candidate results are identical regardless of worker or order, because seeds derive
  from content.
- The executor interface is a clean seam for a distributed backend.

### Bad

- **Batch straggling.** Quantified above. The mitigation is `budget.max_seconds_per_evaluation`,
  which bounds how long one candidate can hold up its batch.
- **Whole-search determinism is sequential-only.** Individual candidates match; proposal
  order need not.
- **Thread oversubscription is easy to cause and hard to see.** *n* workers each spawning
  `nproc` BLAS threads is *n×* the machine's cores, and everything slows down. The rule is
  `workers × OMP_NUM_THREADS ≤ physical cores`; the Docker image sets `OMP_NUM_THREADS=1`
  for this reason, and the [runbook](../operations/production-runbook.md) leads with it.
- **Latency measurements become meaningless under load.** Concurrent workers contend, so
  measured latency reflects contention rather than the architecture. Set
  `evaluation.measure_latency: false` when running with multiple workers.
- **Spawn costs about a second per worker**, and every payload must be picklable — which
  constrains what a custom dataset provider or evaluator can hold.
- **Adaptive strategies lose information.** Proposing 8 before seeing any result is less
  informed than proposing 1 at a time. Parallelism buys wall-clock at the cost of sample
  efficiency, and for a small budget the sequential run may well find a better architecture.

## Verification

| Property                                            | Test                                                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| A valid payload produces a successful result        | `test_a_valid_payload_produces_a_successful_result`                                                   |
| The returned payload is plain data                  | `test_the_returned_payload_is_plain_data`                                                             |
| **Nothing escapes the worker**                      | `test_nothing_escapes_the_worker`                                                                     |
| A malformed configuration becomes a failed result   | `test_a_malformed_configuration_becomes_a_failed_result`                                              |
| A malformed architecture becomes a failed result    | `test_a_malformed_architecture_becomes_a_failed_result`                                               |
| A missing budget becomes a failed result            | `test_a_missing_budget_becomes_a_failed_result`                                                       |
| Worker results are reproducible across calls        | `test_results_are_reproducible_across_calls`                                                          |
| The evaluator is cached per configuration           | `test_the_evaluator_is_cached_per_configuration`                                                      |
| A different configuration builds a second evaluator | `test_a_different_configuration_builds_a_second_evaluator`                                            |
| A dead worker becomes a retriable failure           | `test_a_dead_worker_becomes_a_retriable_failure`                                                      |
| An empty batch never creates a pool                 | `test_an_empty_batch_is_a_no_op`                                                                      |
| Worker count is validated                           | `test_worker_count_is_validated`                                                                      |
| Sequential execution preserves input order          | `test_sequential_execution_returns_results_in_input_order`                                            |
| Each mode builds the right backend                  | `test_sequential_mode_builds_the_inline_backend`, `test_multiprocessing_mode_builds_the_pool_backend` |
| Multiprocessing produces an equivalent result       | `test_multiprocessing_produces_the_same_kind_of_result`                                               |

`test_nothing_escapes_the_worker` is the load-bearing one. A worker that raises across the
process boundary turns one bad candidate into a dead search — and it is exactly the path
that is hardest to exercise by hand, because it only runs when something has already gone
wrong.

## See also

- [Concurrency](../architecture/concurrency.md) — the mechanics in detail.
- [Reproducibility](../concepts/reproducibility.md) — what parallelism costs.
- [Production runbook](../operations/production-runbook.md) — thread settings and
  troubleshooting.
- [ADR 0002](0002-persistence-layer.md) — the single-writer constraint this design respects.
