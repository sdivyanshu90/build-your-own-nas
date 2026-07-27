# ADR 0003 — Strategies are pull-based `propose`/`observe` objects with explicit serialisable state

**Status:** Accepted
**Date:** 2026-07-27
**Supersedes:** none

## Context

Three search algorithms ship: random search, regularized evolution, and successive halving.
Three more are documented as integration points: Bayesian optimisation, RL-NAS, and DARTS.
They differ enormously.

- Random search is **stateless** beyond its generator, and every proposal is independent.
- Regularized evolution keeps a **population**, and each proposal depends on results
  already observed.
- Successive halving is **synchronous in rungs**: it cannot propose for rung *r* until
  every candidate in rung *r-1* has reported, and it proposes at *different budgets*.
- Bayesian optimisation fits a **surrogate model** over all observations and proposes by
  optimising an acquisition function.

A single interface has to hold all of these without becoming either a lowest common
denominator or a union of every algorithm's needs. It also has to satisfy two constraints
that most published NAS code ignores:

**Resumability.** A strategy's internal state must survive process death. Not
approximately — an evolutionary population restored without its generator state replays the
same mutations, and a resumed search that re-explores its first half is not a resumed
search.

**Not owning the loop.** Whatever the strategy, the engine must remain in charge of
evaluation, persistence, retries, deduplication, and the budget. Those concerns are
identical across algorithms and must not be reimplemented three times.

## Decision

An abstract base class with six required methods and two optional hooks. The engine pulls;
the strategy never calls back into it.

```python
class SearchStrategy(ABC):
    @abstractmethod
    def propose(self, count: int) -> list[Proposal]: ...
    @abstractmethod
    def observe(self, observation: Observation) -> None: ...
    @abstractmethod
    def is_finished(self) -> bool: ...
    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...
    @abstractmethod
    def load_state_dict(self, payload: dict[str, Any]) -> None: ...
    @abstractmethod
    def statistics(self) -> StrategyStatistics: ...

    def on_duplicate(self, architecture_hash: str) -> None: ...   # optional
    def on_rejected(self, spec: ArchitectureSpec, reason: str) -> None: ...  # optional
```

Five decisions are embedded in that shape.

### `propose(count)` returns *up to* `count`, and may return zero

The engine asks for as many as it can run; the strategy returns what it can honestly
supply. Returning fewer is not an error — it is how successive halving expresses "the
current rung is still in flight, ask me again after the next result."

The alternative, a strategy that blocks until it can supply `count`, would deadlock exactly
there.

### A `Proposal` carries a budget

`Proposal.spec` is *what* to evaluate; `Proposal.budget` is *how much to spend on it*. This
is what makes multi-fidelity search expressible without a second interface: successive
halving proposes the same architecture again at a larger budget, and rung is part of
candidate identity so the two are distinct candidates rather than a duplicate.

Strategies that do not care about fidelity return the configured default and never think
about it again.

The proposal also carries `parent_id`, `mutation`, `origin`, and free-form `metadata`, so
lineage is recorded by the engine rather than reconstructed later.

### `observe` receives every outcome, including failures

`Observation.result` is the full `EvaluationResult`, successful or not, and
`objective_value` is `None` when nothing scoreable was produced.

Hiding failures would be convenient and wrong: evolution must not admit a failed candidate
to its population, and successive halving must count a failure toward rung completion or it
waits forever. Both need to *see* the failure to do that.

### `objective_value` is time-stable, and computed by the engine

Scalarising several objectives into one number is a shared concern, not a per-strategy one.
The engine computes it before calling `observe`.

It is computed with the **online** (time-stable) scalariser rather than the
population-relative one. A population-relative normalisation changes every previously
observed candidate's score whenever a new extreme arrives — which would silently rewrite
the fitness of every member of an evolutionary population. Ranking in reports may use
population-relative normalisation, because there the population is fixed.

### State is explicit JSON, not pickle

`state_dict()` returns plain JSON-serialisable data and must include **everything that
affects future proposals** — generator state included. `load_state_dict` validates it as
untrusted input, checks a version field, and rejects a payload from a newer format.

Pickle would be one line and is rejected for the same reason as in
[ADR 0001](0001-search-space-representation.md): it executes code, and it breaks when a
class is renamed. A checkpoint must be readable by a build that is not byte-identical to the
one that wrote it.

Strategies also validate that **configuration has not changed across a resume**. Restoring a
16-member population into a strategy configured for 32 would silently change the algorithm;
`test_changing_population_size_across_a_resume_is_rejected` pins that.

## Alternatives considered

### A callback / inversion-of-control loop: `strategy.run(evaluate_fn)`

The strategy owns the loop and calls a supplied evaluation function.

*Rejected.* Every strategy then reimplements the budget check, the stop conditions,
checkpointing, retries, and deduplication — or the framework passes them in as more
callbacks, which is the same coupling with more indirection. Interrupting and resuming
becomes the strategy's problem, three times over. And parallel evaluation is awkward:
`evaluate_fn` must become batched or async, which pushes concurrency into every strategy.

Pull-based inverts exactly one thing and keeps every shared concern in one place.

### A generator: `yield` proposals, `send()` results

Elegant in Python, and a natural fit for the propose/observe rhythm.

*Rejected* on resumability, which is decisive. **A generator's suspended frame cannot be
serialised.** Resuming means re-running the generator from the start and discarding results
until it reaches the right point — which is not resumption, and which re-runs any side
effects. Explicit state is more code and can actually be checkpointed.

### One method: `next_batch(observations) -> list[Proposal]`

Fold `observe` into `propose`, passing new observations each call.

*Rejected*, though it is close. Separating them means an observation is recorded the moment
it arrives, not batched until the next proposal round — which matters when a search is
interrupted between the two. It also lets the engine call `observe` for results that arrive
while the strategy has nothing to propose, which is the normal state of successive halving
mid-rung.

### A plain callable, `strategy(space, history) -> Proposal`

Stateless by construction; all state lives in the history the engine passes in.

*Rejected.* It sounds cleaner and moves the cost rather than removing it. Evolution would
have to reconstruct its aging population from the full history on every call — including
which members were evicted, which the history does not record. A surrogate model would be
refit from scratch each time. Explicit state is the honest representation of algorithms
that genuinely have state.

### Requiring `on_duplicate` and `on_rejected`

*Rejected.* Random search does not care; forcing an empty override on every implementation
is noise. They default to no-ops, and evolution overrides `on_duplicate` to keep its own
novelty bookkeeping aligned with the engine's.

### Auto-discovering strategies by scanning entry points or importing a package

*Rejected*, deliberately, on security grounds. Registration is explicit —
`register_strategy(name, factory)`. Auto-discovery means an installed package can inject
executable code into a search by existing. The registry is a dictionary; adding a strategy
is one call, and the set of things that can run is auditable.

## Consequences

### Good

- The three shipped strategies share zero orchestration code.
- Successive halving's synchronisation barrier — the hardest requirement — is expressed by
  returning an empty list, with no special case in the engine.
- Adding a strategy is one class and one `register_strategy` call; the
  [guide](../guides/adding-a-search-strategy.md) walks through it.
- Every strategy is resumable by construction, because `state_dict` is not optional.
- The engine is testable against a stub strategy, and a strategy is testable without an
  engine, a dataset, or PyTorch.

### Bad

- **`state_dict` is a discipline, not a guarantee.** Nothing forces a new strategy to
  include *all* of its state. Omitting a counter produces a resume bug that only appears
  after an interruption — the hardest kind to notice. The mitigation is a test every
  strategy has: `test_state_round_trip_continues_the_stream`, which asserts that saving,
  restoring, and continuing produces the same sequence as never stopping.
- **Truly asynchronous strategies fit awkwardly.** One that wants to cancel an in-flight
  evaluation based on a partial result has no way to say so; the interface only proposes and
  observes. Cancellation is an engine concern here.
- **Gradient-based NAS (DARTS) does not fit at all.** DARTS trains a supernet with
  architecture parameters by gradient descent; there is no discrete proposal to make. It is
  documented as an integration point requiring a different evaluation path, not a strategy
  implementation. Pretending otherwise would be dishonest about what this interface is.
- Pull-based means the engine decides batch size, so a strategy cannot express "I want
  exactly 8 in flight." It can only supply fewer than asked.

## Verification

| Property | Test |
| --- | --- |
| Every registered strategy builds from the registry | `test_every_strategy_builds_from_the_registry` |
| Optional hooks default to no-ops | `test_default_hooks_are_no_ops` |
| Custom strategies can be registered | `test_custom_strategies_can_be_registered` |
| State round-trip continues rather than replays | `test_state_round_trip_continues_the_stream` (random search and evolution) |
| Evolution's population survives a round trip | `test_state_round_trip_preserves_the_population` |
| Successive halving's rung progress survives | `test_state_round_trip_preserves_rung_progress` |
| A future state version is refused | `test_rejects_a_future_state_version` (all three) |
| Malformed state is refused | `test_rejects_malformed_state` |
| Changing configuration across a resume is refused | `test_changing_population_size_across_a_resume_is_rejected`, `test_changing_the_ladder_across_a_resume_is_rejected` |
| Failures never enter the population | `test_failed_candidates_never_enter_the_population` |
| Aging removes the oldest, not the worst | `test_aging_removes_the_oldest_not_the_worst` |
| Promotion waits for the whole rung | `test_promotion_waits_for_the_whole_rung` |
| A rung of pure failures ends the bracket | `test_all_failures_end_the_bracket` |
| Proposals are reproducible from a seed | `test_is_reproducible` |
| An exhausted space stops the search | `test_exhausted_space_stops_the_search` |

`test_aging_removes_the_oldest_not_the_worst` is the one that pins the actual algorithm.
Regularized evolution's defining property is that eviction is by **age**, not fitness — that
is what stops the population collapsing onto one lineage, and it is the single easiest thing
to get wrong when reimplementing it.

## See also

- [Random search](../concepts/random-search.md) — the simplest implementation of this interface.
- [Regularized evolution](../concepts/regularized-evolution.md)
- [Successive halving](../concepts/successive-halving.md)
- [Adding a search strategy](../guides/adding-a-search-strategy.md)
- [Component design](../architecture/component-design.md)
