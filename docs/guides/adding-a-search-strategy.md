# Adding a search strategy

The project's central extension point. Implement one interface, register it, and the engine
uses it — no engine change required.

## The contract

```python
class SearchStrategy(ABC):
    name: ClassVar[str]
    requires_synchronous_observations: ClassVar[bool] = False

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

    # optional hooks with no-op defaults
    def on_duplicate(self, architecture_hash: str) -> None: ...
    def on_rejected(self, spec: ArchitectureSpec, reason: str) -> None: ...
```

The engine drives the candidate lifecycle — validation, deduplication, persistence,
evaluation, retries, checkpointing — and knows nothing about *how* candidates are chosen. A
strategy chooses candidates and knows nothing about how they are evaluated or stored.

That is dependency inversion in the literal sense: both depend on this abstraction, and
neither depends on the other.

### Method semantics, precisely

**`propose(count)`** — return *up to* `count` proposals.

| Return                                               | Means                                      | Engine's response                                  |
| ---------------------------------------------------- | ------------------------------------------ | -------------------------------------------------- |
| `count` proposals                                    | Business as usual                          | Evaluates them                                     |
| Fewer than `count`                                   | "I cannot usefully propose more right now" | Evaluates what it got                              |
| `[]` with work outstanding                           | "I am waiting for results"                 | Drains in-flight work, asks again                  |
| `[]` with nothing outstanding, `is_finished()` false | The strategy is stuck                      | Stops with `SPACE_EXHAUSTED` after two idle rounds |

**`observe(observation)`** — exactly one completed evaluation, successful **or failed**.
Strategies must handle failures: a candidate that crashed still consumed budget and must
not be proposed again.

**`is_finished()`** — whether the plan is complete. The engine enforces its own budget too;
whichever triggers first wins.

**`state_dict()` / `load_state_dict()`** — serialise and restore *everything* affecting
future proposals, **including generator state**. A strategy that re-seeds on resume replays
proposals it already made.

**`statistics()`** — free-form counters for reports and logs. Keys should be stable.

---

## A worked example: greedy hill-climbing

Keep the single best architecture seen; mutate it repeatedly; move only when a child beats
it. Deliberately naive — its purpose is to show the mechanics end to end.

```python
"""Greedy hill-climbing: mutate the incumbent, accept only improvements."""

from __future__ import annotations

from typing import Any, ClassVar

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import CheckpointError, CheckpointVersionError, MutationError
from nas_engine.search.strategy import (
    Observation, Proposal, SearchStrategy, StrategyStatistics,
    deserialize_spec, serialize_spec,
)
from nas_engine.search_space.mutation import MutationOperator
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace
from nas_engine.utilities.seeding import derive_seed

HILL_CLIMB_STATE_VERSION = 1


class HillClimbing(SearchStrategy):
    """Mutate the incumbent; accept a child only if it scores better."""

    name: ClassVar[str] = "hill_climbing"

    def __init__(self, space: SearchSpace, *, seed: int, max_evaluations: int,
                 budget: TrainingBudget) -> None:
        if max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1, received {max_evaluations}")
        self._space = space
        self._seed = seed
        self._max_evaluations = max_evaluations
        self._budget = budget
        self._sampler = ArchitectureSampler(space, seed=derive_seed(seed, "hill:sampler"))
        self._mutator = MutationOperator(space, seed=derive_seed(seed, "hill:mutation"))
        self._incumbent: ArchitectureSpec | None = None
        self._incumbent_id: str | None = None
        self._incumbent_value = float("-inf")
        self._seen: set[str] = set()
        self._proposed = self._observed = self._improvements = 0

    # -- proposing -----------------------------------------------------------------
    def propose(self, count: int) -> list[Proposal]:
        wanted = max(0, min(count, self._max_evaluations - self._proposed))
        proposals = []
        for _ in range(wanted):
            proposal = self._propose_one()
            if proposal is None:
                break
            proposals.append(proposal)
        return proposals

    def _propose_one(self) -> Proposal | None:
        if self._incumbent is None:
            spec = self._sampler.sample_unique(self._seen)
            if spec is None:
                return None
            origin, parent, mutation = "random", None, None
        else:
            try:
                result = self._mutator.mutate(self._incumbent)
            except MutationError:
                spec = self._sampler.sample_unique(self._seen)
                if spec is None:
                    return None
                origin, parent, mutation = "random_fallback", None, None
            else:
                if architecture_hash(result.child) in self._seen:
                    return None          # the neighbourhood is saturated
                spec = result.child
                origin, parent, mutation = "mutation", self._incumbent_id, result.description

        self._seen.add(architecture_hash(spec))
        self._proposed += 1
        return Proposal(spec=spec, budget=self._budget, parent_id=parent,
                        mutation=mutation, origin=origin)

    # -- observing -----------------------------------------------------------------
    def observe(self, observation: Observation) -> None:
        self._observed += 1
        self._seen.add(observation.architecture_hash)
        if not observation.succeeded or observation.objective_value is None:
            return                                   # failures carry no fitness
        if observation.objective_value > self._incumbent_value:
            self._incumbent = observation.spec
            self._incumbent_id = observation.candidate_id
            self._incumbent_value = observation.objective_value
            self._improvements += 1

    def is_finished(self) -> bool:
        return self._proposed >= self._max_evaluations

    def on_duplicate(self, architecture_hash: str) -> None:
        self._seen.add(architecture_hash)

    # -- reporting -----------------------------------------------------------------
    def statistics(self) -> StrategyStatistics:
        return StrategyStatistics(
            proposed=self._proposed, observed=self._observed,
            extra={"incumbent_value": self._incumbent_value,
                   "improvements": self._improvements,
                   "unique_architectures": len(self._seen)},
        )

    # -- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HILL_CLIMB_STATE_VERSION,
            "incumbent": serialize_spec(self._incumbent) if self._incumbent else None,
            "incumbent_id": self._incumbent_id,
            "incumbent_value": self._incumbent_value,
            "seen": sorted(self._seen),
            "proposed": self._proposed,
            "observed": self._observed,
            "improvements": self._improvements,
            "sampler": self._sampler.state_dict(),      # includes RNG position
            "mutator": self._mutator.state_dict(),      # includes RNG position
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("version") != HILL_CLIMB_STATE_VERSION:
            raise CheckpointVersionError(
                f"hill_climbing state version {payload.get('version')} is not supported"
            )
        try:
            incumbent = payload["incumbent"]
            self._incumbent = deserialize_spec(incumbent) if incumbent else None
            self._incumbent_id = payload["incumbent_id"]
            self._incumbent_value = float(payload["incumbent_value"])
            self._seen = set(payload["seen"])
            self._proposed = int(payload["proposed"])
            self._observed = int(payload["observed"])
            self._improvements = int(payload["improvements"])
            self._sampler.load_state_dict(payload["sampler"])
            self._mutator.load_state_dict(payload["mutator"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"hill_climbing state is malformed: {exc}") from exc
```

### Register it

```python
from nas_engine.search.registry import register_strategy

def _build_hill_climbing(*, space, seed, budget, max_evaluations, native_resolution, params):
    return HillClimbing(space, seed=seed, max_evaluations=max_evaluations, budget=budget)

register_strategy("hill_climbing", _build_hill_climbing)
```

The factory signature is uniform across strategies; ignore what you do not need.

### Use it

```yaml
algorithm:
  name: hill_climbing
```

Registration must happen before the engine builds the strategy — in your own module, which
you import before constructing the engine. There is deliberately no auto-discovery:
importing a module named in a configuration file *is* arbitrary code execution.

### Test it

Every guarantee the interface makes should be asserted. The shipped strategies' tests in
[`tests/unit/test_strategies.py`](../../tests/unit/test_strategies.py) are the template:

```python
def test_state_round_trip_continues_the_stream(tiny_space):
    original = HillClimbing(tiny_space, seed=1, max_evaluations=10, budget=BUDGET)
    original.propose(2)
    state = original.state_dict()
    expected = [architecture_hash(p.spec) for p in original.propose(2)]

    restored = HillClimbing(tiny_space, seed=999, max_evaluations=10, budget=BUDGET)
    restored.load_state_dict(state)
    assert [architecture_hash(p.spec) for p in restored.propose(2)] == expected
```

Also cover: failures never poison the state; duplicates are avoided; `is_finished` respects
the budget; a corrupt payload raises; statistics are populated.

---

## Three advanced strategies

These are **not implemented** here. What follows is the integration contract for each: what
would need to be built, where it plugs in, and what to be careful about. Enough to
implement them without re-deriving the design.

### 1. Bayesian optimisation

Fit a probabilistic surrogate to observed `(architecture, score)` pairs; propose the point
that maximises an acquisition function.

**The hard part is the kernel.** Bayesian optimisation needs a similarity measure over
architectures. Three options, in increasing fidelity and cost:

| Approach | How | Trade-off |
| --- | --- | --- |
| Feature vector | Encode as fixed-length numeric features (depth, widths, operation counts) and use a standard GP | Simple; loses structure |
| Graph kernel | Weisfeiler-Lehman or a path kernel over the architecture DAG | Structure-aware; expensive |
| Learned embedding | Train an encoder on observed architectures | Powerful; needs data you do not have early on |

Start with the feature vector. The genotype makes it easy:

```python
def features(spec: ArchitectureSpec) -> list[float]:
    cost = compute_cost(spec)
    counts = Counter(block.operation for _, _, block in spec.iter_blocks())
    return [
        float(spec.num_stages), float(spec.total_blocks), float(spec.total_stride),
        math.log10(cost.trainable_parameters), math.log10(cost.multiply_accumulates),
        *[float(counts[op]) for op in OperationType],
        *stage_widths(spec), *[0.0] * (MAX_STAGES - spec.num_stages),   # zero-padded
    ]
```

**Integration:**

| Method | Implementation |
| --- | --- |
| `propose` | Sample $M$ random candidates, score each by expected improvement, return the top $k$ |
| `observe` | Append to the training set; refit the surrogate (or defer to the next `propose`) |
| `state_dict` | Observations *and* the surrogate's hyperparameters — refitting from scratch changes the acquisition surface |
| `requires_synchronous_observations` | Consider `True`: the surrogate is stale between an observation and the refit. Or implement a batch acquisition (q-EI, constant liar) |

**Watch out for:** the cold start (the first ~10 observations produce a useless surrogate;
sample randomly until then), refit cost (an exact GP is $O(n^3)$ — fine below a thousand
observations), and the fact that
[the online scalar is what you observe](../concepts/multi-objective-optimization.md#online-versus-final-scoring),
not the population-relative score.

### 2. Reinforcement-learning NAS

A controller — usually an LSTM — emits architecture choices as a sequence of actions.
Validation accuracy is the reward; the controller is updated by a policy gradient such as
REINFORCE.

**The choice sequence is already there.** The sampler makes each decision in a fixed order,
which is exactly the action sequence a controller would produce:

```text
stem width → stem kernel → stem stride → num_stages
→ for each stage: width, depth, stride
  → for each block: operation, kernel, expansion, norm, activation, residual
→ head pooling, hidden, dropout
```

An RL controller replaces `rng.choice(options)` with `sample_from_policy(logits, options)`.

**Integration:**

| Method       | Implementation                                                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `propose`    | Roll out the controller; record the log-probability of each action taken                                                            |
| `observe`    | Compute the reward (typically `accuracy - baseline`); accumulate the policy-gradient loss                                           |
| Update       | After every $k$ observations, take an optimiser step on the controller                                                              |
| `state_dict` | Controller weights (as a list, not a tensor — the payload must be JSON), optimiser state, the reward baseline, and the RNG position |

**Watch out for:** sample efficiency (RL-NAS classically needed thousands of evaluations —
budget accordingly), reward variance (an exponential-moving-average baseline is essential;
without it the gradient is dominated by noise), entropy collapse (add an entropy bonus or
the controller becomes deterministic within a few dozen updates), and the fact that the
controller is itself a neural network with its own hyperparameters — which is exactly the
criticism levelled at RL-NAS in the literature.

### 3. Differentiable architecture search (DARTS)

Relax the discrete choice into a continuous mixture. Instead of picking one operation,
compute a weighted sum of all of them:

$$
\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})}\, o(x)
$$

Train the architecture parameters $\alpha$ and the weights $w$ jointly by alternating
gradient descent, then discretise by taking the arg-max.

**This does not fit the `SearchStrategy` interface**, and that is the important point.
DARTS has no propose/evaluate loop: there is one continuous optimisation producing one
architecture at the end. Forcing it into `propose`/`observe` would be a lie about what it
does.

**How to integrate it properly:**

1. Add a `SupernetBuilder` alongside `ModelBuilder` that constructs the mixed-operation
   network from a `SearchSpace`.
2. Add a `DartsTrainer` alongside `Trainer` implementing the bi-level alternating update.
3. Add a `DartsSearchEngine` — a *sibling* of `SearchEngine`, not a strategy — that runs the
   supernet optimisation and produces a `SearchResult`.
4. Reuse `ArchitectureSpec`, the persistence layer, the objectives, and the reporting
   unchanged. Those are the parts that generalise.

The discretised architecture is a normal `ArchitectureSpec`, so everything downstream —
hashing, storage, ranking, reporting, model rebuilding — works without modification.

**Watch out for:** memory (the supernet holds every operation simultaneously; memory scales
with $|\mathcal{O}|$), the discretisation gap (the arg-max architecture is not the mixture
that was optimised, and the two can perform very differently), the well-documented collapse
towards parameter-free operations (skip connections dominate as training proceeds), and the
[weight-sharing bias](../concepts/common-pitfalls.md#7-weight-sharing-nas-can-bias-rankings)
that applies to any one-shot method.

---

## A checklist

- [ ] `name` is unique and stable — it is persisted with every search
- [ ] `propose` respects `count` and the evaluation budget
- [ ] `propose` returns `[]` rather than blocking when waiting for results
- [ ] `observe` handles failed observations and `objective_value is None`
- [ ] `is_finished` eventually returns `True`
- [ ] `state_dict` includes **every** generator's position
- [ ] `load_state_dict` validates the version and raises on a malformed payload
- [ ] `statistics` returns useful, stable keys
- [ ] `on_duplicate` keeps internal bookkeeping in sync
- [ ] `requires_synchronous_observations` is set if the strategy adapts after each result
- [ ] A resume test asserts the stream continues rather than replaying
- [ ] Registered with `register_strategy`

## See also

- [ADR 0003](../adr/0003-search-strategy-interface.md) — why this interface.
- [Component design](../architecture/component-design.md) — where strategies sit.
- [Random search](../concepts/random-search.md) — the simplest complete implementation to
  read.
