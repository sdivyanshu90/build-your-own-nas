# Glossary

Every term this project uses in a specific sense. Where a word has a general meaning and a
narrower one here, the narrower one is what the code means.

---

## A

**Aging** — In regularized evolution, removing the **oldest** population member rather than
the worst. This is the algorithm's defining property: fitness-based removal lets one lucky
lineage dominate and the population collapse, whereas aging forces continual turnover, so a
member survives only by leaving descendants. Implemented as `deque(maxlen=population_size)`.
See [regularized evolution](concepts/regularized-evolution.md).

**Architecture hash** — A 32-character lowercase hex string identifying an architecture by
its content: BLAKE2b over its canonical JSON, truncated to 128 bits. Two specs that describe
the same network share a hash; two that differ in any *active* field do not. The basis of
deduplication, of per-candidate seeding, and of candidate identity in the database.

**`ArchitectureSpec`** — The genotype. A frozen Pydantic model holding a stem, a tuple of
stages, and a head. Contains no tensors and no device; it is a *description* of a network,
not a network.

**Artifact** — A file a candidate produced: trained weights, a training checkpoint.
Stored on the filesystem under the artifact root, with the relative path recorded in the
database. Paths are validated against the root before writing.

**Artifact root** — The directory artifacts are written under, `<output_dir>/candidates` by
default. The security boundary for artifact writes: nothing may resolve outside it.

---

## B

**Block** — One repeated unit inside a stage: an operation, optional normalisation, optional
activation, a stride, and an optional residual connection. The smallest thing a mutation
operator changes.

**Budget** — See **training budget**. Distinct from the **search budget**, which is the total
number of evaluations or seconds the whole search may spend.

**Bi-level optimisation** — The formal shape of NAS: an outer loop searching over
architectures, an inner loop training weights for each one. The outer objective depends on
the inner optimum, which is why NAS is expensive — every outer step contains a full inner
optimisation. See [NAS foundations](concepts/nas-foundations.md).

---

## C

**Canonical form** — The single representation chosen for architectures that are
semantically identical. Keys sorted, separators compact, ASCII only, and — critically —
**inactive fields reset to their defaults**. A pooling block's `expansion_ratio` means
nothing, so canonicalisation zeroes it and two blocks that build the same layer hash the
same. Idempotent: canonicalising twice changes nothing.

**Candidate** — One architecture *within one search at one rung*, together with its
lifecycle state, its trials, and its metrics. Not the same as an architecture: the same
architecture at rung 0 and rung 1 is two candidates.

**Candidate state** — Where a candidate is in its lifecycle: `PROPOSED`, `VALIDATED`,
`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `PRUNED`, `CANCELLED`. Transitions are checked
against an explicit table; an illegal one raises rather than silently corrupting state.

**Checkpoint (search)** — A snapshot of the engine's counters and the strategy's
`state_dict`, written to the database periodically. What makes `resume` continue rather than
restart.

**Checkpoint (training)** — A per-candidate snapshot of model and optimiser state, so a
long training run can resume mid-way. Off by default; it costs roughly three times the
weight file per candidate.

**Configuration hash** — A hash of the resolved configuration, stored with each search. Two
searches with the same configuration hash are comparable; two without are not.

**Constraint** — A hard feasibility limit. Three kinds, applied at three different times and
with three different outcomes — see the table in the
[runbook](operations/production-runbook.md#three-different-limits-three-different-outcomes).
Most importantly, a *search-space* constraint prunes analytically before any work is done.

**Crowding distance** — In NSGA-II, a measure of how isolated a solution is along the front.
Used to break ties within the same Pareto rank, preferring solutions in sparse regions so
the front stays spread out rather than clustering.

---

## D

**DARTS** — Differentiable Architecture Search. Relaxes the discrete choice into a
continuous mixture and optimises architecture weights by gradient descent on a supernet.
Documented here as an integration point, not implemented: it needs a fundamentally different
evaluation path, not another `SearchStrategy`.

**Deduplication** — Refusing to evaluate an architecture already seen at the same rung in the
same search. Enforced by a `UNIQUE(search_id, architecture_hash, rung)` constraint, so it
holds even under a race between workers.

**Derived seed** — A seed computed deterministically from a master seed and a label, via
`derive_seed(master, label)`. Every component gets its own stream instead of drawing from a
shared global RNG, so adding a component does not shift every other component's numbers.

**Dominance** — In multi-objective optimisation, *A* dominates *B* when *A* is at least as
good as *B* on every objective and strictly better on at least one. See **Pareto front**.

---

## E

**Evaluation** — Building a model from a spec, training it under a budget, and measuring it.
The expensive step, and the unit the search budget is spent in.

**`EvaluationResult`** — Everything one evaluation produced: metrics, artifacts, timing,
device, and — on failure — a structured `EvaluationFailure`. Plain data, so it crosses a
process boundary.

**Event** — A structured log record with a name from a closed vocabulary. Names are a public
interface; see [observability](operations/observability.md).

---

## F

**Failure kind** — A coarse classification of why an evaluation failed: `validation`,
`constraint`, `build`, `training`, `divergence`, `timeout`, `resource`, `persistence`,
`worker`, `unknown`. Determines the retry decision and is what to group by when diagnosing a
run.

**Feasible** — Satisfying every objective constraint. An infeasible candidate is ranked
below every feasible one regardless of its score, and is excluded from the Pareto front.

**Fidelity** — How much is spent evaluating a candidate: epochs, fraction of the training
data, input resolution. **Low fidelity** is cheap and noisy; **high fidelity** is expensive
and accurate. Multi-fidelity search spends low fidelity on many candidates and high fidelity
on few.

---

## G

**Genotype** — The description of an architecture (`ArchitectureSpec`). Cheap to copy, hash,
store, and send. Contrast **phenotype**.

**Golden fixture** — A stored value a test asserts against, versioned deliberately. A
failure is not necessarily a bug — it is a demand for a decision, because the value changing
means stored hashes or historical results are now incomparable.

---

## H

**Head** — The final part of a network: global pooling, optional hidden layer, and the
classifier.

---

## L

**Latency** — Measured inference time per batch. Reported as a median over repeats after
warm-up. **Deliberately not asserted to be reproducible** — it depends on machine load,
which nobody controls. Meaningless when multiple workers are running.

**Lineage** — The parent-child graph produced by mutation: which candidate came from which,
by which operator. Recorded as candidates are created, not reconstructed afterwards.

---

## M

**MACs** — Multiply-accumulate operations per image. A hardware-independent proxy for
compute cost. Roughly half the FLOP count, since one MAC is a multiply and an add.
**Estimated** here, unlike parameter counts, which are exact.

**Multi-fidelity** — Evaluating at several budgets, spending more on candidates that look
promising at less. Successive halving is the implementation here.

**Multi-objective** — Optimising several objectives that trade against each other — accuracy
against parameters, against latency, against size. There is no single best solution, only a
**Pareto front**.

**Mutation** — A small, structured change to an architecture producing a valid child. Twelve
operators ship, covering operation type, kernel size, channels, depth, normalisation,
activation, stride, residuals, and the head.

---

## N

**NAS** — Neural Architecture Search: automating the design of network architectures rather
than hand-designing them.

**NSGA-II** — A multi-objective evolutionary algorithm. Two of its ideas are used here:
non-dominated sorting (partitioning a population into fronts by dominance) and crowding
distance (tie-breaking within a front).

---

## O

**Objective** — A metric to optimise, with a direction (maximise or minimise), a weight, and
a normalisation. `validation_accuracy` maximised is the default; `trainable_parameters`
minimised is the usual second.

**Online scalarisation** — Combining objectives into one number using **fixed** reference
points, so a candidate's score never changes as the population grows. Used for the value fed
back to a strategy. Contrast **population-relative scalarisation**, which renormalises
against the current population — fine for a final report, wrong for feeding an evolutionary
population whose members' fitness would silently be rewritten.

**Operator** (mutation) — One kind of structured change, e.g. "change a block's kernel
size."

**Operation** (architecture) — What a block computes: `convolution`,
`separable_convolution`, `pooling`, `identity`.

---

## P

**Pareto front** — The set of candidates not dominated by any other. Every member is better
than every non-member on at least one objective. The honest answer to a multi-objective
search: not one architecture, but the set representing the available trade-offs.

**Pareto rank** — Which front a candidate lies on. Rank 0 is the true front; rank 1 is the
front remaining after rank 0 is removed, and so on.

**Phenotype** — The actual `torch.nn.Module`, built from a genotype on demand. Holds tensors
and lives on a device. Never stored, never serialised, never crosses a process boundary.

**Preset** — A named, ready-made search space (`default_cnn`, `micro_cnn`, …), refinable
with `overrides` rather than redefined from scratch.

**Proposal** — A strategy's suggestion: an architecture *plus the budget to spend on it*,
plus lineage metadata. Carrying the budget is what makes multi-fidelity search expressible
without a second interface.

**Pruned** — Rejected before evaluation for violating a resource constraint. Distinct from
**failed**: pruning is a normal, cheap, expected outcome; failure means something went
wrong. Conflating them makes a healthy search look broken.

---

## R

**Recovery sweep** — On resume, moving every candidate left in `RUNNING` back to `QUEUED`
(or failing it, if retries are exhausted). What makes a killed process recoverable, since a
crashed evaluation cannot clean up after itself.

**Regularized evolution** — Evolution with **aging**: tournament selection for parents,
mutation to produce a child, and removal of the oldest member. The "regularization" is the
aging.

**Repair** — Adjusting a sampled or mutated architecture to restore validity — for instance
fixing a channel count so a residual connection remains legal — instead of discarding it.
Cheaper than rejection sampling in a constrained space.

**Repository** — The object mediating all database access, returning detached frozen
dataclasses rather than ORM objects. One public method is one transaction.

**Rung** — One level of the successive-halving ladder, each with its own budget and its own
number of candidates. Part of candidate identity, so the same architecture at rung 0 and
rung 1 is two candidates rather than a duplicate.

---

## S

**Scalarisation** — Collapsing several objectives into one number so candidates can be
ordered. Necessarily lossy; the Pareto front is the lossless view.

**Search budget** — The total the search may spend: `max_evaluations`, `max_seconds`. Distinct
from a **training budget**, which governs one candidate.

**Search space** — The set of architectures a search may consider, defined by the choices
available at each position plus the constraints every member must satisfy. The single most
consequential design decision in a NAS project: nothing outside it can be found.

**Search strategy** — The algorithm deciding what to try next. Random search, regularized
evolution, and successive halving ship.

**Seed bundle** — The derived seeds for one search, recorded so a run can be reproduced.

**Spawn** — The multiprocessing start method used here: a fresh interpreter that inherits
nothing. Slower to start than `fork`, and the only safe option with PyTorch.

**Stage** — A group of blocks sharing a channel width. Typically the first block of a stage
strides and widens; the rest maintain.

**Stem** — The first convolution, mapping input channels to the initial width.

**Successive halving** — Evaluate *n* candidates at a small budget, keep the best *1/η*,
repeat at *η×* the budget. Spends most of the budget on the candidates that survive.

**Supernet** — A single over-parameterised network containing every architecture in the
space as a subnetwork, used by weight-sharing NAS. Not used here; the trade-off is discussed
in [training and evaluation](concepts/training-and-evaluation.md).

---

## T

**Total stride** — The product of every stride in a network. Determines how much the spatial
resolution is reduced overall: total stride 8 turns 32×32 into 4×4.

**Training budget** — What one candidate gets: epochs, fraction of the training data,
resolution, and a wall-clock cap. Carried on the proposal.

**Trial** — One *attempt* at evaluating a candidate. A candidate can have several — retries,
and re-evaluations at a higher rung.

**Trust boundary** — Where untrusted data enters: configuration files, imported architecture
JSON, checkpoint payloads, and rows read back from the database. Everything crossing one is
validated; nothing crossing one is executed. See [security](architecture/security.md).

---

## V

**Validation accuracy** — Accuracy on the validation split, the default objective. **Not**
test accuracy: the search selects on it, so it is optimistically biased. The test split is
touched once, at the end, by `nas-engine evaluate`.

---

## W

**WAL** — SQLite's write-ahead logging mode. Lets readers proceed while a writer is active,
which is why `nas-engine status` works during a search. Also why a live backup needs the
backup API rather than `cp`.

**Weight sharing** — Reusing trained weights across architectures to avoid training each
from scratch. Dramatically cheaper and known to rank architectures unreliably; see
[training and evaluation](concepts/training-and-evaluation.md).

---

## See also

- [Index](index.md)
- [NAS foundations](concepts/nas-foundations.md) — the concepts, at length.
- [Common pitfalls](concepts/common-pitfalls.md) — where these distinctions matter most.
