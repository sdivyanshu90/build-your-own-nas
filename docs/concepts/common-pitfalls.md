# Common pitfalls

Ten claims about NAS that sound reasonable and are wrong, stated precisely and corrected.

Every generated report carries an abbreviated version of this list, because a headline
number without these caveats reads as stronger than it is.

---

## 1. NAS does not guarantee a globally optimal architecture

**The misconception:** "We ran NAS, so we found the best architecture."

**What is true:** NAS finds the best architecture *it happened to evaluate*, inside *one*
search space, under *one* training recipe, with *one* budget.

The default space here holds roughly $10^{21}$ members. A search evaluating 100 of them has
seen a fraction of $10^{-19}$. No search algorithm — not evolution, not Bayesian
optimisation, not reinforcement learning — provides an optimality guarantee over a
combinatorial space of that size under a noisy objective. The problem is
NP-hard-in-practice and the objective is not even deterministic.

**What to say instead:** "The best architecture found by *N* evaluations of *method* over
*space*, trained with *recipe*." All four qualifiers matter.

**In this project:** the report's first limitation, and `SearchResult.stop_reason` records
*why* the search ended — a budget-exhausted search almost certainly had further to go.

---

## 2. More expensive search does not automatically mean better generalisation

**The misconception:** "We spent 1000 GPU-hours, so the result must be good."

**What is true:** More search means more selection on the validation split, which means
*more* selection bias — not less. Taking the maximum over $N$ noisy estimates
systematically overestimates the winner's true value, and the overestimate grows with $N$.

Beyond some point, extra search buys overfitting to the validation split rather than a
better architecture. That point arrives sooner with a small validation split.

**How to tell:** track the *test* accuracy of the best-so-far candidate as the search
proceeds. If validation keeps improving while test plateaus or declines, the search is
overfitting the validation split. (Doing this properly requires a fourth split, or accepting
that you have spent your test set on diagnostics.)

**In this project:** the search-progress figure plots the running best. A curve that rises
steeply then flattens is the signal that additional budget is not buying much.

---

## 3. Validation accuracy is a noisy estimate

**The misconception:** "Architecture A scored 0.847 and B scored 0.839, so A is better."

**What is true:** With $n$ validation examples and true accuracy $p$, the measured accuracy
has standard error

$$
\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}.
$$

| $n$    | $p$  | SE    | 95% interval half-width |
| -----: | ---: | ----: | ----------------------: |
|    256 | 0.85 |  2.2% |                    4.4% |
|  1 000 | 0.85 |  1.1% |                    2.2% |
|  5 000 | 0.85 |  0.5% |                    1.0% |
| 10 000 | 0.85 | 0.36% |                    0.7% |

A 0.8-point difference on a 1 000-example split is *within one standard error*. It is not
evidence.

And that is only the finite-sample noise. Training noise — different initialisation,
different data order — adds more, often *more* than the sampling noise at short budgets.

**Rule of thumb:** train the same architecture with three different seeds. The spread you
observe is the resolution of your measurement. Differences smaller than that spread mean
nothing.

**In this project:** the report states the validation split size, so the standard error can
be computed. [Interpreting results](../guides/interpreting-results.md#is-this-difference-real)
does the arithmetic.

---

## 4. Reusing the test set during search causes leakage

**The misconception:** "We only have train and test, so we searched on test."

**What is true:** Selecting on a split destroys that split's ability to estimate
generalisation. Search over 100 candidates picking the best on the test set produces a test
number that is optimistically biased by roughly the same amount as a validation number
would be — you have simply moved the problem and lost your ability to detect it.

**Correct procedure:**

1. Split into train / validation / test **before** the search.
2. Fit weights on train.
3. Rank architectures on validation.
4. Touch test **once**, at the end, for the number you report.

**In this project:** enforced structurally. `build_dataloaders(..., include_test=False)` is
the default; the evaluator is only ever handed train and validation. The test split is
reachable only through `nas-engine evaluate`, which is a separate, explicit command.

The residual risk is running `nas-engine evaluate` repeatedly and picking the best result.
Nothing can prevent that; the documentation names it.

---

## 5. Comparing latency across hardware is misleading

**The misconception:** "This architecture takes 4.2 ms, so it will be fast on the phone."

**What is true:** Latency depends on the CPU or GPU, the thread count, the batch size, the
memory bandwidth, the kernel library, the compiler, and what else the machine is doing. A
model that is fast on a desktop CPU can be slow on a mobile SoC and vice versa, because the
bottleneck moves.

The classic case is depthwise convolution. It has very few MACs and poor **arithmetic
intensity** — few operations per byte moved. On a compute-bound device it looks excellent;
on a memory-bound device it is often *slower* per MAC than a dense convolution. Ranking by
MACs and ranking by measured latency can disagree substantially.

**In this project:** every `LatencyMeasurement` carries device metadata and an explicit
warning:

> Latency is hardware-, thread-, and load-dependent. These numbers are comparable only
> between candidates measured on the same machine during the same run. Do not compare them
> across machines, batch sizes, or PyTorch versions.

The warning appears in the report next to the latency figure. To make a deployment
decision, measure on the deployment hardware.

---

## 6. Search cost must be included when comparing NAS methods

**The misconception:** "Method X found a 94% model, method Y found 93%, so X is better."

**What is true:** That statement is empty without the compute each method used. If X used
ten times the budget, the comparison says nothing about the methods.

This is the most common flaw in NAS papers, and the reason random search kept turning out
to be competitive once people started controlling for it.

**How to compare fairly:**

1. Fix the search space.
2. Fix the total compute — GPU-hours, or evaluation-epochs, or any consistent unit.
3. Run each method with several seeds.
4. Compare distributions, not single best values.

**In this project:** `TrainingBudget.relative_cost` gives a dimensionless per-evaluation
cost, and the report totals evaluations and wall-clock duration. Successive halving is the
case that needs care: it performs *more* evaluations at *lower* fidelity, so comparing by
evaluation count rather than total compute flatters it.

---

## 7. Weight-sharing NAS can bias rankings

**The misconception:** "One-shot NAS is just a faster way to get the same ranking."

**What is true:** In weight sharing, a candidate's inherited weights were shaped by
training every *other* candidate too. The measured performance is of "this subnetwork
inside this supernet", not of "this architecture trained properly".

Published analyses repeatedly find weak rank correlation between supernet evaluation and
independent training. The bias is *systematic*, not random noise: supernets tend to favour
architectures that train quickly and share structure with the rest of the population.

**In this project:** every candidate is trained **independently**. That is slower and it is
honest — each measurement means what it appears to mean. Weight sharing is a documented
extension point, not a hidden default. If you add it, validate the rank correlation on your
space before trusting it.

---

## 8. Search-space design often matters as much as the search algorithm

**The misconception:** "We should focus on a better search algorithm."

**What is true:** A well-designed space with random search frequently matches a poorly
designed space with a sophisticated method. The space encodes the prior knowledge that does
most of the work.

The corollary is uncomfortable: **a strong NAS result is partly a property of the space.**
A method evaluated on a space where every member is decent will look good regardless of
what it does.

**How to check:** run random search on your space at the same budget. If your method does
not clearly beat it, the method is not what is producing your result.

**In this project:** random search is a first-class strategy, not an afterthought, and
[the documentation](random-search.md#when-to-use-it) tells you to run it first. The space
description, including its size and its biases, appears in every report.

---

## 9. A discovered architecture may overfit the dataset and the training recipe

**The misconception:** "This architecture is better, so it will be better everywhere."

**What is true:** The search optimised for *this* dataset, *this* augmentation, *this*
optimiser, *this* learning rate, and *this* epoch budget. Change any of them and the
ranking can reorder.

The optimiser case is the sharpest. This project defaults to AdamW because it is forgiving
of architectural variation — which is exactly what makes the ranking a ranking *under
AdamW*. An architecture that suits SGD with a tuned schedule may be systematically
undervalued.

Short budgets add their own bias: they favour architectures that converge quickly, which is
not the same as architectures that converge well. See
[successive halving's failure modes](successive-halving.md#how-the-assumption-fails).

**In this project:** the full training recipe is persisted with the search and printed in
the report, so a result is always accompanied by the recipe it is a result *for*.

---

## 10. Reproducibility requires more than setting one random seed

**The misconception:** "We set `random.seed(42)`, so the run is reproducible."

**What is true:** A NAS run touches at least six independent randomness sources — Python's
`random`, NumPy's global generator, PyTorch's CPU RNG, PyTorch's CUDA RNGs, DataLoader
worker processes, and the search strategy's own generator. Seeding one leaves the rest free.

And even with all six seeded, floating-point reduction order inside kernels is not
determined by seeds. `(a + b) + c` and `a + (b + c)` differ in the last bits, which
occasionally flips an `argmax`.

**What is achievable:**

| Guarantee                                             | Achievable?                           |
| ----------------------------------------------------- | ------------------------------------- |
| Same *decisions* on the same machine, sequential      | Yes                                   |
| Bit-identical metrics on the same machine, sequential | Usually                               |
| Bit-identical metrics across machines                 | **No**                                |
| Identical results under multiprocessing               | **No** — observation order varies     |
| Conclusions that survive a change of seed             | Only if you check, with several seeds |

**In this project:** [reproducibility](reproducibility.md) covers seeding in full, and
[`tests/regression/test_determinism.py`](../../tests/regression/test_determinism.py)
asserts each achievable guarantee and documents each unachievable one.

---

## A short checklist before believing a result

- [ ] Is the search space described, including its size and its biases?
- [ ] Is the total search compute reported, not just the final training run?
- [ ] Was the test split used exactly once?
- [ ] Is the validation split large enough that the reported difference exceeds its
      standard error?
- [ ] Was random search run on the same space at the same budget?
- [ ] Were several seeds used, and are conclusions drawn from distributions?
- [ ] Are latency numbers labelled with the hardware they came from?
- [ ] Is the full training recipe reported?

Every "no" weakens the claim. Several "no"s mean there is not yet a result.

## See also

- [NAS foundations](nas-foundations.md) — the theory these pitfalls follow from.
- [Interpreting results](../guides/interpreting-results.md) — applying this to a report.
- [Reproducibility](reproducibility.md) — the seeding story in full.
