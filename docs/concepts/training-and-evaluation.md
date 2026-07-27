# Training and evaluation

Every neural-network concept this project uses, connected to where it appears in the code
and why it was chosen.

---

## Part 1: the building blocks

### Convolution

A convolution slides a small learned filter across the input and computes a dot product at
each position. For an input with $C_{in}$ channels and a $k \times k$ kernel producing
$C_{out}$ channels:

$$
y_{o,i,j} = \sum_{c=1}^{C_{in}} \sum_{u=1}^{k} \sum_{v=1}^{k}
   w_{o,c,u,v} \cdot x_{c,\, i+u-p,\, j+v-p} \;+\; b_o
$$

| Term | Meaning |
| --- | --- |
| **Channels** | Feature maps. Input images have 3 (RGB); intermediate layers have as many as the architecture specifies |
| **Kernel** | The learned filter. $k \times k$ spatial extent, applied across all input channels |
| **Stride** | Step size. Stride 2 halves the output resolution |
| **Padding** | Border pixels added so the kernel can be centred at the edges. This project uses $p = \lfloor k/2 \rfloor$ ("same" padding), which is why kernels must be odd |
| **Receptive field** | The input region one output value depends on. Grows with depth and with stride |

Parameter count: $k^2 C_{in} C_{out}$ (plus $C_{out}$ if there is a bias).

**In the code:** [`models/operations.py`](../../src/nas_engine/models/operations.py) builds
them; [`architectures/cost.py`](../../src/nas_engine/architectures/cost.py) counts them
analytically; [`architectures/shapes.py`](../../src/nas_engine/architectures/shapes.py)
computes the output extent.

### Depthwise-separable convolution

A dense convolution does two jobs at once: mixing *spatially* within a $k \times k$
neighbourhood, and mixing *across channels*. Factorising them:

$$
\underbrace{k^2 C_{in}}_{\text{depthwise}} \;+\; \underbrace{C_{in} C_{out}}_{\text{pointwise}}
\qquad\text{versus}\qquad
k^2 C_{in} C_{out}
$$

For $k=3$, $C_{in}=C_{out}=64$: 4 672 parameters instead of 36 864 — an 8× reduction, at
some cost in representational power.

**Inverted bottleneck.** When `expansion_ratio > 1`, the block first *widens* with a 1×1
convolution, applies the depthwise convolution in the wider space, then projects back down.
The projection has normalisation but **no activation** — MobileNetV2's "linear bottleneck".

The argument: a ReLU zeroes roughly half its inputs. In a low-dimensional space that
destroys information which cannot be recovered; in a high-dimensional space the same
information survives in the surviving coordinates. So nonlinearity goes where the
representation is wide, and the narrow projection stays linear.

**In the code:** [`models/blocks.py`](../../src/nas_engine/models/blocks.py),
`SeparableConvBlock`.

### Pooling

Downsampling without parameters. Max pooling selects the strongest activation in each
window; average pooling smooths. Both preserve the channel count, which is why a
pooling block must sit where the width does not change — enforced by
[shape inference](architecture-encoding.md#shape-inference).

Average pooling uses `count_include_pad=False`, so border averages are not biased towards
zero by the padded region. That matters most at the small feature-map sizes appearing late
in these networks.

**Global pooling** in the head collapses the spatial dimensions to 1×1. This is what allows
a network to accept any input resolution — and therefore what makes
[resolution-based fidelity scaling](successive-halving.md#the-resource-ladder) possible.

### Normalisation

Normalisation rescales activations so their distribution stays stable through depth. Two
kinds are available:

**BatchNorm** normalises per channel over the batch and spatial dimensions:

$$
\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}},
\qquad y = \gamma \hat{x} + \beta
$$

Strong regulariser and optimisation aid. Its weakness is that the statistics depend on
batch size, so very small batches are unstable — and it maintains running statistics
(`running_mean`, `running_var`) that are saved in the state dict but never optimised.

**GroupNorm** normalises within channel groups of a *single* example. Batch-size
independent, which is useful for the tiny batches common in low-fidelity NAS evaluation
and for deployment at batch size 1. No running statistics.

**Convention:** a convolution followed by normalisation has **no bias**. The normalisation's
own shift parameter $\beta$ makes it redundant — it would be immediately cancelled by the
mean subtraction. A convolution with `NormalizationType.NONE` *does* carry a bias, because
otherwise the layer could not represent an affine shift at all.

That convention is load-bearing: the analytic cost model depends on it, and an exactness
test would fail immediately if it changed.

### Activation functions

| Function  | Definition            | Character                                        |
| --------- | --------------------- | ------------------------------------------------ |
| ReLU      | $\max(0, x)$          | Cheap, sparse, the historical default            |
| ReLU6     | $\min(\max(0,x), 6)$  | Bounded output; better low-precision robustness  |
| SiLU      | $x \cdot \sigma(x)$   | Smooth, non-monotonic; often more accurate       |
| GELU      | $x \cdot \Phi(x)$     | Smooth; standard in transformers                 |
| Hardswish | Piecewise-linear SiLU | Cheap approximation for mobile inference         |
| Identity  | $x$                   | The canonical value where there is no activation |

`inplace=True` is deliberately not used: in-place activations break residual branches that
need the pre-activation tensor, and the memory saving is irrelevant at these model sizes.

### Residual connections

$$
y = x + f(x)
$$

The gradient of the sum is the gradient of the identity *plus* the gradient through $f$, so
it cannot vanish through depth alone. That is what makes deep stacks trainable.

Only **identity** shortcuts are supported: input and output shapes must match exactly.
Projection shortcuts (a 1×1 convolution on the skip path) are excluded deliberately —
silently inserting parameters would make the parameter-count objective misleading. A
mismatched residual is a validation error with an actionable message, not a silent fix.

### Dropout

Randomly zeroes a fraction $p$ of activations during training, forcing the network not to
rely on any single unit. Disabled at evaluation time.

Placed immediately before the classifier, because that is where overfitting concentrates
and because placing it earlier would perturb BatchNorm's statistics.

### Weight initialisation

In ordinary training a poor initialisation costs some epochs. **In NAS it costs correctness
of the ranking**: candidates are trained for a handful of epochs, so an architecture that
merely starts badly is indistinguishable from one that is genuinely worse.

**He (Kaiming) initialisation.** Weights drawn from $\mathcal{N}(0, 2/n_{out})$ where
$n_{out} = k^2 C_{out}$. The factor 2 compensates for ReLU zeroing half its inputs; without
it, activation variance shrinks by a factor of 2 per layer and vanishes exponentially with
depth. `fan_out` mode preserves variance in the *backward* pass, which is the direction
that matters for gradient flow.

This is right for the ReLU family; Xavier/Glorot, which assumes a symmetric activation like
tanh, would under-scale them.

**Zero-initialised residual branches.** The affine weight of each residual block's *last*
normalisation starts at zero, so the block computes $x + 0 = x$ at step zero. The network
starts as a shallow identity mapping and learns to use its depth. It measurably stabilises
early training — exactly the regime NAS operates in — and changes parameter *values* only,
never counts.

**In the code:** [`models/initialization.py`](../../src/nas_engine/models/initialization.py).

---

## Part 2: optimisation

### Backpropagation

The chain rule applied to a computation graph. The forward pass computes the loss; the
backward pass computes $\partial \mathcal{L} / \partial w$ for every parameter by
propagating gradients from the loss backwards. PyTorch's autograd does this; nothing here
reimplements it.

### The loss

Cross-entropy — the negative log-probability the model assigns to the true class:

$$
\mathcal{L} = -\sum_{c} y_c \log \hat{p}_c
$$

**Label smoothing** replaces the one-hot target with a mixture of the target and the
uniform distribution. It discourages the network from driving logits to infinity and acts
as a mild regulariser.

### Optimisers

**SGD with momentum** is the reference for convolutional image classification: with a tuned
schedule it generalises at least as well as anything else. Its weakness for NAS is
sensitivity — a learning rate tuned for one architecture can be badly wrong for another,
and the search would then be ranking learning-rate compatibility rather than architecture
quality.

**AdamW** is the default here for exactly that reason. Per-parameter adaptive step sizes
make it far more forgiving of architectural variation, which matters when one training
recipe must be applied unchanged to hundreds of different networks.

The cost is a modest generalisation gap on some vision tasks — **a bias worth stating: the
ranking this framework produces is a ranking under AdamW, not an architecture-intrinsic
truth.**

*Decoupled weight decay.* AdamW applies the decay term directly to the weights rather than
folding it into the gradient. In plain Adam, L2 regularisation added to the gradient is
divided by the adaptive step size, so parameters with large gradients get *less*
regularisation — the opposite of the intent.

*No decay on normalisation and bias parameters.* Weight decay pulls parameters towards
zero. For a BatchNorm scale, zero means "delete this channel"; for a bias it removes the
layer's ability to shift its output. Neither is a useful prior. Parameters are split into
two groups and the no-decay group is exempt.

### Learning-rate schedules

Schedules matter more in NAS than elsewhere. A constant learning rate leaves every model
still oscillating when training stops, and the size of that oscillation depends on the
architecture — so the ranking picks up noise proportional to each model's curvature.
Annealing to near zero forces every candidate into a comparable "settled" state before it
is measured.

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
$$

Cosine (above), step, and constant are available. All are stepped **per optimiser step**,
not per epoch, so the schedule's shape does not change when the dataset fraction changes
under multi-fidelity evaluation.

Optional linear **warm-up** over the first few steps: adaptive optimisers have unreliable
second-moment estimates in the first iterations, which can produce a very large first step
and destabilise BatchNorm statistics.

### Gradient clipping

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
$$

Rescales the gradient when its global $L_2$ norm exceeds a threshold: direction preserved,
magnitude bounded.

It earns its place in NAS because the search deliberately proposes unusual networks — very
deep stacks without residuals, unnormalised convolutions — and a single exploding batch
would otherwise turn an interesting candidate into a `NaN` and lose the information it
carried.

### Overfitting and early stopping

**Overfitting** is fitting the training set's noise rather than its signal: training loss
keeps falling while validation loss rises.

**Early stopping** ends training when the monitored validation metric has not improved for
`patience` consecutive epochs. In NAS its value is not primarily regularisation — the
budgets are too short for much overfitting — it is **budget reallocation**: an architecture
that has plateaued at 30% will not reach 80% in the remaining epochs, and the compute is
better spent on the next candidate.

Two subtleties handled explicitly:

- `min_delta` guards against declaring improvement from noise. With a validation split of
  256 examples the standard error is around 3 percentage points, so a threshold should be
  set deliberately.
- The **best** state is what matters, not the last. The trainer restores the best epoch's
  weights, so a late-epoch collapse cannot be mistaken for the candidate's quality.

### Validation leakage

The three splits have three jobs:

| Split          | Job                                                                 |
| -------------- | ------------------------------------------------------------------- |
| **train**      | Fits the weights                                                    |
| **validation** | Ranks architectures during the search                               |
| **test**       | Touched **once**, after the search, to report the winner's accuracy |

Because the search *selects on* validation accuracy, that split stops being an unbiased
estimate of generalisation: with enough candidates something scores well by luck. This is
selection bias — the NAS analogue of overfitting.

The engine enforces the separation structurally: the evaluator is only ever handed the
training and validation loaders, and the test loader is reachable only through the explicit
`nas-engine evaluate` command. `include_test` defaults to `False` in
[`build_dataloaders`](../../src/nas_engine/datasets/loaders.py), so leaking the test split
requires deliberately passing a flag.

---

## Part 3: what the evaluator measures

Every completed evaluation records:

| Metric                                   | Meaning                          | Notes                                                  |
| ---------------------------------------- | -------------------------------- | ------------------------------------------------------ |
| `validation_accuracy`                    | Top-1 accuracy at the best epoch | The primary objective                                  |
| `validation_loss`                        | Cross-entropy at the best epoch  |                                                        |
| `validation_topk_accuracy`               | Top-$k$ accuracy                 | Changes more smoothly than top-1; a useful tie-breaker |
| `train_loss`                             | Final training loss              | Compare with validation loss to spot overfitting       |
| `trainable_parameters`                   | Optimised parameters             | Exact, from the analytic model                         |
| `non_trainable_parameters`               | Buffers (BatchNorm statistics)   | Saved but not optimised                                |
| `multiply_accumulates`                   | Estimated MACs per image         | Convolution and linear only                            |
| `model_size_bytes`                       | Serialised state-dict size       | Measured, not derived                                  |
| `latency_median_ms`                      | Median forward-pass latency      | **Machine-specific** — see below                       |
| `latency_p90_ms`, `latency_p99_ms`       | Tail latency                     | A large gap from the median means contention           |
| `epochs_completed`                       | Epochs actually run              | Lower than the budget means early stopping             |
| `training_seconds`, `evaluation_seconds` | Wall-clock                       |                                                        |
| `train_examples`, `effective_resolution` | Fidelity actually used           |                                                        |

### Metric aggregation

The mean of per-batch means is **not** the mean over examples unless every batch is the
same size. The final batch of an epoch is usually short, so naive averaging biases every
reported number. Every aggregator accumulates `value × batch_size` and divides by the total
example count.

### Latency measurement

**Latency numbers produced here are not portable.** They describe one model, on one
machine, at one thread count, with one batch size, under whatever else that machine was
doing. Every measurement carries its device metadata and an explicit warning, and reports
repeat the warning.

What it *is* good for is relative comparison within one search run on one machine — which
is what hardware-aware NAS needs.

Methodology:

1. **Warm-up.** The first forward passes are unrepresentative: allocators are cold, cuDNN
   may still be selecting algorithms, CPU frequency scaling has not settled. Warm-up
   iterations are discarded.
2. **Repeated timed blocks.** Timing a block of iterations rather than a single call
   amortises clock-read overhead, which is comparable to a small model's forward pass.
3. **Median, not mean.** Latency distributions have a long right tail from scheduler
   preemption and garbage collection. Both are reported, along with p90 and p99, so a
   suspiciously large gap is visible.
4. **Explicit synchronisation.** CUDA kernel launches are asynchronous; without
   `torch.cuda.synchronize` the measurement would time the launch, not the computation.
5. **Fixed thread count**, recorded in the metadata.

### Model size

Parameter count and on-disk size are related but not the same:

- Buffers are saved but are not parameters.
- Integer buffers such as `num_batches_tracked` are 8 bytes, not 4.
- The `.pt` format is a ZIP container with per-tensor metadata.

So the measurement serialises the real state dict rather than multiplying the parameter
count by four.

---

## Part 4: budgets and fidelity

A **budget** is the resources given to one evaluation, expressed along three independent
dimensions:

| Dimension        | Cost scaling            |
| ---------------- | ----------------------- |
| `epochs`         | Linear                  |
| `train_fraction` | Linear                  |
| `resolution`     | Quadratic (pixel count) |

Budgets are first-class serialisable values, persisted with every trial, because two
measurements of the same architecture at different fidelities must never be confused with
each other.

Subsets use a **seeded random** selection rather than a prefix: dataset files are often
ordered by class, and taking the first half would silently drop classes.

---

## Part 5: failure handling

| Failure                | Classification | Retriable | Why                                               |
| ---------------------- | -------------- | --------- | ------------------------------------------------- |
| Non-finite loss        | `DIVERGENCE`   | **No**    | The same seed and architecture will diverge again |
| Model build error      | `BUILD`        | No        | Structural                                        |
| Invalid architecture   | `VALIDATION`   | No        | Deterministic                                     |
| Constraint violation   | `CONSTRAINT`   | No        | The architecture is simply too expensive          |
| Out of memory          | `RESOURCE`     | Yes       | Depends on what else was resident                 |
| Timeout                | `TIMEOUT`      | Yes       | Depends on machine load                           |
| Worker crash           | `WORKER`       | Yes       | Infrastructure, not the candidate                 |
| Database write failure | `PERSISTENCE`  | Yes       | Transient lock contention                         |

Getting this wrong is expensive in both directions: retrying a permanent failure burns the
budget three times over, and giving up on a transient one silently loses a good candidate.

The evaluator **never raises** for a candidate-level problem. Every exception is caught,
classified, and returned as a failed result, so the orchestration loop has exactly one path
through it and a failing candidate cannot abort the search.

---

## Where this lives

| Concern        | File                                                                            |
| -------------- | ------------------------------------------------------------------------------- |
| Layers         | [`models/operations.py`](../../src/nas_engine/models/operations.py)             |
| Blocks         | [`models/blocks.py`](../../src/nas_engine/models/blocks.py)                     |
| Initialisation | [`models/initialization.py`](../../src/nas_engine/models/initialization.py)     |
| Optimisers     | [`training/optimizers.py`](../../src/nas_engine/training/optimizers.py)         |
| Schedules      | [`training/schedulers.py`](../../src/nas_engine/training/schedulers.py)         |
| Metrics        | [`training/metrics.py`](../../src/nas_engine/training/metrics.py)               |
| Early stopping | [`training/early_stopping.py`](../../src/nas_engine/training/early_stopping.py) |
| The loop       | [`training/trainer.py`](../../src/nas_engine/training/trainer.py)               |
| Latency        | [`evaluation/latency.py`](../../src/nas_engine/evaluation/latency.py)           |
| Model size     | [`evaluation/model_size.py`](../../src/nas_engine/evaluation/model_size.py)     |
| The evaluator  | [`evaluation/evaluator.py`](../../src/nas_engine/evaluation/evaluator.py)       |

## See also

- [Reproducibility](reproducibility.md) — how seeding makes these measurements repeatable.
- [Common pitfalls](common-pitfalls.md) — how to misread these numbers.
