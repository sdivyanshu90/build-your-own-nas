r"""Weight initialisation.

Why initialisation matters for NAS specifically
-----------------------------------------------
In ordinary training, a poor initialisation costs some epochs. In NAS it costs
*correctness of the ranking*: candidates are trained for a handful of epochs, so an
architecture that merely starts badly is indistinguishable from one that is genuinely
worse. A principled, scale-preserving initialisation removes a large source of ranking
noise that has nothing to do with architecture quality.

He (Kaiming) initialisation
---------------------------
For a layer with :math:`n_{out} = k^2 C_{out}` outgoing connections per unit, weights are
drawn from :math:`\mathcal{N}(0, 2/n_{out})`. The factor 2 compensates for ReLU zeroing
roughly half of its inputs: without it, activation variance shrinks by a factor of 2 at
every layer and vanishes exponentially with depth. ``fan_out`` mode is used because it
preserves variance in the *backward* pass, which is the direction that matters for
gradient flow through the deep stacks this space can produce.

This is the right choice for the ReLU family (ReLU, ReLU6, SiLU, GELU, Hardswish); all of
them are approximately half-rectifying. Xavier/Glorot initialisation, which assumes a
symmetric activation like tanh, would under-scale them.

Zero-initialised residual branches
----------------------------------
When a block has a residual connection, the affine weight of its **last** normalisation
layer is initialised to zero. The block therefore computes ``x + 0 == x`` at step zero:
the network starts as a shallow identity mapping and *learns* to use its depth. This
trick (from "Bag of Tricks for Image Classification") measurably stabilises early
training, which is exactly the regime NAS operates in. It changes parameter *values*
only — never counts — so the analytic cost model is unaffected.

Determinism
-----------
Initialisation draws from the global PyTorch RNG. Seeding it via
:func:`nas_engine.utilities.seeding.seed_everything` before building a model makes weights
reproducible; the same architecture built twice with the same seed has bit-identical
weights on the same platform.
"""

from __future__ import annotations

from torch import nn

#: Standard deviation for classifier weights. A small value keeps the initial logits
#: near zero, so the initial loss is close to ``log(num_classes)`` and the first
#: optimiser steps are not dominated by an arbitrary large-margin prediction.
CLASSIFIER_INIT_STD: float = 0.01


def initialize_module(module: nn.Module) -> None:
    """Apply the project's initialisation policy to a single module.

    Intended for use with :meth:`torch.nn.Module.apply`.

    Args:
        module: Module to initialise. Types with no policy are left untouched.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=CLASSIFIER_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _last_normalization(module: nn.Module) -> nn.Module | None:
    """Return the last normalisation layer inside ``module``, if any.

    Args:
        module: Module to search, in definition order.

    Returns:
        The final :class:`torch.nn.BatchNorm2d` or :class:`torch.nn.GroupNorm`
        descendant, or ``None``.
    """
    found: nn.Module | None = None
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm2d, nn.GroupNorm)) and child.weight is not None:
            found = child
    return found


def zero_initialize_residual_branches(model: nn.Module) -> int:
    """Zero the final normalisation weight of every residual block.

    Args:
        model: A built network.

    Returns:
        The number of blocks that were zero-initialised, for logging and testing.
    """
    from nas_engine.models.blocks import NasBlock

    count = 0
    for module in model.modules():
        if not isinstance(module, NasBlock) or not module.use_residual:
            continue
        normalization = _last_normalization(module.operation)
        if normalization is None or normalization.weight is None:
            continue
        nn.init.zeros_(normalization.weight)
        count += 1
    return count


def initialize_weights(model: nn.Module, *, zero_init_residual: bool = True) -> None:
    """Initialise every parameter in ``model`` according to the project policy.

    Args:
        model: Network to initialise, modified in place.
        zero_init_residual: Whether to zero residual branches' last normalisation weight.
    """
    model.apply(initialize_module)
    if zero_init_residual:
        zero_initialize_residual_branches(model)


__all__ = [
    "CLASSIFIER_INIT_STD",
    "initialize_module",
    "initialize_weights",
    "zero_initialize_residual_branches",
]
