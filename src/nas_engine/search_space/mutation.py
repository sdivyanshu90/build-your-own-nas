"""Mutation operators for evolutionary architecture search.

What a mutation must guarantee
------------------------------
1. **Purity.** The parent is never modified. Genotypes are frozen and every operator
   builds a new object, so a parent still sitting in the population cannot be corrupted
   by a child's creation. This is asserted by a property test.
2. **Locality.** A mutation changes one decision. Large jumps turn evolution into
   random search and destroy the local-search signal that makes it work.
3. **Closure.** The child must remain inside the search space. Operators only choose
   from the space's own choice sets, and :func:`~nas_engine.search_space.repair.repair_architecture`
   restores global invariants afterwards.
4. **Progress.** A mutation that returns the parent unchanged wastes a generation, so
   every operator either produces a genuinely different genotype or declines by
   returning ``None``.

Applicability
-------------
Not every operator applies to every parent: there is no expansion ratio to change if the
architecture contains no depthwise-separable block, and depth cannot grow beyond
``blocks_per_stage``. Operators therefore *declare inapplicability* rather than failing.
:class:`MutationOperator` samples uniformly from the operators that are applicable to
the given parent, which keeps the effective mutation distribution well defined even as
architectures change shape.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import (
    ArchitectureSpec,
    BlockSpec,
    StageSpec,
)
from nas_engine.architectures.types import OperationType
from nas_engine.exceptions import MutationError
from nas_engine.search_space.repair import repair_architecture, stage_widths
from nas_engine.search_space.space import SearchSpace
from nas_engine.search_space.validation import check_architecture
from nas_engine.utilities.seeding import rng_state_from_json, rng_state_to_json

#: Version of the mutation operator's checkpoint payload.
MUTATION_STATE_VERSION: int = 1

#: Default number of mutation attempts before declaring failure for one parent.
DEFAULT_MAX_MUTATION_ATTEMPTS: int = 50


@dataclass(frozen=True)
class MutationResult:
    """The outcome of one successful mutation.

    Attributes:
        child: The mutated architecture.
        operator: Name of the operator that produced it.
        description: Human-readable summary persisted with the candidate, e.g.
            ``"kernel_size s1b0: 3 -> 5"``.
        parent_hash: Hash of the parent architecture.
    """

    child: ArchitectureSpec
    operator: str
    description: str
    parent_hash: str


#: An operator takes ``(spec, space, rng)`` and returns ``(child, description)`` or
#: ``None`` when it does not apply to this parent.
MutationFunction = Callable[
    [ArchitectureSpec, SearchSpace, random.Random],
    "tuple[ArchitectureSpec, str] | None",
]


def _block_input_channels(spec: ArchitectureSpec) -> list[list[int]]:
    """Return the input channel count of every block, indexed ``[stage][block]``.

    Args:
        spec: Architecture to analyse.

    Returns:
        Nested lists mirroring the stage/block structure.
    """
    result: list[list[int]] = []
    current = spec.stem.out_channels
    for stage in spec.stages:
        stage_inputs: list[int] = []
        for block in stage.blocks:
            stage_inputs.append(current)
            current = block.out_channels
        result.append(stage_inputs)
    return result


def _all_block_positions(spec: ArchitectureSpec) -> list[tuple[int, int]]:
    """Return every ``(stage_index, block_index)`` pair."""
    return [
        (stage_index, block_index)
        for stage_index, stage in enumerate(spec.stages)
        for block_index in range(len(stage.blocks))
    ]


def _choose_other(rng: random.Random, options: tuple[Any, ...], current: Any) -> Any | None:
    """Pick uniformly from ``options`` excluding ``current``.

    Args:
        rng: Generator to draw from.
        options: Available choices.
        current: Value to exclude.

    Returns:
        A different option, or ``None`` if no alternative exists.
    """
    alternatives = [option for option in options if option != current]
    if not alternatives:
        return None
    return rng.choice(alternatives)


# ---------------------------------------------------------------------------------
# Block-level operators
# ---------------------------------------------------------------------------------
def mutate_operation(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Replace one block's primitive operation.

    Only operations legal at the chosen position are offered: a channel-preserving
    operation cannot replace a block that changes the width, and identity cannot occupy
    a strided position.

    Args:
        spec: Parent architecture.
        space: Space supplying the operation menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None`` when no position admits an alternative.
    """
    inputs = _block_input_channels(spec)
    positions = _all_block_positions(spec)
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        in_channels = inputs[stage_index][block_index]
        preserves_channels = in_channels == block.out_channels
        candidates = [
            operation
            for operation in space.block.operations
            if operation is not block.operation
            and (preserves_channels or operation.can_change_channels)
            and not (block.stride > 1 and operation is OperationType.IDENTITY)
        ]
        if not candidates:
            continue
        operation = rng.choice(candidates)
        child_block = block.evolve(operation=operation)
        child = spec.with_block(stage_index, block_index, child_block)
        description = (
            f"operation s{stage_index}b{block_index}: {block.operation.value} -> {operation.value}"
        )
        return child, description
    return None


def mutate_kernel_size(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change the kernel size of one block that actually uses one.

    Args:
        spec: Parent architecture.
        space: Space supplying the kernel menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    positions = [
        (stage_index, block_index)
        for stage_index, block_index in _all_block_positions(spec)
        if spec.stages[stage_index].blocks[block_index].operation.uses_kernel_size
    ]
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        kernel = _choose_other(rng, space.block.kernel_sizes, block.kernel_size)
        if kernel is None:
            continue
        child = spec.with_block(stage_index, block_index, block.evolve(kernel_size=kernel))
        return child, f"kernel_size s{stage_index}b{block_index}: {block.kernel_size} -> {kernel}"
    return None


def mutate_expansion_ratio(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change the inverted-bottleneck ratio of one depthwise-separable block.

    Args:
        spec: Parent architecture.
        space: Space supplying the ratio menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    positions = [
        (stage_index, block_index)
        for stage_index, block_index in _all_block_positions(spec)
        if spec.stages[stage_index].blocks[block_index].operation.uses_expansion_ratio
    ]
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        ratio = _choose_other(rng, space.block.expansion_ratios, block.expansion_ratio)
        if ratio is None:
            continue
        child = spec.with_block(stage_index, block_index, block.evolve(expansion_ratio=ratio))
        return (
            child,
            f"expansion s{stage_index}b{block_index}: {block.expansion_ratio:g} -> {ratio:g}",
        )
    return None


def mutate_activation(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change the nonlinearity of one parametric block.

    Args:
        spec: Parent architecture.
        space: Space supplying the activation menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    positions = [
        (stage_index, block_index)
        for stage_index, block_index in _all_block_positions(spec)
        if spec.stages[stage_index].blocks[block_index].operation.is_parametric
    ]
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        activation = _choose_other(rng, space.block.activations, block.activation)
        if activation is None:
            continue
        child = spec.with_block(stage_index, block_index, block.evolve(activation=activation))
        return (
            child,
            f"activation s{stage_index}b{block_index}: "
            f"{block.activation.value} -> {activation.value}",
        )
    return None


def mutate_normalization(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change the normalisation layer of one parametric block.

    Args:
        spec: Parent architecture.
        space: Space supplying the normalisation menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    positions = [
        (stage_index, block_index)
        for stage_index, block_index in _all_block_positions(spec)
        if spec.stages[stage_index].blocks[block_index].operation.is_parametric
    ]
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        norm = _choose_other(rng, space.block.normalizations, block.normalization)
        if norm is None:
            continue
        child = spec.with_block(stage_index, block_index, block.evolve(normalization=norm))
        return (
            child,
            f"normalization s{stage_index}b{block_index}: "
            f"{block.normalization.value} -> {norm.value}",
        )
    return None


def mutate_residual(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Toggle a residual connection where one is legal.

    Args:
        spec: Parent architecture.
        space: Space; residuals must be enabled.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    if not space.block.allow_residual:
        return None
    inputs = _block_input_channels(spec)
    positions = _all_block_positions(spec)
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        if block.operation is OperationType.IDENTITY:
            continue
        legal = block.stride == 1 and inputs[stage_index][block_index] == block.out_channels
        if not legal:
            continue
        child_block = block.evolve(use_residual=not block.use_residual)
        child = spec.with_block(stage_index, block_index, child_block)
        state = "on" if child_block.use_residual else "off"
        return child, f"residual s{stage_index}b{block_index}: {state}"
    return None


def mutate_stride(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change the stride of a stage's first block.

    Only the first block of a stage downsamples, matching the sampler's convention. The
    child is repaired afterwards, which reverts the change if it would shrink the feature
    map below one pixel.

    Args:
        spec: Parent architecture.
        space: Space supplying the stride menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    if len(space.stage_strides) < 2:
        return None
    stage_indices = list(range(len(spec.stages)))
    rng.shuffle(stage_indices)
    for stage_index in stage_indices:
        block = spec.stages[stage_index].blocks[0]
        if block.operation is OperationType.IDENTITY:
            continue
        stride = _choose_other(rng, space.stage_strides, block.stride)
        if stride is None:
            continue
        child = spec.with_block(stage_index, 0, block.evolve(stride=stride))
        return child, f"stride s{stage_index}: {block.stride} -> {stride}"
    return None


# ---------------------------------------------------------------------------------
# Macro operators
# ---------------------------------------------------------------------------------
def mutate_stage_width(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change one stage's width, respecting the monotonic-width rule.

    Args:
        spec: Parent architecture.
        space: Space supplying the width menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    widths = list(stage_widths(spec))
    stage_indices = list(range(len(widths)))
    rng.shuffle(stage_indices)
    for stage_index in stage_indices:
        lower = widths[stage_index - 1] if space.monotonic_widths and stage_index > 0 else 0
        upper = (
            widths[stage_index + 1]
            if space.monotonic_widths and stage_index + 1 < len(widths)
            else None
        )
        options = tuple(
            width
            for width in space.stage_channels
            if width >= lower and (upper is None or width <= upper)
        )
        new_width = _choose_other(rng, options, widths[stage_index])
        if new_width is None:
            continue
        old_width = widths[stage_index]
        widths[stage_index] = new_width
        child, _ = repair_architecture(spec, target_widths=tuple(widths))
        return child, f"width s{stage_index}: {old_width} -> {new_width}"
    return None


def mutate_stage_depth(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Add or remove a block in one stage.

    A new block is inserted at the end of the stage with stride 1 and the stage's width,
    so it is legal by construction. Removal never empties a stage.

    Args:
        spec: Parent architecture.
        space: Space supplying the depth menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    stage_indices = list(range(len(spec.stages)))
    rng.shuffle(stage_indices)
    for stage_index in stage_indices:
        stage = spec.stages[stage_index]
        depth = len(stage.blocks)
        options = [value for value in space.blocks_per_stage if abs(value - depth) == 1]
        if not options:
            continue
        target = rng.choice(options)
        blocks = list(stage.blocks)
        if target > depth:
            width = stage.out_channels
            operation = rng.choice(space.block.parametric_operations)
            blocks.append(
                BlockSpec(
                    operation=operation,
                    kernel_size=rng.choice(space.block.kernel_sizes),
                    expansion_ratio=rng.choice(space.block.expansion_ratios),
                    out_channels=width,
                    stride=1,
                    use_residual=space.block.allow_residual,
                    normalization=rng.choice(space.block.normalizations),
                    activation=rng.choice(space.block.activations),
                )
            )
            action = "grow"
        else:
            if depth <= 1:
                continue
            # Never drop the first block: it owns the stage's stride and its width change.
            blocks.pop(rng.randrange(1, depth))
            action = "shrink"
        stages = list(spec.stages)
        stages[stage_index] = StageSpec(blocks=tuple(blocks))
        child, _ = repair_architecture(spec.with_stages(tuple(stages)))
        return child, f"depth s{stage_index}: {action} {depth} -> {len(blocks)}"
    return None


def mutate_num_stages(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Add a stage at the end or remove the last stage.

    Args:
        spec: Parent architecture.
        space: Space supplying the stage-count menu.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    current = len(spec.stages)
    options = [value for value in space.num_stages if abs(value - current) == 1]
    if not options:
        return None
    target = rng.choice(options)
    stages = list(spec.stages)
    if target > current:
        last_width = stages[-1].out_channels
        widths = [width for width in space.stage_channels if width >= last_width] or [last_width]
        width = rng.choice(widths)
        operation = rng.choice(space.block.parametric_operations)
        new_stage = StageSpec(
            blocks=(
                BlockSpec(
                    operation=operation,
                    kernel_size=rng.choice(space.block.kernel_sizes),
                    expansion_ratio=rng.choice(space.block.expansion_ratios),
                    out_channels=width,
                    stride=rng.choice(space.stage_strides),
                    use_residual=False,
                    normalization=rng.choice(space.block.normalizations),
                    activation=rng.choice(space.block.activations),
                ),
            )
        )
        stages.append(new_stage)
        action = "append"
    else:
        if current <= 1:
            return None
        stages.pop()
        action = "drop"
    child, _ = repair_architecture(spec.with_stages(tuple(stages)))
    return child, f"stages: {action} {current} -> {len(stages)}"


def mutate_stem(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change one stem field.

    Args:
        spec: Parent architecture.
        space: Space supplying the stem menus.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    fields: list[tuple[str, tuple[Any, ...], Any]] = [
        ("out_channels", space.stem.out_channels, spec.stem.out_channels),
        ("kernel_size", space.stem.kernel_sizes, spec.stem.kernel_size),
        ("stride", space.stem.strides, spec.stem.stride),
        ("normalization", space.stem.normalizations, spec.stem.normalization),
        ("activation", space.stem.activations, spec.stem.activation),
    ]
    rng.shuffle(fields)
    for name, options, current in fields:
        replacement = _choose_other(rng, options, current)
        if replacement is None:
            continue
        child_spec = spec.evolve(stem=spec.stem.evolve(**{name: replacement}))
        child, _ = repair_architecture(child_spec)
        current_text = getattr(current, "value", current)
        replacement_text = getattr(replacement, "value", replacement)
        return child, f"stem.{name}: {current_text} -> {replacement_text}"
    return None


def mutate_head(
    spec: ArchitectureSpec, space: SearchSpace, rng: random.Random
) -> tuple[ArchitectureSpec, str] | None:
    """Change one classifier-head field.

    Args:
        spec: Parent architecture.
        space: Space supplying the head menus.
        rng: Generator.

    Returns:
        ``(child, description)`` or ``None``.
    """
    fields: list[tuple[str, tuple[Any, ...], Any]] = [
        ("pooling", space.head.poolings, spec.head.pooling),
        ("hidden_units", space.head.hidden_units, spec.head.hidden_units),
        ("dropout", space.head.dropouts, spec.head.dropout),
    ]
    if spec.head.hidden_units > 0:
        fields.append(("activation", space.head.activations, spec.head.activation))
    rng.shuffle(fields)
    for name, options, current in fields:
        replacement = _choose_other(rng, options, current)
        if replacement is None:
            continue
        child = spec.evolve(head=spec.head.evolve(**{name: replacement}))
        current_text = getattr(current, "value", current)
        replacement_text = getattr(replacement, "value", replacement)
        return child, f"head.{name}: {current_text} -> {replacement_text}"
    return None


#: The default operator set, in a fixed order so that a seeded run is reproducible.
DEFAULT_OPERATORS: tuple[tuple[str, MutationFunction], ...] = (
    ("operation", mutate_operation),
    ("kernel_size", mutate_kernel_size),
    ("expansion_ratio", mutate_expansion_ratio),
    ("activation", mutate_activation),
    ("normalization", mutate_normalization),
    ("residual", mutate_residual),
    ("stride", mutate_stride),
    ("stage_width", mutate_stage_width),
    ("stage_depth", mutate_stage_depth),
    ("num_stages", mutate_num_stages),
    ("stem", mutate_stem),
    ("head", mutate_head),
)


@dataclass
class MutationStatistics:
    """Counters describing mutation behaviour.

    Attributes:
        attempts: Mutation attempts made.
        successes: Attempts that produced a valid, novel child.
        failures: Attempts that produced nothing usable.
        by_operator: Successful mutations per operator name.
        rejected_by_operator: Rejected mutations per operator name.
    """

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    by_operator: Counter[str] = field(default_factory=Counter)
    rejected_by_operator: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "by_operator": dict(self.by_operator),
            "rejected_by_operator": dict(self.rejected_by_operator),
        }


class MutationOperator:
    """Applies a randomly chosen, applicable mutation to a parent architecture.

    Args:
        space: The search space the child must remain inside.
        seed: Seed for the private generator.
        max_attempts: Mutation attempts before declaring failure for one parent.
        operators: Override the operator set; defaults to :data:`DEFAULT_OPERATORS`.

    Raises:
        ValueError: If ``max_attempts`` is not positive or ``operators`` is empty.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        seed: int,
        max_attempts: int = DEFAULT_MAX_MUTATION_ATTEMPTS,
        operators: tuple[tuple[str, MutationFunction], ...] | None = None,
    ) -> None:
        if max_attempts < 1:
            msg = f"max_attempts must be >= 1, received {max_attempts}"
            raise ValueError(msg)
        chosen = operators if operators is not None else DEFAULT_OPERATORS
        if not chosen:
            msg = "at least one mutation operator is required"
            raise ValueError(msg)
        self._space = space
        self._seed = seed
        self._max_attempts = max_attempts
        self._operators = chosen
        self._rng = random.Random(seed)  # noqa: S311 - reproducibility, not security
        self._statistics = MutationStatistics()

    @property
    def statistics(self) -> MutationStatistics:
        """Live mutation statistics."""
        return self._statistics

    @property
    def operator_names(self) -> tuple[str, ...]:
        """Names of the configured operators."""
        return tuple(name for name, _ in self._operators)

    def try_mutate(self, parent: ArchitectureSpec) -> MutationResult | None:
        """Attempt to mutate ``parent``, returning ``None`` on failure.

        Args:
            parent: Architecture to mutate. Never modified.

        Returns:
            A :class:`MutationResult`, or ``None`` if no attempt succeeded within budget.
        """
        parent_hash = architecture_hash(parent)
        operators = list(self._operators)
        for _ in range(self._max_attempts):
            self._statistics.attempts += 1
            name, operator = operators[self._rng.randrange(len(operators))]
            outcome = operator(parent, self._space, self._rng)
            if outcome is None:
                self._statistics.rejected_by_operator[name] += 1
                continue
            child, description = outcome
            repaired, _ = repair_architecture(child)
            if architecture_hash(repaired) == parent_hash:
                # Repair undid the change (e.g. a stride that would shrink the map below
                # 1x1). Treat it as a no-op rather than reporting a mutation that did
                # nothing.
                self._statistics.rejected_by_operator[name] += 1
                continue
            report = check_architecture(repaired, self._space)
            if not report.is_valid:
                self._statistics.rejected_by_operator[name] += 1
                continue
            self._statistics.successes += 1
            self._statistics.by_operator[name] += 1
            return MutationResult(
                child=repaired,
                operator=name,
                description=description,
                parent_hash=parent_hash,
            )
        self._statistics.failures += 1
        return None

    def mutate(self, parent: ArchitectureSpec) -> MutationResult:
        """Mutate ``parent``, raising when no valid child can be produced.

        Args:
            parent: Architecture to mutate.

        Returns:
            A :class:`MutationResult`.

        Raises:
            MutationError: If no operator produced a valid, novel child within budget.
                The most likely causes are a search space with only one choice per
                dimension, or constraints so tight that every neighbour is infeasible.
        """
        result = self.try_mutate(parent)
        if result is not None:
            return result
        msg = (
            f"no valid mutation found for architecture {architecture_hash(parent)} in "
            f"{self._max_attempts} attempts. Rejections by operator: "
            f"{dict(self._statistics.rejected_by_operator)}. Widen the search space or "
            "relax its constraints."
        )
        raise MutationError(
            msg,
            details={
                "parent_hash": architecture_hash(parent),
                "max_attempts": self._max_attempts,
                "rejected_by_operator": dict(self._statistics.rejected_by_operator),
            },
        )

    # -- checkpointing -------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the operator's state."""
        return {
            "version": MUTATION_STATE_VERSION,
            "seed": self._seed,
            "rng": rng_state_to_json(self._rng),
            "statistics": self._statistics.to_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore operator state from :meth:`state_dict` output.

        Args:
            payload: Previously captured state.

        Raises:
            MutationError: If the payload version is unsupported or malformed.
        """
        version = payload.get("version")
        if version != MUTATION_STATE_VERSION:
            msg = (
                f"mutation state version {version} is not supported by this build "
                f"(expected {MUTATION_STATE_VERSION})"
            )
            raise MutationError(msg, details={"version": version})
        try:
            self._rng = rng_state_from_json(payload["rng"])
        except (KeyError, ValueError) as exc:
            msg = f"mutation state could not be restored: {exc}"
            raise MutationError(msg, details={"error": str(exc)}) from exc
        stats = payload.get("statistics", {})
        self._statistics = MutationStatistics(
            attempts=int(stats.get("attempts", 0)),
            successes=int(stats.get("successes", 0)),
            failures=int(stats.get("failures", 0)),
            by_operator=Counter(stats.get("by_operator", {})),
            rejected_by_operator=Counter(stats.get("rejected_by_operator", {})),
        )


__all__ = [
    "DEFAULT_MAX_MUTATION_ATTEMPTS",
    "DEFAULT_OPERATORS",
    "MUTATION_STATE_VERSION",
    "MutationFunction",
    "MutationOperator",
    "MutationResult",
    "MutationStatistics",
    "mutate_activation",
    "mutate_expansion_ratio",
    "mutate_head",
    "mutate_kernel_size",
    "mutate_normalization",
    "mutate_num_stages",
    "mutate_operation",
    "mutate_residual",
    "mutate_stage_depth",
    "mutate_stage_width",
    "mutate_stem",
    "mutate_stride",
]
