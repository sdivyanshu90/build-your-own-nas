"""Structural repair of architecture genotypes.

Mutation is *local*: it changes one field. But architectures have *global* invariants —
a channel-preserving operation must sit where the width does not change, a residual
needs matching shapes, and the product of strides must not shrink the feature map below
one pixel. A local change can break a global invariant several blocks away.

Two responses are possible:

**Reject and resample.** Simple, but wasteful. Changing the width of stage 1 invalidates
every downstream pooling block, so a large fraction of otherwise sensible mutations
would be thrown away, and the mutation distribution would be quietly skewed towards
whichever changes happen to be locally safe.

**Repair.** Apply the local change, then restore the global invariants with a
deterministic, minimal rewrite. This is what the project does. Repair is:

* *Deterministic* — the same input always produces the same output, so it cannot inject
  hidden randomness into a seeded search.
* *Minimal* — it only touches fields that are actually inconsistent.
* *Idempotent* — ``repair(repair(x)) == repair(x)``, verified by property tests.

The cost is that a mutation's effect is not always exactly what was requested: widening
stage 1 also rewrites the declared widths of that stage's pooling blocks. The mutation
record therefore describes the *requested* change, and the resulting hash is the source
of truth for identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from nas_engine.architectures.shapes import conv_output_size
from nas_engine.architectures.spec import ArchitectureSpec, BlockSpec, StageSpec


@dataclass(frozen=True)
class RepairReport:
    """What :func:`repair_architecture` had to change.

    Attributes:
        changed: Whether anything was rewritten.
        channel_fixes: Locations whose ``out_channels`` was corrected.
        residual_fixes: Locations whose illegal residual was removed.
        stride_fixes: Locations whose stride was reduced to keep the map above 1x1.
    """

    changed: bool
    channel_fixes: tuple[str, ...] = ()
    residual_fixes: tuple[str, ...] = ()
    stride_fixes: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return a short human-readable description of the repairs performed."""
        if not self.changed:
            return "no repairs required"
        parts: list[str] = []
        if self.channel_fixes:
            parts.append(f"channels@{','.join(self.channel_fixes)}")
        if self.residual_fixes:
            parts.append(f"residual@{','.join(self.residual_fixes)}")
        if self.stride_fixes:
            parts.append(f"stride@{','.join(self.stride_fixes)}")
        return "repaired " + "; ".join(parts)


def stage_widths(spec: ArchitectureSpec) -> tuple[int, ...]:
    """Return the effective width of each stage.

    After repair, the last block of a stage always carries that stage's effective
    width, because channel-preserving blocks copy the running width forward. This
    function therefore reads widths straight off the genotype.

    Args:
        spec: Architecture to inspect.

    Returns:
        One width per stage.
    """
    return tuple(stage.out_channels for stage in spec.stages)


def _intended_width(stage: StageSpec, incoming: int) -> int:
    """Return the width a stage is trying to produce.

    Args:
        stage: The stage to inspect.
        incoming: Channel count entering the stage.

    Returns:
        The ``out_channels`` of the first channel-changing block, or ``incoming`` when the
        stage contains none.
    """
    for block in stage.blocks:
        if block.operation.can_change_channels:
            return block.out_channels
    return incoming


def repair_architecture(
    spec: ArchitectureSpec, *, target_widths: tuple[int, ...] | None = None
) -> tuple[ArchitectureSpec, RepairReport]:
    """Restore structural invariants, returning a repaired copy and a report.

    Invariants enforced, in order:

    1. **Channel consistency.** Every channel-preserving operation (identity, pooling)
       declares an ``out_channels`` equal to its input width. Parametric operations
       adopt the stage's target width.
    2. **Residual legality.** ``use_residual`` is cleared wherever the block's input and
       output shapes differ.
    3. **Resolution floor.** Strides are reduced to 1, from the last strided block
       backwards, until the final feature map is at least 1x1.

    ``spec`` is never modified; genotypes are frozen and a new object is returned.

    Args:
        spec: Architecture to repair.
        target_widths: Desired width per stage. Defaults to each stage's current
            effective width, which makes repair a pure consistency pass.

    Returns:
        A ``(repaired_spec, report)`` tuple. When nothing needed changing, the returned
        specification is ``spec`` itself.

    Raises:
        ValueError: If ``target_widths`` has the wrong length.
    """
    if target_widths is not None and len(target_widths) != len(spec.stages):
        msg = (
            f"target_widths has {len(target_widths)} entries but the architecture has "
            f"{len(spec.stages)} stages"
        )
        raise ValueError(msg)

    channel_fixes: list[str] = []
    residual_fixes: list[str] = []
    stride_fixes: list[str] = []

    current_channels = spec.stem.out_channels
    current_size = conv_output_size(spec.input_size, spec.stem.kernel_size, spec.stem.stride)
    new_stages: list[StageSpec] = []

    for stage_index, stage in enumerate(spec.stages):
        # The stage's intended width comes from its *first* channel-changing block, which
        # is the convention the sampler and every mutation operator follow. Reading it from
        # the last block instead would let a corrupt trailing pooling block silently
        # redefine the whole stage.
        width = (
            target_widths[stage_index]
            if target_widths is not None
            else _intended_width(stage, current_channels)
        )
        new_blocks: list[BlockSpec] = []
        for block_index, block in enumerate(stage.blocks):
            location = f"s{stage_index}b{block_index}"
            changes: dict[str, object] = {}

            desired_channels = width if block.operation.can_change_channels else current_channels
            if block.out_channels != desired_channels:
                changes["out_channels"] = desired_channels
                channel_fixes.append(location)

            output_size = conv_output_size(current_size, block.kernel_size, block.stride)
            stride = block.stride
            if output_size < 1:
                changes["stride"] = 1
                stride = 1
                output_size = conv_output_size(current_size, block.kernel_size, 1)
                stride_fixes.append(location)

            shape_preserved = stride == 1 and desired_channels == current_channels
            if block.use_residual and not shape_preserved:
                changes["use_residual"] = False
                residual_fixes.append(location)

            repaired = block.evolve(**changes) if changes else block
            new_blocks.append(repaired)
            current_channels = desired_channels
            current_size = output_size

        new_stages.append(stage.evolve(blocks=tuple(new_blocks)))

    changed = bool(channel_fixes or residual_fixes or stride_fixes)
    report = RepairReport(
        changed=changed,
        channel_fixes=tuple(channel_fixes),
        residual_fixes=tuple(residual_fixes),
        stride_fixes=tuple(stride_fixes),
    )
    if not changed:
        return spec, report
    return spec.with_stages(tuple(new_stages)), report


__all__ = ["RepairReport", "repair_architecture", "stage_widths"]
