"""Human-readable architecture summaries.

A 32-character hash is a good identifier and a terrible explanation. Every place a
person looks at a candidate — the CLI, the Markdown report, a failure log — should show
what the network actually *is*: its stages, its widths, where it downsamples, and what
it costs.

The renderer is deliberately plain text with fixed column widths rather than a Rich
table, so the same function serves terminal output, Markdown code fences, and log
messages without a rendering dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from nas_engine.architectures.cost import ArchitectureCost, compute_cost
from nas_engine.architectures.hashing import architecture_hash, short_hash
from nas_engine.architectures.shapes import ShapeTrace, infer_shapes
from nas_engine.architectures.spec import ArchitectureSpec


def _format_count(value: int) -> str:
    """Render a large integer with thousands separators."""
    return f"{value:,}"


def _format_millions(value: int) -> str:
    """Render a count in millions with two decimal places."""
    return f"{value / 1e6:.2f}M"


@dataclass(frozen=True)
class ArchitectureSummary:
    """A rendered description of an architecture.

    Attributes:
        architecture_hash: Full canonical hash.
        spec: The architecture that was summarised.
        cost: Analytic cost.
        trace: Shape trace.
    """

    architecture_hash: str
    spec: ArchitectureSpec
    cost: ArchitectureCost
    trace: ShapeTrace

    def compact(self) -> str:
        """Return a single-line description suitable for log lines and tables.

        Example output::

            a1b2c3d4 | 3 stages | 7 blocks | 0.42M params | 12.3M MACs | stride 8
        """
        return (
            f"{short_hash(self.architecture_hash)} | "
            f"{self.spec.num_stages} stages | "
            f"{self.spec.total_blocks} blocks | "
            f"{_format_millions(self.cost.trainable_parameters)} params | "
            f"{_format_millions(self.cost.multiply_accumulates)} MACs | "
            f"stride {self.spec.total_stride}"
        )

    def to_text(self) -> str:
        """Return a multi-line description with a per-layer shape table."""
        header = [
            f"Architecture {self.architecture_hash}",
            f"  input           : {self.trace.input_shape} ({self.spec.num_classes} classes)",
            f"  stages          : {self.spec.num_stages}",
            f"  blocks          : {self.spec.total_blocks}",
            f"  total stride    : {self.spec.total_stride} "
            f"(final feature map {self.trace.features_shape})",
            f"  trainable params: {_format_count(self.cost.trainable_parameters)}",
            f"  buffers         : {_format_count(self.cost.non_trainable_parameters)}",
            f"  MACs (per image): {_format_count(self.cost.multiply_accumulates)}",
            f"  float32 size    : {self.cost.parameter_bytes / 1024:.1f} KiB",
            "",
            f"  {'layer':<24} {'kind':<16} {'input':>14} {'output':>14}",
            f"  {'-' * 24} {'-' * 16} {'-' * 14} {'-' * 14}",
        ]
        rows = [
            f"  {name:<24} {kind:<16} {shape_in:>14} {shape_out:>14}"
            for name, kind, shape_in, shape_out in self.trace.to_rows()
        ]
        blocks = ["", "  block detail:"]
        for stage_index, stage in enumerate(self.spec.stages):
            descriptions = ", ".join(block.describe() for block in stage.blocks)
            blocks.append(f"    stage {stage_index}: {descriptions}")
        head = self.spec.head
        blocks.append(
            f"    head: global_{head.pooling.value}"
            f"{f', hidden {head.hidden_units}' if head.hidden_units else ''}"
            f", dropout {head.dropout:g}"
        )
        return "\n".join([*header, *rows, *blocks])

    def to_markdown(self) -> str:
        """Return a Markdown fragment with a shape table, for inclusion in reports."""
        lines = [
            f"**Architecture `{self.architecture_hash}`**",
            "",
            f"- Input: `{self.trace.input_shape}` → {self.spec.num_classes} classes",
            f"- Stages: {self.spec.num_stages}, blocks: {self.spec.total_blocks}, "
            f"total stride: {self.spec.total_stride}",
            f"- Trainable parameters: {_format_count(self.cost.trainable_parameters)}",
            f"- MACs per image: {_format_count(self.cost.multiply_accumulates)}",
            "",
            "| Layer | Kind | Input | Output |",
            "| --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| `{name}` | {kind} | `{shape_in}` | `{shape_out}` |"
            for name, kind, shape_in, shape_out in self.trace.to_rows()
        )
        return "\n".join(lines)


def summarise(spec: ArchitectureSpec) -> ArchitectureSummary:
    """Build an :class:`ArchitectureSummary` for a specification.

    Args:
        spec: Architecture to summarise.

    Returns:
        The summary.

    Raises:
        ShapeInferenceError: If the architecture is structurally invalid.
    """
    trace = infer_shapes(spec)
    return ArchitectureSummary(
        architecture_hash=architecture_hash(spec),
        spec=spec,
        cost=compute_cost(spec, trace),
        trace=trace,
    )


__all__ = ["ArchitectureSummary", "summarise"]
