"""Architecture validation: schema, semantics, membership, and constraints.

Validation happens in four distinct layers, and keeping them distinct matters because
each answers a different question and each has a different remedy.

===============  ======================================  ==============================
Layer            Question                                Where it runs
===============  ======================================  ==============================
Schema           Are the field types and ranges valid?   Pydantic, at construction
Semantic         Do the tensors actually line up?        :func:`infer_shapes`
Membership       Is every choice allowed by this space?  :func:`check_membership`
Constraint       Is it within the resource budget?       :func:`check_constraints`
===============  ======================================  ==============================

A schema failure means the document is malformed. A semantic failure means the network
cannot be built. A membership failure means the candidate came from a different space
— which is exactly what happens when a resumed search is pointed at an edited
configuration, so the error names the offending field and both values. A constraint
failure means the network is buildable but too expensive; those candidates are recorded
as ``PRUNED`` rather than ``FAILED``, because nothing went wrong.

All findings are collected before raising, so a user fixing an imported architecture
sees every problem at once instead of one per attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nas_engine.architectures.cost import ArchitectureCost, compute_cost
from nas_engine.architectures.shapes import ShapeTrace, infer_shapes
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.architectures.types import OperationType
from nas_engine.exceptions import (
    ArchitectureValidationError,
    ConstraintViolationError,
    ShapeInferenceError,
)
from nas_engine.search_space.space import SearchSpace


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found during validation.

    Attributes:
        category: ``"semantic"``, ``"membership"``, or ``"constraint"``.
        location: Dotted path to the offending element.
        message: Actionable description of the problem.
        received: The value that was seen.
        expected: A description of what would have been acceptable.
    """

    category: str
    location: str
    message: str
    received: Any = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "category": self.category,
            "location": self.location,
            "message": self.message,
            "received": repr(self.received) if self.received is not None else None,
            "expected": repr(self.expected) if self.expected is not None else None,
        }

    def __str__(self) -> str:
        """Render as ``location: message``."""
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating one architecture.

    Attributes:
        issues: Every problem found, in discovery order.
        trace: Shape trace, present when semantic validation succeeded.
        cost: Analytic cost, present when semantic validation succeeded.
    """

    issues: tuple[ValidationIssue, ...] = ()
    trace: ShapeTrace | None = None
    cost: ArchitectureCost | None = None

    @property
    def is_valid(self) -> bool:
        """Whether the architecture passed every check."""
        return not self.issues

    def issues_of(self, category: str) -> tuple[ValidationIssue, ...]:
        """Return only the issues in one category.

        Args:
            category: ``"semantic"``, ``"membership"``, or ``"constraint"``.

        Returns:
            Matching issues.
        """
        return tuple(issue for issue in self.issues if issue.category == category)

    @property
    def only_constraint_violations(self) -> bool:
        """Whether the architecture is buildable but exceeds a resource budget.

        The engine uses this to distinguish a *pruned* candidate (too expensive, nothing
        broken) from an *invalid* one (structurally impossible).
        """
        return bool(self.issues) and all(issue.category == "constraint" for issue in self.issues)

    def summary(self) -> str:
        """Return a single-line summary listing every issue."""
        if self.is_valid:
            return "architecture is valid"
        return "; ".join(str(issue) for issue in self.issues)

    def raise_if_invalid(self) -> None:
        """Raise the appropriate error when issues are present.

        Raises:
            ConstraintViolationError: When every issue is a resource-budget violation.
            ArchitectureValidationError: When any structural or membership issue exists.
        """
        if self.is_valid:
            return
        details: dict[str, Any] = {"issues": [issue.to_dict() for issue in self.issues]}
        if self.only_constraint_violations:
            raise ConstraintViolationError(
                f"architecture violates search-space constraints: {self.summary()}",
                details=details,
            )
        raise ArchitectureValidationError(
            f"architecture is invalid: {self.summary()}", details=details
        )


@dataclass
class _IssueCollector:
    """Mutable accumulator used while walking an architecture."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        category: str,
        location: str,
        message: str,
        *,
        received: Any = None,
        expected: Any = None,
    ) -> None:
        """Record one issue."""
        self.issues.append(
            ValidationIssue(
                category=category,
                location=location,
                message=message,
                received=received,
                expected=expected,
            )
        )


def check_membership(  # noqa: PLR0912 - one branch per search-space dimension
    spec: ArchitectureSpec, space: SearchSpace
) -> list[ValidationIssue]:
    """Verify every choice in ``spec`` is permitted by ``space``.

    Args:
        spec: Architecture to check.
        space: Space the architecture must belong to.

    Returns:
        A list of membership issues; empty when the architecture is a member.
    """
    collector = _IssueCollector()

    if spec.input_channels != space.input_channels:
        collector.add(
            "membership",
            "input_channels",
            "input channel count does not match the search space",
            received=spec.input_channels,
            expected=space.input_channels,
        )
    if spec.input_size != space.input_size:
        collector.add(
            "membership",
            "input_size",
            "input size does not match the search space",
            received=spec.input_size,
            expected=space.input_size,
        )
    if spec.num_classes != space.num_classes:
        collector.add(
            "membership",
            "num_classes",
            "class count does not match the search space",
            received=spec.num_classes,
            expected=space.num_classes,
        )

    if spec.num_stages not in space.num_stages:
        collector.add(
            "membership",
            "stages",
            "stage count is not offered by the search space",
            received=spec.num_stages,
            expected=list(space.num_stages),
        )

    stem = spec.stem
    for attribute, allowed, location in (
        (stem.out_channels, space.stem.out_channels, "stem.out_channels"),
        (stem.kernel_size, space.stem.kernel_sizes, "stem.kernel_size"),
        (stem.stride, space.stem.strides, "stem.stride"),
        (stem.normalization, space.stem.normalizations, "stem.normalization"),
        (stem.activation, space.stem.activations, "stem.activation"),
    ):
        if attribute not in allowed:
            collector.add(
                "membership",
                location,
                "value is not offered by the search space",
                received=attribute,
                expected=list(allowed),
            )

    # The monotonic-width rule constrains stages relative to each other. The stem is a
    # separate entity and is deliberately excluded: a network may legitimately narrow
    # from a 32-channel stem into a 16-channel first stage.
    previous_width: int | None = None
    for stage_index, stage in enumerate(spec.stages):
        if len(stage.blocks) not in space.blocks_per_stage:
            collector.add(
                "membership",
                f"stages.{stage_index}.blocks",
                "block count is not offered by the search space",
                received=len(stage.blocks),
                expected=list(space.blocks_per_stage),
            )
        width = stage.out_channels
        if width not in space.stage_channels:
            collector.add(
                "membership",
                f"stages.{stage_index}.width",
                "stage width is not offered by the search space",
                received=width,
                expected=list(space.stage_channels),
            )
        if space.monotonic_widths and previous_width is not None and width < previous_width:
            collector.add(
                "membership",
                f"stages.{stage_index}.width",
                "search space requires non-decreasing stage widths",
                received=width,
                expected=f">= {previous_width}",
            )
        previous_width = width

        for block_index, block in enumerate(stage.blocks):
            location = f"stages.{stage_index}.blocks.{block_index}"
            if block.operation not in space.block.operations:
                collector.add(
                    "membership",
                    f"{location}.operation",
                    "operation is not offered by the search space",
                    received=block.operation.value,
                    expected=[op.value for op in space.block.operations],
                )
            if block.operation.uses_kernel_size and (
                block.kernel_size not in space.block.kernel_sizes
            ):
                collector.add(
                    "membership",
                    f"{location}.kernel_size",
                    "kernel size is not offered by the search space",
                    received=block.kernel_size,
                    expected=list(space.block.kernel_sizes),
                )
            if block.operation.uses_expansion_ratio and (
                block.expansion_ratio not in space.block.expansion_ratios
            ):
                collector.add(
                    "membership",
                    f"{location}.expansion_ratio",
                    "expansion ratio is not offered by the search space",
                    received=block.expansion_ratio,
                    expected=list(space.block.expansion_ratios),
                )
            if block.operation.is_parametric:
                if block.normalization not in space.block.normalizations:
                    collector.add(
                        "membership",
                        f"{location}.normalization",
                        "normalisation is not offered by the search space",
                        received=block.normalization.value,
                        expected=[n.value for n in space.block.normalizations],
                    )
                if block.activation not in space.block.activations:
                    collector.add(
                        "membership",
                        f"{location}.activation",
                        "activation is not offered by the search space",
                        received=block.activation.value,
                        expected=[a.value for a in space.block.activations],
                    )
            if block.use_residual and not space.block.allow_residual:
                collector.add(
                    "membership",
                    f"{location}.use_residual",
                    "residual connections are disabled in this search space",
                    received=True,
                    expected=False,
                )
            allowed_strides = space.stage_strides if block_index == 0 else (1,)
            if block.operation is not OperationType.IDENTITY and (
                block.stride not in allowed_strides
            ):
                collector.add(
                    "membership",
                    f"{location}.stride",
                    "stride is not offered at this position; only the first block of a "
                    "stage may downsample",
                    received=block.stride,
                    expected=list(allowed_strides),
                )

    head = spec.head
    for attribute, allowed, location in (
        (head.pooling, space.head.poolings, "head.pooling"),
        (head.hidden_units, space.head.hidden_units, "head.hidden_units"),
        (head.dropout, space.head.dropouts, "head.dropout"),
    ):
        if attribute not in allowed:
            collector.add(
                "membership",
                location,
                "value is not offered by the search space",
                received=attribute,
                expected=list(allowed),
            )
    if head.hidden_units > 0 and head.activation not in space.head.activations:
        collector.add(
            "membership",
            "head.activation",
            "value is not offered by the search space",
            received=head.activation.value,
            expected=[a.value for a in space.head.activations],
        )

    return collector.issues


def check_constraints(
    cost: ArchitectureCost,
    trace: ShapeTrace,
    spec: ArchitectureSpec,
    space: SearchSpace,
) -> list[ValidationIssue]:
    """Verify an architecture fits within the space's resource budget.

    Args:
        cost: Analytic cost of the architecture.
        trace: Shape trace of the architecture.
        spec: The architecture.
        space: Space supplying the constraints.

    Returns:
        A list of constraint issues; empty when the architecture is feasible.
    """
    collector = _IssueCollector()
    limits = space.constraints

    if limits.max_parameters is not None and cost.trainable_parameters > limits.max_parameters:
        collector.add(
            "constraint",
            "parameters",
            f"trainable parameter count {cost.trainable_parameters:,} exceeds the limit "
            f"of {limits.max_parameters:,}; reduce stage_channels or blocks_per_stage",
            received=cost.trainable_parameters,
            expected=f"<= {limits.max_parameters}",
        )
    if limits.min_parameters is not None and cost.trainable_parameters < limits.min_parameters:
        collector.add(
            "constraint",
            "parameters",
            f"trainable parameter count {cost.trainable_parameters:,} is below the "
            f"minimum of {limits.min_parameters:,}; this usually means the candidate is "
            "mostly pooling and identity operations",
            received=cost.trainable_parameters,
            expected=f">= {limits.min_parameters}",
        )
    if (
        limits.max_multiply_accumulates is not None
        and cost.multiply_accumulates > limits.max_multiply_accumulates
    ):
        collector.add(
            "constraint",
            "multiply_accumulates",
            f"estimated MACs {cost.multiply_accumulates:,} exceed the limit of "
            f"{limits.max_multiply_accumulates:,}",
            received=cost.multiply_accumulates,
            expected=f"<= {limits.max_multiply_accumulates}",
        )
    if limits.max_total_stride is not None and spec.total_stride > limits.max_total_stride:
        collector.add(
            "constraint",
            "total_stride",
            f"total stride {spec.total_stride} exceeds the limit of {limits.max_total_stride}",
            received=spec.total_stride,
            expected=f"<= {limits.max_total_stride}",
        )
    final_resolution = min(trace.features_shape.height, trace.features_shape.width)
    if final_resolution < limits.min_final_resolution:
        collector.add(
            "constraint",
            "final_resolution",
            f"final feature map is {trace.features_shape} but the space requires at "
            f"least {limits.min_final_resolution} spatial pixels; remove a stride-2 block",
            received=final_resolution,
            expected=f">= {limits.min_final_resolution}",
        )
    if limits.max_depth is not None and spec.total_blocks > limits.max_depth:
        collector.add(
            "constraint",
            "depth",
            f"block count {spec.total_blocks} exceeds the limit of {limits.max_depth}",
            received=spec.total_blocks,
            expected=f"<= {limits.max_depth}",
        )
    return collector.issues


def check_architecture(
    spec: ArchitectureSpec,
    space: SearchSpace,
    *,
    check_space_membership: bool = True,
) -> ValidationReport:
    """Validate an architecture and return a report without raising.

    Semantic validation runs first: if shapes cannot be reconciled, cost and constraint
    checks are meaningless and are skipped.

    Args:
        spec: Architecture to validate.
        space: Space the architecture should belong to.
        check_space_membership: Whether to enforce that every choice is offered by the
            space. Disable when validating an architecture that was deliberately
            hand-written or imported from another space and only needs to be buildable.

    Returns:
        A :class:`ValidationReport`.
    """
    issues: list[ValidationIssue] = []
    trace: ShapeTrace | None = None
    cost: ArchitectureCost | None = None

    try:
        trace = infer_shapes(spec)
    except ShapeInferenceError as exc:
        location = str(exc.details.get("location", "architecture"))
        issues.append(
            ValidationIssue(
                category="semantic",
                location=location,
                message=exc.message,
                received=exc.details.get("received"),
            )
        )

    if trace is not None:
        cost = compute_cost(spec, trace)
        if check_space_membership:
            issues.extend(check_membership(spec, space))
        issues.extend(check_constraints(cost, trace, spec, space))
    elif check_space_membership:
        issues.extend(check_membership(spec, space))

    return ValidationReport(issues=tuple(issues), trace=trace, cost=cost)


def validate_architecture(
    spec: ArchitectureSpec,
    space: SearchSpace,
    *,
    check_space_membership: bool = True,
) -> ValidationReport:
    """Validate an architecture, raising on the first report containing issues.

    Args:
        spec: Architecture to validate.
        space: Space the architecture should belong to.
        check_space_membership: Whether to enforce space membership.

    Returns:
        The (valid) report.

    Raises:
        ConstraintViolationError: If only resource constraints were violated.
        ArchitectureValidationError: If any structural or membership issue was found.
    """
    report = check_architecture(spec, space, check_space_membership=check_space_membership)
    report.raise_if_invalid()
    return report


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "check_architecture",
    "check_constraints",
    "check_membership",
    "validate_architecture",
]
