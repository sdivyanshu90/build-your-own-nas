"""Seeded sampling of architectures from a search space.

Sampling strategy
-----------------
Candidates are drawn **hierarchically**, not by picking a random point in a flat
product space:

1. Choose the stem.
2. Choose the number of stages.
3. Choose each stage's width, honouring the monotonic-width rule when enabled.
4. Choose each stage's depth and its first-block stride.
5. Choose each block, restricted to the operations that are legal *at that position*.
6. Choose the head.
7. Repair, then validate. Retry on failure.

Restricting choices per position (step 5) rather than sampling freely and rejecting is
what keeps the acceptance rate high. A channel-preserving operation cannot sit where
the width changes, so it is simply not offered there.

Rejection sampling
------------------
Constraints (parameter ceilings, MAC ceilings) are still enforced by rejection, because
they depend on the whole architecture and cannot be decided one field at a time. The
sampler records why draws were rejected and exposes the rate, so a misconfigured
constraint surfaces as a diagnosable statistic rather than as a mysterious stall.

Determinism
-----------
The sampler owns a private :class:`random.Random`. It never touches the global RNG, so
its output depends only on its seed and on how many times it has been called — not on
what any other component did. Its full generator state is checkpointable, which is what
lets a resumed search continue the same stream rather than replaying it.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nas_engine.architectures.spec import (
    ArchitectureSpec,
    BlockSpec,
    HeadSpec,
    StageSpec,
    StemSpec,
)
from nas_engine.architectures.types import OperationType
from nas_engine.exceptions import SearchSpaceError
from nas_engine.search_space.repair import repair_architecture
from nas_engine.search_space.space import SearchSpace
from nas_engine.search_space.validation import ValidationReport, check_architecture
from nas_engine.utilities.seeding import rng_state_from_json, rng_state_to_json

#: Default number of draws attempted before the sampler gives up on one request.
DEFAULT_MAX_ATTEMPTS: int = 200

#: Version of the sampler's checkpoint payload.
SAMPLER_STATE_VERSION: int = 1


@dataclass
class SamplerStatistics:
    """Counters describing sampler behaviour.

    Attributes:
        attempts: Total draws made, including rejected ones.
        accepted: Draws that passed validation.
        rejected: Draws that failed validation.
        duplicates: Draws discarded because their hash was already seen.
        rejection_reasons: Count of rejections keyed by issue category and location.
    """

    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draws that produced a usable candidate."""
        return self.accepted / self.attempts if self.attempts else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "acceptance_rate": self.acceptance_rate,
            "rejection_reasons": dict(self.rejection_reasons),
        }


class ArchitectureSampler:
    """Draws valid architectures from a search space using a private generator.

    Args:
        space: The space to sample from.
        seed: Seed for the private generator.
        max_attempts: Draws attempted per request before raising.
        residual_probability: Probability of enabling a residual connection where one is
            legal. ``0.5`` gives an unbiased coin flip; lower values bias the search
            towards plain stacks.

    Raises:
        ValueError: If ``max_attempts`` is not positive or ``residual_probability`` lies
            outside ``[0, 1]``.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        seed: int,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        residual_probability: float = 0.5,
    ) -> None:
        if max_attempts < 1:
            msg = f"max_attempts must be >= 1, received {max_attempts}"
            raise ValueError(msg)
        if not 0.0 <= residual_probability <= 1.0:
            msg = f"residual_probability must lie in [0.0, 1.0], received {residual_probability}"
            raise ValueError(msg)
        space.require_non_empty()
        self._space = space
        self._seed = seed
        self._max_attempts = max_attempts
        self._residual_probability = residual_probability
        # `random.Random` is a Mersenne Twister: fast, reproducible, and completely
        # unsuitable for cryptography — which is exactly right for search sampling.
        self._rng = random.Random(seed)  # noqa: S311 - reproducibility, not security
        self._statistics = SamplerStatistics()

    # -- properties ----------------------------------------------------------------
    @property
    def space(self) -> SearchSpace:
        """The space being sampled."""
        return self._space

    @property
    def statistics(self) -> SamplerStatistics:
        """Live sampler statistics."""
        return self._statistics

    # -- sampling ------------------------------------------------------------------
    def _draw_stem(self) -> StemSpec:
        """Draw a stem configuration."""
        stem = self._space.stem
        return StemSpec(
            out_channels=self._rng.choice(stem.out_channels),
            kernel_size=self._rng.choice(stem.kernel_sizes),
            stride=self._rng.choice(stem.strides),
            normalization=self._rng.choice(stem.normalizations),
            activation=self._rng.choice(stem.activations),
        )

    def _draw_head(self) -> HeadSpec:
        """Draw a classifier-head configuration."""
        head = self._space.head
        return HeadSpec(
            pooling=self._rng.choice(head.poolings),
            hidden_units=self._rng.choice(head.hidden_units),
            dropout=self._rng.choice(head.dropouts),
            activation=self._rng.choice(head.activations),
        )

    def _draw_widths(self, num_stages: int) -> list[int]:
        """Draw one width per stage, honouring the monotonic-width rule.

        Args:
            num_stages: Number of stages to draw widths for.

        Returns:
            A list of widths.

        Raises:
            SearchSpaceError: If the monotonic rule leaves no admissible width, which
                can only happen if ``stage_channels`` is empty.
        """
        widths: list[int] = []
        for _ in range(num_stages):
            if self._space.monotonic_widths and widths:
                allowed = [width for width in self._space.stage_channels if width >= widths[-1]]
            else:
                allowed = list(self._space.stage_channels)
            if not allowed:
                msg = (
                    "no stage width satisfies the monotonic-width rule; this indicates a "
                    "corrupt search space"
                )
                raise SearchSpaceError(msg, details={"previous_width": widths[-1]})
            widths.append(self._rng.choice(allowed))
        return widths

    def _draw_block(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        allow_channel_preserving: bool,
    ) -> BlockSpec:
        """Draw one block, restricted to operations legal at this position.

        Args:
            in_channels: Channels entering the block.
            out_channels: Channels the block must produce.
            stride: Stride the block must apply.
            allow_channel_preserving: Whether identity and pooling are admissible, i.e.
                whether the width is unchanged at this position.

        Returns:
            A canonical :class:`BlockSpec`.

        Raises:
            SearchSpaceError: If no operation is admissible at this position.
        """
        choices = self._space.block
        operations = [
            operation
            for operation in choices.operations
            if (allow_channel_preserving or operation.can_change_channels)
            and not (stride > 1 and operation is OperationType.IDENTITY)
        ]
        if not operations:
            msg = (
                "no operation in the search space can be placed at this position "
                f"(in_channels={in_channels}, out_channels={out_channels}, stride={stride}); "
                "add a convolutional operation to block.operations"
            )
            raise SearchSpaceError(
                msg,
                details={
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "stride": stride,
                },
            )
        operation = self._rng.choice(operations)

        # Draw every conditional field unconditionally so the number of random draws is
        # independent of which operation was selected. Keeping the draw count fixed means
        # a mutation that changes only the operation does not shift every later draw in
        # the stream, which makes seeded runs far easier to reason about. Inactive values
        # are erased by canonicalisation in `BlockSpec`.
        kernel_size = self._rng.choice(choices.kernel_sizes)
        expansion_ratio = self._rng.choice(choices.expansion_ratios)
        normalization = self._rng.choice(choices.normalizations)
        activation = self._rng.choice(choices.activations)
        residual_draw = self._rng.random()

        shape_preserved = stride == 1 and in_channels == out_channels
        use_residual = (
            choices.allow_residual
            and shape_preserved
            and operation is not OperationType.IDENTITY
            and residual_draw < self._residual_probability
        )
        return BlockSpec(
            operation=operation,
            kernel_size=kernel_size,
            expansion_ratio=expansion_ratio,
            out_channels=out_channels,
            stride=stride,
            use_residual=use_residual,
            normalization=normalization,
            activation=activation,
        )

    def _draw_architecture(self) -> ArchitectureSpec:
        """Draw one raw architecture, before repair and validation."""
        space = self._space
        stem = self._draw_stem()
        num_stages = self._rng.choice(space.num_stages)
        widths = self._draw_widths(num_stages)

        stages: list[StageSpec] = []
        current_channels = stem.out_channels
        for width in widths:
            depth = self._rng.choice(space.blocks_per_stage)
            stride = self._rng.choice(space.stage_strides)
            blocks: list[BlockSpec] = []
            for block_index in range(depth):
                block_stride = stride if block_index == 0 else 1
                block = self._draw_block(
                    in_channels=current_channels,
                    out_channels=width,
                    stride=block_stride,
                    allow_channel_preserving=current_channels == width,
                )
                blocks.append(block)
                current_channels = block.out_channels
            stages.append(StageSpec(blocks=tuple(blocks)))

        return ArchitectureSpec(
            input_channels=space.input_channels,
            input_size=space.input_size,
            num_classes=space.num_classes,
            stem=stem,
            stages=tuple(stages),
            head=self._draw_head(),
        )

    def try_sample(self) -> tuple[ArchitectureSpec, ValidationReport] | None:
        """Attempt one draw, returning ``None`` if it is rejected.

        Returns:
            A ``(spec, report)`` pair on success, or ``None`` on rejection.
        """
        self._statistics.attempts += 1
        raw = self._draw_architecture()
        repaired, _ = repair_architecture(raw)
        report = check_architecture(repaired, self._space)
        if report.is_valid:
            self._statistics.accepted += 1
            return repaired, report
        self._statistics.rejected += 1
        for issue in report.issues:
            self._statistics.rejection_reasons[f"{issue.category}:{issue.location}"] += 1
        return None

    def sample(self) -> ArchitectureSpec:
        """Draw one valid architecture.

        Returns:
            A validated architecture.

        Raises:
            SearchSpaceError: If ``max_attempts`` draws all failed validation. The error
                lists the most common rejection reasons so the misconfigured constraint
                is immediately visible.
        """
        for _ in range(self._max_attempts):
            result = self.try_sample()
            if result is not None:
                return result[0]
        top_reasons = self._statistics.rejection_reasons.most_common(5)
        msg = (
            f"failed to sample a valid architecture in {self._max_attempts} attempts. "
            f"Most common rejection reasons: {top_reasons}. Relax the search-space "
            "constraints (max_parameters, max_multiply_accumulates, min_final_resolution) "
            "or widen the choice sets."
        )
        raise SearchSpaceError(
            msg,
            details={
                "max_attempts": self._max_attempts,
                "rejection_reasons": dict(self._statistics.rejection_reasons),
            },
        )

    def sample_unique(
        self, seen_hashes: set[str], *, max_attempts: int | None = None
    ) -> ArchitectureSpec | None:
        """Draw an architecture whose hash is not already in ``seen_hashes``.

        Duplicate avoidance is best-effort by construction: once a space is largely
        exhausted, no number of draws will find a novel member, and the caller must
        decide whether to stop. Returning ``None`` rather than raising lets the search
        strategy make that decision.

        Args:
            seen_hashes: Hashes already proposed or evaluated.
            max_attempts: Override for the configured attempt budget.

        Returns:
            A novel valid architecture, or ``None`` if none was found in budget.
        """
        from nas_engine.architectures.hashing import architecture_hash

        budget = max_attempts if max_attempts is not None else self._max_attempts
        for _ in range(budget):
            result = self.try_sample()
            if result is None:
                continue
            spec = result[0]
            if architecture_hash(spec) in seen_hashes:
                self._statistics.duplicates += 1
                continue
            return spec
        return None

    # -- checkpointing -------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the sampler's state.

        Returns:
            The checkpoint payload, including the exact generator position.
        """
        return {
            "version": SAMPLER_STATE_VERSION,
            "seed": self._seed,
            "rng": rng_state_to_json(self._rng),
            "statistics": self._statistics.to_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore sampler state from :meth:`state_dict` output.

        Args:
            payload: Previously captured state.

        Raises:
            SearchSpaceError: If the payload version is unsupported or malformed.
        """
        version = payload.get("version")
        if version != SAMPLER_STATE_VERSION:
            msg = (
                f"sampler state version {version} is not supported by this build "
                f"(expected {SAMPLER_STATE_VERSION})"
            )
            raise SearchSpaceError(msg, details={"version": version})
        try:
            self._rng = rng_state_from_json(payload["rng"])
        except (KeyError, ValueError) as exc:
            msg = f"sampler state could not be restored: {exc}"
            raise SearchSpaceError(msg, details={"error": str(exc)}) from exc
        stats = payload.get("statistics", {})
        self._statistics = SamplerStatistics(
            attempts=int(stats.get("attempts", 0)),
            accepted=int(stats.get("accepted", 0)),
            rejected=int(stats.get("rejected", 0)),
            duplicates=int(stats.get("duplicates", 0)),
            rejection_reasons=Counter(stats.get("rejection_reasons", {})),
        )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "SAMPLER_STATE_VERSION",
    "ArchitectureSampler",
    "SamplerStatistics",
]
