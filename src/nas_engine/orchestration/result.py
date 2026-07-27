"""The object a completed search returns.

:class:`SearchResult` is part of the public API, so it deliberately contains no database
handles, no ORM rows, and no PyTorch objects. It is plain data that can be printed,
serialised, exported, or archived, and it stays valid after the engine and its database
connection are gone.

To go from a result back to a usable model, take ``best.architecture_hash`` (or
``best.candidate_id``) and ask the repository for the specification and weights. Keeping
that step explicit is what allows a result to be inspected long after the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nas_engine.objectives.ranking import RankedCandidate
from nas_engine.orchestration.checkpoint import EngineState


class StopReason(str, Enum):
    """Why a search stopped.

    Members:
        BUDGET_EXHAUSTED: The evaluation budget was fully spent.
        STRATEGY_FINISHED: The strategy reported it had nothing left to propose.
        TIME_LIMIT: The wall-clock limit was reached.
        SPACE_EXHAUSTED: No novel valid candidate could be produced.
        INTERRUPTED: The operator interrupted the run; it can be resumed.
        ERROR: An engine-level error stopped the run.
    """

    BUDGET_EXHAUSTED = "budget_exhausted"
    STRATEGY_FINISHED = "strategy_finished"
    TIME_LIMIT = "time_limit"
    SPACE_EXHAUSTED = "space_exhausted"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    def describe(self) -> str:
        """Return a human-readable explanation."""
        return {
            StopReason.BUDGET_EXHAUSTED: "the evaluation budget was fully spent",
            StopReason.STRATEGY_FINISHED: "the search strategy completed its plan",
            StopReason.TIME_LIMIT: "the wall-clock time limit was reached",
            StopReason.SPACE_EXHAUSTED: ("the search space ran out of novel valid architectures"),
            StopReason.INTERRUPTED: "the run was interrupted and can be resumed",
            StopReason.ERROR: "an engine-level error stopped the run",
        }[self]


@dataclass(frozen=True)
class SearchResult:
    """The outcome of a search run.

    Attributes:
        search_id: Identifier of the run, used to resume or inspect it later.
        status: Final search status.
        stop_reason: Why the run stopped.
        best: The top-ranked candidate, or ``None`` when nothing completed.
        pareto_front: Feasible, non-dominated candidates.
        ranked: Every completed candidate in rank order.
        engine_state: Engine counters.
        strategy_statistics: Strategy-specific counters.
        duration_seconds: Wall-clock duration of this run segment.
        total_evaluations: Successful evaluations across the whole search.
        warnings: Non-fatal issues worth surfacing to the operator.
        resumed: Whether this segment resumed an existing search.
    """

    search_id: str
    status: str
    stop_reason: StopReason
    best: RankedCandidate | None
    pareto_front: tuple[RankedCandidate, ...]
    ranked: tuple[RankedCandidate, ...]
    engine_state: EngineState
    strategy_statistics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    total_evaluations: int = 0
    warnings: tuple[str, ...] = ()
    resumed: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether the run finished normally with at least one completed candidate."""
        return self.stop_reason not in {StopReason.ERROR} and self.best is not None

    @property
    def best_accuracy(self) -> float | None:
        """Validation accuracy of the best candidate, when available."""
        if self.best is None:
            return None
        return self.best.metrics.get("validation_accuracy")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "search_id": self.search_id,
            "status": self.status,
            "stop_reason": self.stop_reason.value,
            "stop_reason_description": self.stop_reason.describe(),
            "best": self.best.to_dict() if self.best else None,
            "pareto_front": [candidate.to_dict() for candidate in self.pareto_front],
            "ranked": [candidate.to_dict() for candidate in self.ranked],
            "engine_state": self.engine_state.to_dict(),
            "strategy_statistics": dict(self.strategy_statistics),
            "duration_seconds": self.duration_seconds,
            "total_evaluations": self.total_evaluations,
            "warnings": list(self.warnings),
            "resumed": self.resumed,
        }

    def summary(self) -> str:
        """Return a short human-readable summary for CLI output."""
        lines = [
            f"Search {self.search_id} finished: {self.stop_reason.describe()}",
            f"  status            : {self.status}",
            f"  duration          : {self.duration_seconds:.1f}s"
            f"{' (resumed)' if self.resumed else ''}",
            f"  proposed          : {self.engine_state.proposed}",
            f"  evaluated         : {self.engine_state.completed}",
            f"  duplicates        : {self.engine_state.duplicates}",
            f"  invalid           : {self.engine_state.invalid}",
            f"  pruned            : {self.engine_state.pruned}",
            f"  failed            : {self.engine_state.failed}",
            f"  Pareto front size : {len(self.pareto_front)}",
        ]
        if self.best is not None:
            accuracy = self.best.metrics.get("validation_accuracy")
            parameters = self.best.metrics.get("trainable_parameters")
            accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "n/a"
            parameter_text = f"{int(parameters):,}" if parameters is not None else "n/a"
            score_text = f"{self.best.score:.4f}" if self.best.score is not None else "n/a"
            lines.extend(
                [
                    "",
                    f"  best candidate    : {self.best.architecture_hash}",
                    f"    validation acc  : {accuracy_text}",
                    f"    parameters      : {parameter_text}",
                    f"    score           : {score_text}",
                ]
            )
        else:
            lines.append("  best candidate    : none (no candidate completed successfully)")
        if self.warnings:
            lines.append("")
            lines.append("  warnings:")
            lines.extend(f"    - {warning}" for warning in self.warnings)
        return "\n".join(lines)


__all__ = ["SearchResult", "StopReason"]
