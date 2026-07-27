"""Random search.

Random search is the baseline every NAS method must beat, and it is a genuinely strong
one. Bergstra and Bengio's result for hyperparameter optimisation applies directly: when
only a few dimensions matter, random sampling covers those dimensions far better than a
grid, and it degrades gracefully as dimensionality grows. In NAS specifically, published
comparisons repeatedly find that random search with an equal *total compute* budget lands
close to much more elaborate methods — which usually says more about the search space than
about the algorithms.

Because it is the baseline, it must be implemented *correctly*, not carelessly:

* **Seeded and reproducible.** The same seed produces the same proposal sequence, and the
  full generator state is checkpointed so a resumed search continues rather than replays.
* **Duplicate-avoiding.** Re-training an architecture already evaluated yields no
  information and costs a full budget. Every hash seen — proposed, evaluated, or failed —
  is remembered.
* **Honest about exhaustion.** In a small space, novel candidates run out. The strategy
  reports that as completion instead of looping forever.
* **Constraint-aware.** Sampling delegates to
  :class:`~nas_engine.search_space.sampler.ArchitectureSampler`, so search-space
  constraints are enforced by construction rather than by the engine rejecting proposals.
"""

from __future__ import annotations

from typing import Any, ClassVar

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import CheckpointError, CheckpointVersionError
from nas_engine.observability.logging import get_logger
from nas_engine.search.strategy import (
    Observation,
    Proposal,
    SearchStrategy,
    StrategyStatistics,
)
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace

_LOGGER = get_logger(__name__)

#: Version of this strategy's state payload.
RANDOM_SEARCH_STATE_VERSION: int = 1


class RandomSearch(SearchStrategy):
    """Uniform random sampling from the search space, without replacement.

    Args:
        space: Space to sample from.
        seed: Seed for the private sampler.
        max_evaluations: Total candidates to propose over the whole search.
        budget: Training budget applied to every candidate.
        sample_attempts: Draws attempted per proposal before giving up on novelty.
        max_consecutive_exhaustions: Consecutive failed novelty searches tolerated before
            declaring the space exhausted. More than one is required because sampling is
            probabilistic: a single unlucky run of duplicates is not proof of exhaustion.

    Raises:
        ValueError: If ``max_evaluations`` is not positive.
    """

    name: ClassVar[str] = "random_search"
    requires_synchronous_observations: ClassVar[bool] = False

    def __init__(
        self,
        space: SearchSpace,
        *,
        seed: int,
        max_evaluations: int,
        budget: TrainingBudget,
        sample_attempts: int = 200,
        max_consecutive_exhaustions: int = 3,
    ) -> None:
        if max_evaluations < 1:
            msg = f"max_evaluations must be >= 1, received {max_evaluations}"
            raise ValueError(msg)
        self._space = space
        self._seed = seed
        self._max_evaluations = max_evaluations
        self._budget = budget
        self._sample_attempts = sample_attempts
        self._max_consecutive_exhaustions = max_consecutive_exhaustions
        self._sampler = ArchitectureSampler(space, seed=seed, max_attempts=sample_attempts)
        self._seen: set[str] = set()
        self._proposed = 0
        self._observed = 0
        self._succeeded = 0
        self._failed = 0
        self._duplicates = 0
        self._consecutive_exhaustions = 0
        self._exhausted = False

    # -- strategy interface ---------------------------------------------------------
    def propose(self, count: int) -> list[Proposal]:
        """Draw up to ``count`` novel architectures.

        Args:
            count: Maximum proposals wanted.

        Returns:
            Novel proposals; fewer than ``count`` when the remaining evaluation budget or
            the space's novelty runs out.
        """
        remaining = self._max_evaluations - self._proposed
        wanted = max(0, min(count, remaining))
        proposals: list[Proposal] = []

        for _ in range(wanted):
            if self._exhausted:
                break
            spec = self._sampler.sample_unique(self._seen)
            if spec is None:
                self._consecutive_exhaustions += 1
                if self._consecutive_exhaustions >= self._max_consecutive_exhaustions:
                    self._exhausted = True
                    _LOGGER.info(
                        "random_search.exhausted",
                        proposed=self._proposed,
                        unique_seen=len(self._seen),
                        attempts_per_proposal=self._sample_attempts,
                    )
                break
            self._consecutive_exhaustions = 0
            self._seen.add(architecture_hash(spec))
            self._proposed += 1
            proposals.append(Proposal(spec=spec, budget=self._budget, origin="random"))
        return proposals

    def observe(self, observation: Observation) -> None:
        """Record an evaluation outcome.

        Random search does not adapt, but it still tracks hashes so that a resumed run
        does not re-propose anything already evaluated.

        Args:
            observation: The evaluation outcome.
        """
        self._observed += 1
        self._seen.add(observation.architecture_hash)
        if observation.succeeded:
            self._succeeded += 1
        else:
            self._failed += 1

    def is_finished(self) -> bool:
        """Report whether the evaluation budget is spent or the space is exhausted."""
        return self._exhausted or self._proposed >= self._max_evaluations

    def on_duplicate(self, architecture_hash: str) -> None:
        """Record a duplicate rejected by the engine.

        Args:
            architecture_hash: Hash of the duplicate architecture.
        """
        self._duplicates += 1
        self._seen.add(architecture_hash)

    def statistics(self) -> StrategyStatistics:
        """Return counters for reporting."""
        return StrategyStatistics(
            proposed=self._proposed,
            observed=self._observed,
            succeeded=self._succeeded,
            failed=self._failed,
            duplicates_avoided=self._duplicates,
            extra={
                "unique_architectures": len(self._seen),
                "max_evaluations": self._max_evaluations,
                "exhausted": self._exhausted,
                "sampler": self._sampler.statistics.to_dict(),
            },
        )

    def describe(self) -> str:
        """Return a human-readable configuration summary."""
        return (
            f"random_search: {self._max_evaluations} evaluations at "
            f"{self._budget.describe()} from space '{self._space.name}'"
        )

    # -- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the strategy state."""
        return {
            "version": RANDOM_SEARCH_STATE_VERSION,
            "strategy": self.name,
            "seed": self._seed,
            "max_evaluations": self._max_evaluations,
            "proposed": self._proposed,
            "observed": self._observed,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "duplicates": self._duplicates,
            "exhausted": self._exhausted,
            "consecutive_exhaustions": self._consecutive_exhaustions,
            "seen": sorted(self._seen),
            "sampler": self._sampler.state_dict(),
            "budget": self._budget.to_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore state captured by :meth:`state_dict`.

        Args:
            payload: Previously captured state.

        Raises:
            CheckpointVersionError: If the payload version is unsupported.
            CheckpointError: If the payload is malformed.
        """
        version = payload.get("version")
        if version != RANDOM_SEARCH_STATE_VERSION:
            msg = (
                f"random_search state version {version} is not supported by this build "
                f"(expected {RANDOM_SEARCH_STATE_VERSION})"
            )
            raise CheckpointVersionError(msg, details={"version": version})
        try:
            self._proposed = int(payload["proposed"])
            self._observed = int(payload.get("observed", 0))
            self._succeeded = int(payload.get("succeeded", 0))
            self._failed = int(payload.get("failed", 0))
            self._duplicates = int(payload.get("duplicates", 0))
            self._exhausted = bool(payload.get("exhausted", False))
            self._consecutive_exhaustions = int(payload.get("consecutive_exhaustions", 0))
            self._seen = set(payload.get("seen", []))
            self._sampler.load_state_dict(payload["sampler"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"random_search state payload is malformed: {exc}"
            raise CheckpointError(msg, details={"error": str(exc)}) from exc


__all__ = ["RANDOM_SEARCH_STATE_VERSION", "RandomSearch"]
