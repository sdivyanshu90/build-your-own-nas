r"""Successive halving: multi-fidelity resource allocation.

The idea
--------
Training every candidate to convergence is the accurate way to rank architectures and the
most wasteful. Most candidates are obviously bad after a fraction of the budget. Successive
halving exploits that: evaluate many candidates cheaply, keep the best fraction, and
re-evaluate the survivors with more resources. Repeat.

With :math:`n` initial candidates, reduction factor :math:`\eta`, and :math:`R` rungs:

.. math::

    n_r = \left\lfloor n\,\eta^{-r} \right\rfloor, \qquad
    b_r = b_0\,\eta^{r}

so each rung spends roughly :math:`n_r b_r \approx n b_0` — the *same* compute per rung.
Total cost is about :math:`R \cdot n \cdot b_0`, versus :math:`n \cdot b_{R-1}` for training
everything at full fidelity: a saving of roughly :math:`\eta^{R-1}/R`. With
:math:`\eta = 3` and :math:`R = 3` that is about a 3x saving, and the saving grows with
the ladder.

The assumption, stated plainly
-------------------------------
Successive halving assumes **low-fidelity rank correlates with high-fidelity rank**. If it
does not, the method confidently discards the eventual winner and no amount of compute at
later rungs recovers it.

The assumption fails in identifiable ways:

* **Slow starters.** Deep or unnormalised networks often train slowly at first and
  overtake later. A one-epoch rung systematically prefers shallow, wide models.
* **Regularisation crossover.** A model with strong regularisation (dropout, small width)
  underperforms early and wins late. Low-fidelity ranking is biased against it.
* **Learning-rate schedules.** A cosine schedule compressed into one epoch is a different
  optimisation problem from the same schedule over thirty. This implementation rescales
  the schedule to each rung's budget, which mitigates but does not eliminate the effect.
* **Data-fraction fidelity.** Halving the data changes the effective regularisation, not
  just the compute. Small models suffer least, so this dimension is biased towards small
  models.

Practical mitigations, all supported here: use a reduction factor of 3 rather than 2 so
fewer rungs are needed; make rung 0 large enough to be informative; and prefer scaling
epochs over scaling data. ``docs/concepts/successive-halving.md`` covers this in depth.

Relationship to Hyperband
-------------------------
Hyperband runs several successive-halving brackets with different :math:`(n, b_0)`
trade-offs to hedge against a badly chosen starting fidelity. This project implements the
single-bracket algorithm; the bracket loop is a documented extension point rather than an
implementation, because a single bracket is what a small local budget can actually afford.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import CheckpointError, CheckpointVersionError, ConfigurationError
from nas_engine.observability.logging import get_logger
from nas_engine.search.strategy import (
    Observation,
    Proposal,
    SearchStrategy,
    StrategyStatistics,
    deserialize_spec,
    serialize_spec,
)
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace
from nas_engine.utilities.seeding import derive_seed

_LOGGER = get_logger(__name__)

#: Version of this strategy's state payload.
SUCCESSIVE_HALVING_STATE_VERSION: int = 1


@dataclass(frozen=True)
class ResourceLadder:
    """The sequence of budgets a successive-halving bracket climbs.

    Attributes:
        base_budget: Budget for rung 0, the cheapest rung.
        num_rungs: Number of rungs including rung 0.
        reduction_factor: ``eta``; each rung multiplies resources by this and divides the
            surviving population by it.
        scale_epochs: Whether epochs grow with the rung.
        scale_train_fraction: Whether the training-data fraction grows with the rung.
        scale_resolution: Whether input resolution grows with the rung.
        native_resolution: The dataset's native resolution, used as the ceiling when
            scaling resolution.

    Raises:
        ConfigurationError: If the ladder is not usable.
    """

    base_budget: TrainingBudget
    num_rungs: int = 3
    reduction_factor: float = 3.0
    scale_epochs: bool = True
    scale_train_fraction: bool = False
    scale_resolution: bool = False
    native_resolution: int | None = None

    def __post_init__(self) -> None:
        """Validate the ladder.

        Raises:
            ConfigurationError: If the rung count or reduction factor is invalid, or no
                resource dimension is being scaled.
        """
        if self.num_rungs < 1:
            msg = f"num_rungs must be >= 1, received {self.num_rungs}"
            raise ConfigurationError(msg, details={"num_rungs": self.num_rungs})
        if self.reduction_factor <= 1.0:
            msg = (
                f"reduction_factor must be > 1, received {self.reduction_factor}; a factor "
                "of 1 or less would never reduce the candidate pool"
            )
            raise ConfigurationError(msg, details={"reduction_factor": self.reduction_factor})
        if self.num_rungs > 1 and not (
            self.scale_epochs or self.scale_train_fraction or self.scale_resolution
        ):
            msg = (
                "successive halving needs at least one resource dimension to scale; enable "
                "scale_epochs, scale_train_fraction, or scale_resolution"
            )
            raise ConfigurationError(msg)
        if self.scale_resolution and self.native_resolution is None:
            msg = "scale_resolution requires native_resolution to be set"
            raise ConfigurationError(msg)

    def budgets(self) -> tuple[TrainingBudget, ...]:
        """Return one budget per rung, cheapest first.

        Returns:
            The ladder's budgets.
        """
        result: list[TrainingBudget] = []
        for rung in range(self.num_rungs):
            factor = self.reduction_factor**rung
            epochs = (
                max(1, round(self.base_budget.epochs * factor))
                if self.scale_epochs
                else self.base_budget.epochs
            )
            fraction = (
                min(1.0, self.base_budget.train_fraction * factor)
                if self.scale_train_fraction
                else self.base_budget.train_fraction
            )
            resolution = self.base_budget.resolution
            if self.scale_resolution and self.base_budget.resolution is not None:
                assert self.native_resolution is not None  # guaranteed by validation
                scaled = round(self.base_budget.resolution * factor)
                # Round to an even size so repeated halving stays exact.
                scaled = max(4, scaled - (scaled % 2))
                resolution = min(self.native_resolution, scaled)
                if resolution >= self.native_resolution:
                    resolution = None
            result.append(
                TrainingBudget(
                    epochs=epochs,
                    train_fraction=fraction,
                    resolution=resolution,
                    max_seconds=self.base_budget.max_seconds,
                    rung=rung,
                )
            )
        return tuple(result)

    def rung_sizes(self, initial_candidates: int) -> tuple[int, ...]:
        """Return the number of candidates evaluated at each rung.

        Args:
            initial_candidates: Candidates at rung 0.

        Returns:
            One count per rung, never below 1.
        """
        sizes: list[int] = []
        for rung in range(self.num_rungs):
            count = math.floor(initial_candidates / (self.reduction_factor**rung))
            sizes.append(max(1, count))
        return tuple(sizes)

    def total_evaluations(self, initial_candidates: int) -> int:
        """Return the total number of evaluations the bracket will perform."""
        return sum(self.rung_sizes(initial_candidates))


@dataclass
class _RungState:
    """Bookkeeping for one rung.

    Attributes:
        proposed: Candidate ids proposed at this rung, in proposal order.
        results: Objective value by candidate id; ``None`` for failed evaluations.
        promoted: Architecture hashes selected for the next rung.
    """

    proposed: list[str]
    results: dict[str, float | None]
    promoted: list[str]

    @property
    def outstanding(self) -> int:
        """Number of proposals still awaiting a result."""
        return len(self.proposed) - len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "proposed": list(self.proposed),
            "results": dict(self.results),
            "promoted": list(self.promoted),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _RungState:
        """Rebuild rung state from :meth:`to_dict` output."""
        return cls(
            proposed=list(payload.get("proposed", [])),
            results={
                key: (float(value) if value is not None else None)
                for key, value in payload.get("results", {}).items()
            },
            promoted=list(payload.get("promoted", [])),
        )


class SuccessiveHalving(SearchStrategy):
    """Single-bracket successive halving.

    Args:
        space: Space to sample rung-0 candidates from.
        seed: Seed for the private sampler.
        ladder: The resource ladder.
        initial_candidates: Candidates evaluated at rung 0.

    Raises:
        ValueError: If ``initial_candidates`` is not positive.
    """

    name: ClassVar[str] = "successive_halving"
    # A rung is a barrier: promotions cannot be decided until every candidate in the rung
    # has reported. The engine may still evaluate a rung's candidates concurrently; the
    # barrier is expressed by `propose` returning an empty list while results are pending.
    requires_synchronous_observations: ClassVar[bool] = False

    def __init__(
        self,
        space: SearchSpace,
        *,
        seed: int,
        ladder: ResourceLadder,
        initial_candidates: int = 9,
    ) -> None:
        if initial_candidates < 1:
            msg = f"initial_candidates must be >= 1, received {initial_candidates}"
            raise ValueError(msg)
        self._space = space
        self._seed = seed
        self._ladder = ladder
        self._initial_candidates = initial_candidates
        self._budgets = ladder.budgets()
        self._sizes = ladder.rung_sizes(initial_candidates)
        self._sampler = ArchitectureSampler(
            space, seed=derive_seed(seed, "successive_halving:sampler")
        )
        self._rungs: list[_RungState] = [
            _RungState(proposed=[], results={}, promoted=[]) for _ in self._budgets
        ]
        self._specs: dict[str, ArchitectureSpec] = {}
        self._candidate_rung: dict[str, int] = {}
        self._seen: set[str] = set()
        self._current_rung = 0
        self._proposed = 0
        self._observed = 0
        self._succeeded = 0
        self._failed = 0
        self._duplicates = 0
        self._exhausted = False

    # -- introspection --------------------------------------------------------------
    @property
    def budgets(self) -> tuple[TrainingBudget, ...]:
        """The ladder's budgets, cheapest first."""
        return self._budgets

    @property
    def rung_sizes(self) -> tuple[int, ...]:
        """Planned candidate count per rung."""
        return self._sizes

    @property
    def current_rung(self) -> int:
        """Index of the rung currently being filled."""
        return self._current_rung

    def rung_summary(self) -> list[dict[str, Any]]:
        """Return a per-rung summary for reports.

        Returns:
            One dictionary per rung describing its budget, planned size, and progress.
        """
        return [
            {
                "rung": index,
                "budget": budget.describe(),
                "planned": self._sizes[index],
                "proposed": len(state.proposed),
                "completed": len(state.results),
                "promoted": len(state.promoted),
            }
            for index, (budget, state) in enumerate(zip(self._budgets, self._rungs, strict=True))
        ]

    # -- strategy interface ---------------------------------------------------------
    def propose(self, count: int) -> list[Proposal]:
        """Return up to ``count`` proposals for the current rung.

        Returns an empty list when the current rung is fully proposed but not yet fully
        observed: the barrier cannot be crossed until every result is in.

        Args:
            count: Maximum proposals wanted.

        Returns:
            The proposals.
        """
        proposals: list[Proposal] = []
        while len(proposals) < count:
            if self._exhausted or self._current_rung >= len(self._budgets):
                break
            state = self._rungs[self._current_rung]
            planned = self._sizes[self._current_rung]

            if len(state.proposed) < planned:
                proposal = self._propose_for_rung(self._current_rung)
                if proposal is None:
                    break
                proposals.append(proposal)
                continue

            if state.outstanding > 0:
                # Barrier: wait for the rung to finish before promoting.
                break

            if not self._advance_rung():
                break
        return proposals

    def _propose_for_rung(self, rung: int) -> Proposal | None:
        """Produce one proposal for ``rung``.

        Args:
            rung: Rung index.

        Returns:
            A proposal, or ``None`` when none can be produced.
        """
        state = self._rungs[rung]
        budget = self._budgets[rung]

        if rung == 0:
            spec = self._sampler.sample_unique(self._seen)
            if spec is None:
                self._exhausted = True
                _LOGGER.warning(
                    "successive_halving.exhausted",
                    rung=rung,
                    proposed=len(state.proposed),
                    planned=self._sizes[rung],
                )
                return None
            hash_value = architecture_hash(spec)
            self._seen.add(hash_value)
            origin = "random"
            parent_id = None
        else:
            # Survivors are recorded on the rung they were promoted *from*, so a rung
            # reads its intake from its predecessor rather than from itself.
            promoted = self._rungs[rung - 1].promoted
            position = len(state.proposed)
            if position >= len(promoted):
                return None
            hash_value = promoted[position]
            spec = self._specs[hash_value]
            origin = "promotion"
            parent_id = None

        candidate_key = f"{hash_value}:{rung}"
        state.proposed.append(candidate_key)
        self._candidate_rung[candidate_key] = rung
        self._specs[hash_value] = spec
        self._proposed += 1
        return Proposal(
            spec=spec,
            budget=budget,
            parent_id=parent_id,
            origin=origin,
            metadata={"rung": rung, "rung_key": candidate_key},
        )

    def _advance_rung(self) -> bool:
        """Select survivors and move to the next rung.

        Returns:
            ``True`` when a further rung is available, ``False`` when the bracket is done.
        """
        state = self._rungs[self._current_rung]
        scored = [(value, key) for key, value in state.results.items() if value is not None]
        if self._current_rung + 1 >= len(self._budgets):
            self._current_rung = len(self._budgets)
            return False

        survivors = self._sizes[self._current_rung + 1]
        # Sort by objective descending, breaking ties by key so promotion is deterministic.
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        promoted_hashes = [key.rsplit(":", 1)[0] for _, key in scored[:survivors]]

        if not promoted_hashes:
            self._exhausted = True
            _LOGGER.warning(
                "successive_halving.no_survivors",
                rung=self._current_rung,
                completed=len(state.results),
                failed=sum(1 for value in state.results.values() if value is None),
            )
            return False

        state.promoted = promoted_hashes
        self._current_rung += 1
        _LOGGER.info(
            "successive_halving.rung_advanced",
            from_rung=self._current_rung - 1,
            to_rung=self._current_rung,
            promoted=len(promoted_hashes),
            next_budget=self._budgets[self._current_rung].describe(),
        )
        return True

    def observe(self, observation: Observation) -> None:
        """Record an evaluation result against its rung.

        Args:
            observation: The evaluation outcome.
        """
        self._observed += 1
        rung = observation.result.budget.rung
        key = f"{observation.architecture_hash}:{rung}"
        self._specs.setdefault(observation.architecture_hash, observation.spec)

        if rung >= len(self._rungs):  # pragma: no cover - defensive against stale results
            _LOGGER.warning(
                "successive_halving.unknown_rung",
                rung=rung,
                architecture_hash=observation.architecture_hash,
            )
            return

        state = self._rungs[rung]
        if key not in state.proposed:
            # A result for something this bracket never proposed: possible after a resume
            # where the database holds trials from a longer previous run. Record it so the
            # rung can still complete, but do not let it inflate the planned count.
            state.proposed.append(key)
        if observation.succeeded and observation.objective_value is not None:
            self._succeeded += 1
            state.results[key] = float(observation.objective_value)
        else:
            self._failed += 1
            state.results[key] = None

    def is_finished(self) -> bool:
        """Report whether every rung is complete or the bracket cannot continue."""
        if self._exhausted:
            return True
        if self._current_rung >= len(self._budgets):
            return True
        last = len(self._budgets) - 1
        final_state = self._rungs[last]
        return (
            self._current_rung == last
            and len(final_state.proposed) >= self._sizes[last]
            and final_state.outstanding == 0
        )

    def on_duplicate(self, architecture_hash: str) -> None:
        """Record a duplicate rejected by the engine."""
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
                "rungs": self.rung_summary(),
                "current_rung": self._current_rung,
                "initial_candidates": self._initial_candidates,
                "reduction_factor": self._ladder.reduction_factor,
                "planned_evaluations": self._ladder.total_evaluations(self._initial_candidates),
                "unique_architectures": len(self._seen),
                "exhausted": self._exhausted,
            },
        )

    def describe(self) -> str:
        """Return a human-readable configuration summary."""
        ladder = " -> ".join(
            f"{size}@{budget.epochs}e"
            for size, budget in zip(self._sizes, self._budgets, strict=True)
        )
        return (
            f"successive_halving: eta={self._ladder.reduction_factor:g}, ladder {ladder}, "
            f"{self._ladder.total_evaluations(self._initial_candidates)} total evaluations"
        )

    # -- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the strategy state."""
        return {
            "version": SUCCESSIVE_HALVING_STATE_VERSION,
            "strategy": self.name,
            "seed": self._seed,
            "initial_candidates": self._initial_candidates,
            "reduction_factor": self._ladder.reduction_factor,
            "num_rungs": len(self._budgets),
            "current_rung": self._current_rung,
            "exhausted": self._exhausted,
            "rungs": [state.to_dict() for state in self._rungs],
            "specs": {hash_value: serialize_spec(spec) for hash_value, spec in self._specs.items()},
            "seen": sorted(self._seen),
            "counters": {
                "proposed": self._proposed,
                "observed": self._observed,
                "succeeded": self._succeeded,
                "failed": self._failed,
                "duplicates": self._duplicates,
            },
            "sampler": self._sampler.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore state captured by :meth:`state_dict`.

        Args:
            payload: Previously captured state.

        Raises:
            CheckpointVersionError: If the payload version is unsupported.
            CheckpointError: If the payload is malformed or the ladder shape changed.
        """
        version = payload.get("version")
        if version != SUCCESSIVE_HALVING_STATE_VERSION:
            msg = (
                f"successive_halving state version {version} is not supported by this "
                f"build (expected {SUCCESSIVE_HALVING_STATE_VERSION})"
            )
            raise CheckpointVersionError(msg, details={"version": version})

        stored_rungs = int(payload.get("num_rungs", len(self._budgets)))
        if stored_rungs != len(self._budgets):
            msg = (
                f"checkpoint has {stored_rungs} rungs but the current configuration "
                f"defines {len(self._budgets)}; the ladder must not change across a resume"
            )
            raise CheckpointError(
                msg, details={"checkpoint": stored_rungs, "configured": len(self._budgets)}
            )

        try:
            self._rungs = [_RungState.from_dict(entry) for entry in payload["rungs"]]
            self._specs = {
                hash_value: deserialize_spec(spec)
                for hash_value, spec in payload.get("specs", {}).items()
            }
            self._seen = set(payload.get("seen", []))
            self._current_rung = int(payload.get("current_rung", 0))
            self._exhausted = bool(payload.get("exhausted", False))
            counters = payload.get("counters", {})
            self._proposed = int(counters.get("proposed", 0))
            self._observed = int(counters.get("observed", 0))
            self._succeeded = int(counters.get("succeeded", 0))
            self._failed = int(counters.get("failed", 0))
            self._duplicates = int(counters.get("duplicates", 0))
            self._sampler.load_state_dict(payload["sampler"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"successive_halving state payload is malformed: {exc}"
            raise CheckpointError(msg, details={"error": str(exc)}) from exc


__all__ = [
    "SUCCESSIVE_HALVING_STATE_VERSION",
    "ResourceLadder",
    "SuccessiveHalving",
]
