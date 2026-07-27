"""Training budgets: the resource allocation given to one evaluation.

A budget is the *fidelity* at which a candidate is measured. Multi-fidelity search spends
a small budget on many candidates and a large budget on the few that survive, so a budget
must be a first-class, serialisable value: it is persisted with every trial, it identifies
which rung of a successive-halving ladder a measurement belongs to, and two measurements
of the same architecture at different budgets must never be confused with each other.

Three independent resource dimensions
--------------------------------------
``epochs``
    Passes over the training data. Directly proportional to cost.
``train_fraction``
    Fraction of the training split used. Also directly proportional.
``resolution``
    Input side length. Convolution cost scales with pixel count, so cost scales
    approximately with the *square* of this.

The three are independent, so a budget forms a point in a three-dimensional resource
space. Successive halving moves along whichever dimensions the configuration enables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nas_engine.exceptions import ConfigurationError


@dataclass(frozen=True)
class TrainingBudget:
    """The resources allocated to one candidate evaluation.

    Attributes:
        epochs: Training epochs.
        train_fraction: Fraction of the training split to use, in ``(0, 1]``.
        resolution: Input resolution, or ``None`` for the dataset's native size.
        max_seconds: Wall-clock limit for the evaluation, or ``None`` for no limit.
        rung: Successive-halving rung index; ``0`` for single-fidelity search.

    Raises:
        ConfigurationError: If any field is out of range.
    """

    epochs: int
    train_fraction: float = 1.0
    resolution: int | None = None
    max_seconds: float | None = None
    rung: int = 0

    def __post_init__(self) -> None:
        """Validate the budget.

        Raises:
            ConfigurationError: If any field is out of range.
        """
        if self.epochs < 1:
            msg = f"budget epochs must be >= 1, received {self.epochs}"
            raise ConfigurationError(msg, details={"epochs": self.epochs})
        if not 0.0 < self.train_fraction <= 1.0:
            msg = f"budget train_fraction must lie in (0, 1], received {self.train_fraction}"
            raise ConfigurationError(msg, details={"train_fraction": self.train_fraction})
        if self.resolution is not None and self.resolution < 4:
            msg = f"budget resolution must be at least 4, received {self.resolution}"
            raise ConfigurationError(msg, details={"resolution": self.resolution})
        if self.max_seconds is not None and self.max_seconds <= 0:
            msg = f"budget max_seconds must be positive or None, received {self.max_seconds}"
            raise ConfigurationError(msg, details={"max_seconds": self.max_seconds})
        if self.rung < 0:
            msg = f"budget rung must be non-negative, received {self.rung}"
            raise ConfigurationError(msg, details={"rung": self.rung})

    @property
    def relative_cost(self) -> float:
        """Approximate cost relative to a one-epoch, full-data, native-resolution run.

        Used to report how much compute a search actually consumed, and to sanity-check
        that a successive-halving ladder really is geometric.

        Returns:
            A dimensionless cost estimate.
        """
        cost = float(self.epochs) * self.train_fraction
        if self.resolution is not None:
            # Convolution work scales with the number of pixels. The native resolution is
            # unknown here, so the resolution factor is reported separately by the caller
            # when it matters; within one search all budgets share the native size, so the
            # ratio between budgets stays correct.
            cost *= float(self.resolution) ** 2
        return cost

    @property
    def key(self) -> str:
        """A stable, human-readable identifier used in artifact filenames and logs."""
        resolution = self.resolution if self.resolution is not None else "native"
        return f"e{self.epochs}_f{self.train_fraction:g}_r{resolution}_rung{self.rung}"

    def describe(self) -> str:
        """Return a short human-readable description."""
        parts = [f"{self.epochs} epochs"]
        if self.train_fraction < 1.0:
            parts.append(f"{self.train_fraction:.0%} of training data")
        if self.resolution is not None:
            parts.append(f"resolution {self.resolution}")
        if self.max_seconds is not None:
            parts.append(f"limit {self.max_seconds:g}s")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "epochs": self.epochs,
            "train_fraction": self.train_fraction,
            "resolution": self.resolution,
            "max_seconds": self.max_seconds,
            "rung": self.rung,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingBudget:
        """Rebuild a budget from :meth:`to_dict` output.

        Args:
            payload: Serialised budget.

        Returns:
            The reconstructed budget.

        Raises:
            ConfigurationError: If required fields are missing or invalid.
        """
        if "epochs" not in payload:
            msg = "budget payload is missing the required 'epochs' field"
            raise ConfigurationError(msg, details={"payload_keys": sorted(payload)})
        resolution = payload.get("resolution")
        max_seconds = payload.get("max_seconds")
        return cls(
            epochs=int(payload["epochs"]),
            train_fraction=float(payload.get("train_fraction", 1.0)),
            resolution=int(resolution) if resolution is not None else None,
            max_seconds=float(max_seconds) if max_seconds is not None else None,
            rung=int(payload.get("rung", 0)),
        )


__all__ = ["TrainingBudget"]
