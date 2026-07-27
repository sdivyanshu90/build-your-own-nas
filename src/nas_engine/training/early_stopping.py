r"""Early stopping.

Early stopping ends training when the monitored validation metric has not improved for
``patience`` consecutive epochs. In NAS its value is not primarily regularisation — the
budgets are too short for much overfitting — it is **budget reallocation**: an architecture
that has plateaued at 30% accuracy will not reach 80% in the remaining epochs, and the
compute is better spent on the next candidate.

Two subtleties the implementation handles explicitly:

* ``min_delta`` guards against declaring improvement from noise. Validation accuracy on a
  small split has a standard error of roughly :math:`\sqrt{p(1-p)/n}`, which for
  :math:`n = 256` is about 3 percentage points. A default ``min_delta`` of zero counts any
  numerical increase as progress, so a threshold should be set deliberately when the
  validation split is small.
* The **best** state is what matters, not the last. The trainer restores the best epoch's
  weights when training stops, so a late-epoch collapse cannot be mistaken for the
  candidate's quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nas_engine.exceptions import ConfigurationError


class MonitorMode(str, Enum):
    """Whether the monitored metric should increase or decrease.

    Members:
        MAX: Higher is better (accuracy).
        MIN: Lower is better (loss).
    """

    MAX = "max"
    MIN = "min"


@dataclass
class EarlyStopping:
    """Tracks the best monitored value and decides when to stop.

    Attributes:
        patience: Epochs without improvement tolerated before stopping. ``0`` disables
            early stopping entirely.
        min_delta: Minimum change that counts as an improvement.
        mode: Direction of improvement.
        best_value: Best value seen so far, or ``None`` before the first update.
        best_epoch: Epoch index at which ``best_value`` was observed.
        epochs_without_improvement: Consecutive non-improving epochs.

    Example:
        >>> stopper = EarlyStopping(patience=1, mode=MonitorMode.MAX)
        >>> stopper.update(0.5, epoch=0)
        True
        >>> stopper.update(0.4, epoch=1)
        False
        >>> stopper.should_stop
        False
        >>> stopper.update(0.3, epoch=2)
        False
        >>> stopper.should_stop
        True
    """

    patience: int = 0
    min_delta: float = 0.0
    mode: MonitorMode = MonitorMode.MAX
    best_value: float | None = None
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ConfigurationError: If ``patience`` or ``min_delta`` is negative.
        """
        if self.patience < 0:
            msg = f"patience must be non-negative, received {self.patience}"
            raise ConfigurationError(msg, details={"patience": self.patience})
        if self.min_delta < 0:
            msg = f"min_delta must be non-negative, received {self.min_delta}"
            raise ConfigurationError(msg, details={"min_delta": self.min_delta})

    @property
    def enabled(self) -> bool:
        """Whether early stopping can ever trigger."""
        return self.patience > 0

    def is_improvement(self, value: float) -> bool:
        """Report whether ``value`` beats the best seen by at least ``min_delta``.

        Args:
            value: Newly observed metric value.

        Returns:
            ``True`` when this counts as an improvement.
        """
        if self.best_value is None:
            return True
        if self.mode is MonitorMode.MAX:
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, value: float, *, epoch: int) -> bool:
        """Record an observation and return whether it was an improvement.

        Args:
            value: Newly observed metric value.
            epoch: Epoch index the value belongs to.

        Returns:
            ``True`` when the value improved on the best seen so far.
        """
        if self.is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether training should stop now."""
        return self.enabled and self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-serialisable snapshot for checkpointing."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode.value,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(self, payload: dict[str, float | int | str | None]) -> None:
        """Restore state captured by :meth:`state_dict`.

        Args:
            payload: Previously captured state.
        """
        best = payload.get("best_value")
        self.best_value = float(best) if best is not None else None
        self.best_epoch = int(payload.get("best_epoch", -1) or -1)
        self.epochs_without_improvement = int(payload.get("epochs_without_improvement", 0) or 0)


__all__ = ["EarlyStopping", "MonitorMode"]
