"""Metric computation and aggregation.

Aggregation correctness
-----------------------
The mean of per-batch means is **not** the mean over examples unless every batch is the
same size. The final batch of an epoch is usually short, so naive averaging biases every
reported number. Every aggregator here therefore accumulates ``value * batch_size`` and
divides by the total example count at the end.

Top-k accuracy
--------------
Top-1 accuracy is a step function of the argmax and is therefore a coarse signal: two
architectures can differ substantially in how confidently they rank the true class while
both scoring the same top-1. Top-k is recorded alongside because it changes more smoothly
and is a useful secondary tie-breaker when short training leaves top-1 values clustered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for one batch.

    Args:
        logits: Model outputs of shape ``(batch, num_classes)``.
        targets: Integer labels of shape ``(batch,)``.

    Returns:
        Fraction of correct predictions, in ``[0, 1]``.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if logits.ndim != 2:
        msg = f"logits must be 2-D (batch, classes), received shape {tuple(logits.shape)}"
        raise ValueError(msg)
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        msg = (
            f"targets must be 1-D with the same batch size as logits; received "
            f"targets {tuple(targets.shape)} and logits {tuple(logits.shape)}"
        )
        raise ValueError(msg)
    if targets.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy for one batch.

    ``k`` is clamped to the number of classes, so requesting top-5 on a 3-class problem
    returns 1.0 rather than raising — that is the mathematically correct answer and it
    keeps small synthetic datasets usable with the default configuration.

    Args:
        logits: Model outputs of shape ``(batch, num_classes)``.
        targets: Integer labels of shape ``(batch,)``.
        k: Number of top predictions to consider.

    Returns:
        Fraction of examples whose true label appears in the top ``k`` predictions.

    Raises:
        ValueError: If ``k`` is not positive or shapes are incompatible.
    """
    if k < 1:
        msg = f"k must be >= 1, received {k}"
        raise ValueError(msg)
    if logits.ndim != 2:
        msg = f"logits must be 2-D (batch, classes), received shape {tuple(logits.shape)}"
        raise ValueError(msg)
    if targets.numel() == 0:
        return 0.0
    effective_k = min(k, logits.shape[1])
    top = logits.topk(effective_k, dim=1).indices
    matches = top.eq(targets.unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


@dataclass
class MetricAggregator:
    """Accumulates example-weighted means of named scalar metrics.

    Example:
        >>> aggregator = MetricAggregator()
        >>> aggregator.update({"loss": 1.0}, batch_size=8)
        >>> aggregator.update({"loss": 0.0}, batch_size=2)
        >>> round(aggregator.compute()["loss"], 3)
        0.8
    """

    totals: dict[str, float] = field(default_factory=dict)
    examples: int = 0

    def update(self, values: dict[str, float], *, batch_size: int) -> None:
        """Add one batch of observations.

        Args:
            values: Metric name to batch-mean value.
            batch_size: Number of examples the values were computed over.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size < 1:
            msg = f"batch_size must be >= 1, received {batch_size}"
            raise ValueError(msg)
        for name, value in values.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * batch_size
        self.examples += batch_size

    def compute(self) -> dict[str, float]:
        """Return the example-weighted mean of every accumulated metric.

        Returns:
            A mapping from metric name to mean value; empty when nothing was recorded.
        """
        if self.examples == 0:
            return {}
        return {name: total / self.examples for name, total in self.totals.items()}

    def reset(self) -> None:
        """Discard all accumulated values."""
        self.totals.clear()
        self.examples = 0


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics for one epoch of one phase.

    Attributes:
        epoch: Zero-based epoch index.
        phase: ``"train"``, ``"validation"``, or ``"test"``.
        loss: Mean cross-entropy loss.
        accuracy: Top-1 accuracy.
        topk_accuracy: Top-k accuracy.
        examples: Number of examples the metrics cover.
        duration_seconds: Wall-clock duration of the phase.
        learning_rate: Learning rate in effect, recorded for training phases.
    """

    epoch: int
    phase: str
    loss: float
    accuracy: float
    topk_accuracy: float
    examples: int
    duration_seconds: float
    learning_rate: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-serialisable representation."""
        return {
            "epoch": self.epoch,
            "phase": self.phase,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "topk_accuracy": self.topk_accuracy,
            "examples": self.examples,
            "duration_seconds": self.duration_seconds,
            "learning_rate": self.learning_rate,
        }


__all__ = ["EpochMetrics", "MetricAggregator", "accuracy", "topk_accuracy"]
