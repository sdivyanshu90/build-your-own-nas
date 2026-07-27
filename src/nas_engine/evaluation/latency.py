"""Inference latency benchmarking.

**Latency numbers produced here are not portable.** They describe one model, on one
machine, at one thread count, with one batch size, under whatever else that machine was
doing. Comparing a latency measured on a laptop against one measured in CI is
meaningless. Every measurement carries its device metadata and an explicit warning for
exactly this reason, and reports repeat the warning rather than presenting the number as
an intrinsic property of the architecture.

What the measurement is good for is *relative* comparison within one search run on one
machine, which is what hardware-aware NAS actually needs.

Methodology
-----------
1. **Warm-up.** The first forward passes are unrepresentative: memory allocators are cold,
   cuDNN may still be selecting algorithms, and CPU frequency scaling has not settled.
   Warm-up iterations are timed but discarded.
2. **Repeated timed blocks.** ``repeats`` blocks of ``iterations`` forward passes each.
   Timing a block rather than an individual call amortises clock-read overhead, which is
   comparable to a small model's forward pass.
3. **Median and percentiles, not mean.** Latency distributions have a long right tail from
   scheduler preemption and garbage collection. The mean tracks the tail; the median
   tracks the typical case. Both are reported, along with p90 and p99, so a suspiciously
   large gap between median and p99 is visible.
4. **Explicit synchronisation.** CUDA kernel launches are asynchronous, so without
   ``torch.cuda.synchronize`` the measurement would time the launch, not the computation.
5. **Fixed thread count.** Recorded in the metadata, because a CPU latency measured with
   8 threads is not comparable to one measured with 1.
"""

from __future__ import annotations

import platform
import statistics
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nas_engine.exceptions import ConfigurationError
from nas_engine.training.trainer import evaluation_mode

#: The standard caveat attached to every measurement.
LATENCY_WARNING: str = (
    "Latency is hardware-, thread-, and load-dependent. These numbers are comparable "
    "only between candidates measured on the same machine during the same run. Do not "
    "compare them across machines, batch sizes, or PyTorch versions."
)


@dataclass(frozen=True)
class LatencyMeasurement:
    """Latency statistics for one model on one device.

    Attributes:
        median_ms: Median per-batch latency in milliseconds.
        mean_ms: Mean per-batch latency.
        std_ms: Standard deviation across timed blocks.
        min_ms: Fastest observed block.
        p90_ms: 90th percentile.
        p99_ms: 99th percentile.
        per_image_ms: Median latency divided by batch size.
        batch_size: Batch size used.
        input_shape: Full input shape including the batch dimension.
        warmup_iterations: Discarded warm-up iterations.
        timed_iterations: Iterations per timed block.
        repeats: Number of timed blocks.
        device: Device string.
        device_name: Human-readable device description.
        torch_threads: Intra-op thread count in effect.
        warning: The portability caveat.
    """

    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    p90_ms: float
    p99_ms: float
    per_image_ms: float
    batch_size: int
    input_shape: tuple[int, ...]
    warmup_iterations: int
    timed_iterations: int
    repeats: int
    device: str
    device_name: str
    torch_threads: int
    warning: str = LATENCY_WARNING
    samples_ms: tuple[float, ...] = field(default_factory=tuple)

    def to_metrics(self) -> dict[str, float]:
        """Return the subset of statistics used as optimisation metrics."""
        return {
            "latency_median_ms": self.median_ms,
            "latency_mean_ms": self.mean_ms,
            "latency_p90_ms": self.p90_ms,
            "latency_p99_ms": self.p99_ms,
            "latency_per_image_ms": self.per_image_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "per_image_ms": self.per_image_ms,
            "batch_size": self.batch_size,
            "input_shape": list(self.input_shape),
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "repeats": self.repeats,
            "device": self.device,
            "device_name": self.device_name,
            "torch_threads": self.torch_threads,
            "warning": self.warning,
        }


def _percentile(values: list[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``values`` by nearest-rank.

    Nearest-rank rather than interpolation: with as few as five samples, interpolating
    invents values that were never observed.

    Args:
        values: Observations; need not be sorted.
        fraction: Percentile in ``[0, 1]``.

    Returns:
        The selected observation.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _device_name(device: torch.device) -> str:
    """Return a human-readable description of a device.

    Args:
        device: Device to describe.

    Returns:
        A descriptive string, e.g. a GPU model name or the host CPU description.
    """
    if device.type == "cuda" and torch.cuda.is_available():  # pragma: no cover - needs a GPU
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "mps":  # pragma: no cover - needs Apple Silicon
        return "Apple Silicon GPU (MPS)"
    return platform.processor() or platform.machine() or "cpu"


def measure_latency(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    warmup_iterations: int = 5,
    timed_iterations: int = 10,
    repeats: int = 5,
) -> LatencyMeasurement:
    """Measure a model's forward-pass latency.

    Args:
        model: Model to measure. Its training mode is restored afterwards.
        input_shape: ``(channels, height, width)`` of one example.
        device: Device to measure on.
        batch_size: Batch size for the synthetic input.
        warmup_iterations: Untimed warm-up passes.
        timed_iterations: Forward passes per timed block.
        repeats: Number of timed blocks.

    Returns:
        A :class:`LatencyMeasurement`.

    Raises:
        ConfigurationError: If any count is not positive.
    """
    if batch_size < 1 or timed_iterations < 1 or repeats < 1 or warmup_iterations < 0:
        msg = (
            "latency measurement requires batch_size>=1, timed_iterations>=1, repeats>=1 "
            f"and warmup_iterations>=0; received batch_size={batch_size}, "
            f"timed_iterations={timed_iterations}, repeats={repeats}, "
            f"warmup_iterations={warmup_iterations}"
        )
        raise ConfigurationError(
            msg,
            details={
                "batch_size": batch_size,
                "timed_iterations": timed_iterations,
                "repeats": repeats,
                "warmup_iterations": warmup_iterations,
            },
        )

    torch_device = torch.device(device)
    model = model.to(torch_device)
    sample = torch.randn((batch_size, *input_shape), device=torch_device)

    def synchronize() -> None:
        if torch_device.type == "cuda":  # pragma: no cover - needs a GPU
            torch.cuda.synchronize(torch_device)

    block_times: list[float] = []
    with evaluation_mode(model), torch.no_grad():
        for _ in range(warmup_iterations):
            model(sample)
        synchronize()

        for _ in range(repeats):
            # `perf_counter` is monotonic and has nanosecond resolution on all supported
            # platforms; `time.time` would be vulnerable to clock adjustments mid-run.
            import time

            start = time.perf_counter()
            for _ in range(timed_iterations):
                model(sample)
            synchronize()
            elapsed = time.perf_counter() - start
            block_times.append(elapsed / timed_iterations * 1000.0)

    median = statistics.median(block_times)
    return LatencyMeasurement(
        median_ms=median,
        mean_ms=statistics.fmean(block_times),
        std_ms=statistics.pstdev(block_times) if len(block_times) > 1 else 0.0,
        min_ms=min(block_times),
        p90_ms=_percentile(block_times, 0.90),
        p99_ms=_percentile(block_times, 0.99),
        per_image_ms=median / batch_size,
        batch_size=batch_size,
        input_shape=(batch_size, *input_shape),
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        repeats=repeats,
        device=str(torch_device),
        device_name=_device_name(torch_device),
        torch_threads=int(torch.get_num_threads()),
        samples_ms=tuple(block_times),
    )


__all__ = ["LATENCY_WARNING", "LatencyMeasurement", "measure_latency"]
