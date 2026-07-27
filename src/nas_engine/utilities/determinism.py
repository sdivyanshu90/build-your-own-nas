"""PyTorch determinism configuration.

Seeding fixes the *inputs* to the random number generators. It does not fix the
*order of floating-point reductions* inside kernels. Many GPU kernels (and some
threaded CPU kernels) accumulate partial sums in a nondeterministic order, and
floating-point addition is not associative, so ``(a + b) + c != a + (b + c)`` at the
bit level. Two runs with identical seeds can therefore differ in the last few bits,
which occasionally flips an ``argmax`` and changes reported accuracy.

This module enables PyTorch's deterministic algorithm selection where available and
reports honestly which guarantees are actually in force. It never silently promises
bit-for-bit reproducibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DeterminismReport:
    """What determinism settings were actually applied.

    Attributes:
        requested: Whether deterministic mode was requested by configuration.
        deterministic_algorithms: Whether PyTorch deterministic algorithms are active.
        cudnn_deterministic: Whether cuDNN's deterministic flag is set.
        cudnn_benchmark: Whether cuDNN autotuning is active (nondeterministic).
        cublas_workspace_config: Value of ``CUBLAS_WORKSPACE_CONFIG``, if set.
        warnings: Human-readable caveats about remaining sources of variation.
    """

    requested: bool
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation for logs and reports."""
        return {
            "requested": self.requested,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cublas_workspace_config": self.cublas_workspace_config,
            "warnings": list(self.warnings),
        }


def configure_determinism(*, enabled: bool, warn_only: bool = True) -> DeterminismReport:
    """Configure PyTorch determinism and report what was achieved.

    Args:
        enabled: When ``True``, request deterministic kernels and disable cuDNN
            autotuning. When ``False``, restore throughput-oriented defaults.
        warn_only: When ``True``, operations without a deterministic implementation
            emit a warning instead of raising. This is the default because several
            common layers (for example some pooling backward passes on CUDA) have no
            deterministic kernel, and hard-failing would make the framework unusable
            on GPU. Set to ``False`` in determinism test suites where any
            nondeterministic operation should be a hard error.

    Returns:
        A :class:`DeterminismReport` describing the resulting configuration.
    """
    warnings: list[str] = []

    if not enabled:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return DeterminismReport(
            requested=False,
            deterministic_algorithms=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            warnings=("determinism disabled: repeated runs may differ numerically",),
        )

    # cuBLAS GEMM workspaces must be fixed *before* the first CUDA context is created,
    # otherwise deterministic matmul cannot be guaranteed. Setting it unconditionally
    # is harmless on CPU-only hosts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    deterministic_algorithms = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except (RuntimeError, TypeError) as exc:  # pragma: no cover - version dependent
        deterministic_algorithms = False
        warnings.append(f"torch.use_deterministic_algorithms unavailable: {exc}")

    if torch.cuda.is_available():
        warnings.append(
            "CUDA is in use: atomics and reduction order can still vary between "
            "driver, cuDNN, and GPU-architecture versions"
        )
    thread_count = torch.get_num_threads()
    if thread_count > 1:
        warnings.append(
            f"intra-op parallelism is enabled ({thread_count} threads); reduction "
            "order in some CPU kernels depends on the thread count"
        )

    return DeterminismReport(
        requested=True,
        deterministic_algorithms=deterministic_algorithms,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        warnings=tuple(warnings),
    )


__all__ = ["DeterminismReport", "configure_determinism"]
