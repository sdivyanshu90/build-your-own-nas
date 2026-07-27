"""Environment capture for reproducibility records.

A search result is only interpretable alongside the environment that produced it.
Accuracy numbers depend on library versions; latency numbers depend on the CPU or GPU
and on thread counts. Every search run persists an :class:`EnvironmentInfo` snapshot
so results can be compared honestly later.

Only non-sensitive, allow-listed environment variables are captured — see
``docs/architecture/security.md``.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Environment variables captured verbatim. Chosen because they materially affect
#: numerical results or device selection, and none of them carries credentials.
_CAPTURED_ENV_VARS: tuple[str, ...] = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TORCH_NUM_THREADS",
)


def _git_commit(repository_root: Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` when unavailable.

    The subprocess call uses a fixed argument list (never a shell string), so no
    user-controlled text can reach a shell. Failures are swallowed because running
    outside a git checkout is entirely normal.

    Args:
        repository_root: Directory to run ``git`` in; defaults to the current directory.

    Returns:
        The 40-character commit hash, or ``None``.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - resolved via PATH by design
            capture_output=True,
            check=False,
            cwd=str(repository_root) if repository_root else None,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _package_version() -> str:
    """Return the installed ``nas-engine`` version, or ``"unknown"``."""
    try:
        from importlib.metadata import version

        return version("nas-engine")
    except Exception:
        return "unknown"


def _accelerator_info() -> dict[str, Any]:
    """Describe available accelerators without initialising a CUDA context eagerly."""
    info: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "device_count": 0,
        "device_names": [],
        "mps_available": False,
    }
    backends = getattr(torch.backends, "mps", None)
    if backends is not None and hasattr(backends, "is_available"):
        info["mps_available"] = bool(backends.is_available())
    if info["cuda_available"]:  # pragma: no cover - requires a GPU host
        info["device_count"] = int(torch.cuda.device_count())
        info["device_names"] = [
            torch.cuda.get_device_name(index) for index in range(int(info["device_count"]))
        ]
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None and hasattr(cudnn, "version"):
            info["cudnn_version"] = cudnn.version()
    return info


@dataclass(frozen=True)
class EnvironmentInfo:
    """Immutable snapshot of the execution environment.

    Attributes:
        python_version: Full interpreter version string.
        python_implementation: e.g. ``"CPython"``.
        platform: Human-readable OS description.
        system: OS family, e.g. ``"Linux"``.
        machine: CPU architecture, e.g. ``"x86_64"`` or ``"arm64"``.
        processor: Processor description reported by the OS, when available.
        cpu_count: Logical CPU count.
        torch_version: Installed PyTorch version.
        torch_threads: Intra-op thread count in effect.
        accelerator: Accelerator availability and names.
        package_version: Installed ``nas-engine`` version.
        git_commit: Repository commit hash when running from a checkout.
        environment_variables: Allow-listed environment variables.
    """

    python_version: str
    python_implementation: str
    platform: str
    system: str
    machine: str
    processor: str
    cpu_count: int
    torch_version: str
    torch_threads: int
    accelerator: dict[str, Any]
    package_version: str
    git_commit: str | None
    environment_variables: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "torch_version": self.torch_version,
            "torch_threads": self.torch_threads,
            "accelerator": dict(self.accelerator),
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "environment_variables": dict(self.environment_variables),
        }

    def summary_lines(self) -> list[str]:
        """Return short human-readable lines for CLI output and Markdown reports."""
        accelerator = "cuda" if self.accelerator.get("cuda_available") else "cpu"
        if not self.accelerator.get("cuda_available") and self.accelerator.get("mps_available"):
            accelerator = "mps"
        return [
            f"nas-engine {self.package_version} (commit {self.git_commit or 'n/a'})",
            f"Python {self.python_version} ({self.python_implementation}) on {self.platform}",
            f"PyTorch {self.torch_version}, accelerator: {accelerator}",
            f"CPUs: {self.cpu_count}, torch threads: {self.torch_threads}",
        ]


def collect_environment(*, repository_root: Path | None = None) -> EnvironmentInfo:
    """Collect an :class:`EnvironmentInfo` snapshot for the current process.

    Args:
        repository_root: Optional directory used to look up the git commit.

    Returns:
        A populated environment snapshot.
    """
    import os

    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count() or 1,
        torch_version=str(torch.__version__),
        torch_threads=int(torch.get_num_threads()),
        accelerator=_accelerator_info(),
        package_version=_package_version(),
        git_commit=_git_commit(repository_root),
        environment_variables={
            name: os.environ[name] for name in _CAPTURED_ENV_VARS if name in os.environ
        },
    )


__all__ = ["EnvironmentInfo", "collect_environment"]
