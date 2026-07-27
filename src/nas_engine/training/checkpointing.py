"""Training checkpoints: save, load, and version.

What a checkpoint must contain
------------------------------
Restoring weights alone is not enough to resume training correctly. The optimiser's
momentum buffers, the scheduler's step counter, and the early-stopping counters are all
part of the training trajectory; dropping any of them makes the resumed run different
from the uninterrupted one. Everything needed to continue is stored together.

Versioning
----------
Every checkpoint carries ``format_version``. Loading a checkpoint from a newer format
fails loudly rather than silently mis-reading fields — a corrupted resume that *looks*
like it worked is far worse than one that refuses to start.

Safety
------
Checkpoints are loaded with ``weights_only=True``. Without it, :func:`torch.load` uses
``pickle``, which executes arbitrary code during deserialisation; a checkpoint file from
an untrusted source is then a remote code execution vector. ``weights_only=True``
restricts the loader to tensors and plain data, which is all this format contains. See
``docs/architecture/security.md``.

Atomicity
---------
Writes go to a temporary sibling file and are then renamed. ``os.replace`` is atomic on
POSIX, so a crash during a checkpoint write leaves the previous checkpoint intact instead
of a truncated file that fails to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from nas_engine.exceptions import CheckpointError, CheckpointVersionError

#: Version of the training-checkpoint payload format.
CHECKPOINT_FORMAT_VERSION: int = 1


@dataclass
class TrainingCheckpoint:
    """Everything needed to resume a candidate's training exactly.

    Attributes:
        architecture_hash: Hash of the architecture the weights belong to. Loading a
            checkpoint into a different architecture is caught by this field long before
            a shape mismatch surfaces.
        epoch: Number of completed epochs.
        global_step: Number of completed optimiser steps.
        model_state: Model ``state_dict``.
        optimizer_state: Optimiser ``state_dict``.
        scheduler_state: Scheduler ``state_dict``.
        early_stopping_state: Early-stopping counters.
        history: Per-epoch metric dictionaries recorded so far.
        format_version: Payload format version.
    """

    architecture_hash: str
    epoch: int
    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    early_stopping_state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def to_payload(self) -> dict[str, Any]:
        """Return the plain dictionary written to disk."""
        return {
            "format_version": self.format_version,
            "architecture_hash": self.architecture_hash,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "early_stopping_state": self.early_stopping_state,
            "history": self.history,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TrainingCheckpoint:
        """Rebuild a checkpoint from a loaded payload.

        Args:
            payload: Dictionary read from disk.

        Returns:
            The reconstructed checkpoint.

        Raises:
            CheckpointVersionError: If the payload format is newer than supported.
            CheckpointError: If required fields are missing.
        """
        version = payload.get("format_version")
        if not isinstance(version, int):
            msg = (
                "checkpoint is missing an integer 'format_version' field; the file is "
                "corrupt or was not produced by nas-engine"
            )
            raise CheckpointError(msg, details={"format_version": repr(version)})
        if version > CHECKPOINT_FORMAT_VERSION:
            msg = (
                f"checkpoint format version {version} is newer than the supported "
                f"version {CHECKPOINT_FORMAT_VERSION}; upgrade nas-engine to read it"
            )
            raise CheckpointVersionError(
                msg, details={"found": version, "supported": CHECKPOINT_FORMAT_VERSION}
            )

        required = ("architecture_hash", "epoch", "model_state")
        missing = [key for key in required if key not in payload]
        if missing:
            msg = f"checkpoint is missing required fields {missing}"
            raise CheckpointError(msg, details={"missing": missing})

        return cls(
            architecture_hash=str(payload["architecture_hash"]),
            epoch=int(payload["epoch"]),
            global_step=int(payload.get("global_step", 0)),
            model_state=dict(payload["model_state"]),
            optimizer_state=dict(payload.get("optimizer_state", {})),
            scheduler_state=dict(payload.get("scheduler_state", {})),
            early_stopping_state=dict(payload.get("early_stopping_state", {})),
            history=list(payload.get("history", [])),
            format_version=version,
        )


def save_checkpoint(path: Path, checkpoint: TrainingCheckpoint) -> Path:
    """Atomically write a checkpoint to ``path``.

    Args:
        path: Destination file.
        checkpoint: Checkpoint to write.

    Returns:
        The resolved destination path.

    Raises:
        CheckpointError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(checkpoint.to_payload(), temporary)
        temporary.replace(path)
    except OSError as exc:
        msg = f"failed to write checkpoint to {path}: {exc}"
        raise CheckpointError(msg, details={"path": str(path), "error": str(exc)}) from exc
    return path.resolve()


def load_checkpoint(path: Path, *, expected_hash: str | None = None) -> TrainingCheckpoint:
    """Load and validate a checkpoint.

    Args:
        path: File to read.
        expected_hash: Architecture hash the checkpoint must belong to. When supplied and
            mismatched, the load fails rather than producing confusing shape errors later.

    Returns:
        The loaded checkpoint.

    Raises:
        CheckpointError: If the file is missing, unreadable, corrupt, or belongs to a
            different architecture.
        CheckpointVersionError: If the format version is unsupported.
    """
    if not path.is_file():
        msg = f"checkpoint file not found: {path}"
        raise CheckpointError(msg, details={"path": str(path)})
    try:
        # `weights_only=True` refuses to execute pickled code during load. Checkpoints
        # contain only tensors and plain data, so nothing is lost by the restriction.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        msg = (
            f"checkpoint at {path} could not be read ({type(exc).__name__}: {exc}); the "
            "file is corrupt or truncated. Delete it to restart this candidate from "
            "scratch."
        )
        raise CheckpointError(msg, details={"path": str(path), "error": str(exc)}) from exc

    if not isinstance(payload, dict):
        msg = (
            f"checkpoint at {path} does not contain a dictionary payload (found "
            f"{type(payload).__name__}); the file is corrupt"
        )
        raise CheckpointError(msg, details={"path": str(path)})

    checkpoint = TrainingCheckpoint.from_payload(payload)
    if expected_hash is not None and checkpoint.architecture_hash != expected_hash:
        msg = (
            f"checkpoint at {path} belongs to architecture "
            f"{checkpoint.architecture_hash} but {expected_hash} was expected; refusing "
            "to load weights into a different network"
        )
        raise CheckpointError(
            msg,
            details={
                "path": str(path),
                "found": checkpoint.architecture_hash,
                "expected": expected_hash,
            },
        )
    return checkpoint


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "TrainingCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
]
