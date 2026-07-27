"""Search checkpoints: what has to be saved for a resume to be correct.

A resume that merely reloads the database is not enough. The database records what *has*
happened; the strategy's internal state determines what happens *next*. Restoring only the
first would restart the strategy from scratch: random search would replay its first
proposals, evolution would rebuild an empty population, and successive halving would forget
which rung it was on.

Contents
--------
``strategy_state``
    Everything :meth:`~nas_engine.search.strategy.SearchStrategy.state_dict` produces,
    including the exact position of every random generator.
``engine_state``
    Counters the engine owns: proposals made, evaluations completed, candidates pruned,
    elapsed time. These drive the stopping conditions and are not derivable from the
    strategy.
``config_hash``
    Detects that the configuration changed between the checkpoint and the resume.

Not stored
----------
Model weights and dataset contents. Weights live in artifact files referenced by the
database; the dataset is rebuilt deterministically from its seed. Putting either in a
checkpoint would make it enormous for no benefit.

Versioning and validation
-------------------------
Every checkpoint carries a format version and is validated on load. A checkpoint from a
newer format is rejected outright; a malformed one produces a clear error naming the
missing field. A resume that *looks* successful but silently drops half the state is the
worst possible outcome, so the loader is strict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nas_engine.exceptions import CheckpointError, CheckpointVersionError
from nas_engine.utilities.timing import utc_now, utc_now_iso

#: Version of the search-checkpoint payload format.
SEARCH_CHECKPOINT_FORMAT_VERSION: int = 1


@dataclass
class EngineState:
    """Counters the engine maintains across the run.

    Attributes:
        proposed: Proposals received from the strategy.
        accepted: Proposals that became persisted candidates.
        duplicates: Proposals rejected because their identity already existed.
        invalid: Proposals rejected by validation.
        pruned: Candidates rejected by a resource constraint.
        completed: Evaluations that succeeded.
        failed: Candidates that failed permanently.
        retried: Retry attempts scheduled.
        elapsed_seconds: Accumulated wall-clock time across all run segments.
    """

    proposed: int = 0
    accepted: int = 0
    duplicates: int = 0
    invalid: int = 0
    pruned: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "invalid": self.invalid,
            "pruned": self.pruned,
            "completed": self.completed,
            "failed": self.failed,
            "retried": self.retried,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EngineState:
        """Rebuild engine counters from :meth:`to_dict` output.

        Args:
            payload: Serialised counters.

        Returns:
            The reconstructed state.
        """
        return cls(
            proposed=int(payload.get("proposed", 0)),
            accepted=int(payload.get("accepted", 0)),
            duplicates=int(payload.get("duplicates", 0)),
            invalid=int(payload.get("invalid", 0)),
            pruned=int(payload.get("pruned", 0)),
            completed=int(payload.get("completed", 0)),
            failed=int(payload.get("failed", 0)),
            retried=int(payload.get("retried", 0)),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        )


@dataclass
class SearchCheckpoint:
    """A complete snapshot of a search's resumable state.

    Attributes:
        search_id: The search this checkpoint belongs to.
        strategy_name: Strategy that produced ``strategy_state``.
        strategy_state: The strategy's serialised state.
        engine_state: The engine's counters.
        config_hash: Configuration hash at checkpoint time.
        created_at: ISO-8601 creation timestamp.
        format_version: Payload format version.
    """

    search_id: str
    strategy_name: str
    strategy_state: dict[str, Any]
    engine_state: EngineState = field(default_factory=EngineState)
    config_hash: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    format_version: int = SEARCH_CHECKPOINT_FORMAT_VERSION

    def to_payload(self) -> dict[str, Any]:
        """Return the plain dictionary persisted in the checkpoints table."""
        return {
            "format_version": self.format_version,
            "search_id": self.search_id,
            "strategy_name": self.strategy_name,
            "strategy_state": self.strategy_state,
            "engine_state": self.engine_state.to_dict(),
            "config_hash": self.config_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SearchCheckpoint:
        """Rebuild a checkpoint from a persisted payload.

        Args:
            payload: Stored checkpoint data.

        Returns:
            The reconstructed checkpoint.

        Raises:
            CheckpointVersionError: If the format version is newer than supported.
            CheckpointError: If required fields are missing or the payload is not a mapping.
        """
        if not isinstance(payload, dict):
            msg = (  # type: ignore[unreachable]
                "search checkpoint payload is not a mapping "
                f"(found {type(payload).__name__}); the record is corrupt"
            )
            raise CheckpointError(msg)

        version = payload.get("format_version")
        if not isinstance(version, int):
            msg = (
                "search checkpoint is missing an integer 'format_version'; the record was "
                "not written by nas-engine or is corrupt"
            )
            raise CheckpointError(msg, details={"format_version": repr(version)})
        if version > SEARCH_CHECKPOINT_FORMAT_VERSION:
            msg = (
                f"search checkpoint format version {version} is newer than the supported "
                f"version {SEARCH_CHECKPOINT_FORMAT_VERSION}; upgrade nas-engine to resume "
                "this search"
            )
            raise CheckpointVersionError(
                msg,
                details={"found": version, "supported": SEARCH_CHECKPOINT_FORMAT_VERSION},
            )

        missing = [
            key for key in ("search_id", "strategy_name", "strategy_state") if key not in payload
        ]
        if missing:
            msg = f"search checkpoint is missing required fields {missing}"
            raise CheckpointError(msg, details={"missing": missing})

        state = payload["strategy_state"]
        if not isinstance(state, dict):
            msg = (
                "search checkpoint 'strategy_state' must be a mapping, found "
                f"{type(state).__name__}"
            )
            raise CheckpointError(msg)

        return cls(
            search_id=str(payload["search_id"]),
            strategy_name=str(payload["strategy_name"]),
            strategy_state=dict(state),
            engine_state=EngineState.from_dict(payload.get("engine_state", {})),
            config_hash=str(payload.get("config_hash", "")),
            created_at=str(payload.get("created_at", utc_now().isoformat())),
            format_version=version,
        )

    def validate_for(self, *, strategy_name: str, config_hash: str | None = None) -> list[str]:
        """Check a loaded checkpoint against the current run, returning warnings.

        A strategy mismatch is fatal — evolution state cannot be loaded into successive
        halving. A configuration mismatch is a warning, because changing the log level or
        the device between segments is legitimate; the caller decides how strict to be.

        Args:
            strategy_name: Strategy the resume will use.
            config_hash: Configuration hash the resume will use.

        Returns:
            Human-readable warnings; empty when the checkpoint matches exactly.

        Raises:
            CheckpointError: If the checkpoint belongs to a different strategy.
        """
        if self.strategy_name != strategy_name:
            msg = (
                f"this checkpoint was written by the '{self.strategy_name}' strategy but "
                f"the resume is configured to use '{strategy_name}'. Strategy state is not "
                "interchangeable; either restore the original algorithm setting or start a "
                "new search."
            )
            raise CheckpointError(
                msg, details={"checkpoint": self.strategy_name, "configured": strategy_name}
            )
        warnings: list[str] = []
        if config_hash is not None and self.config_hash and self.config_hash != config_hash:
            warnings.append(
                f"the configuration changed since this checkpoint was written "
                f"({self.config_hash} -> {config_hash}); results from before and after the "
                "resume may not be comparable"
            )
        return warnings


__all__ = [
    "SEARCH_CHECKPOINT_FORMAT_VERSION",
    "EngineState",
    "SearchCheckpoint",
]
