"""Architecture hashing: stable identity for candidate networks.

The hash answers one question: *have we already evaluated this exact network?*
Training a duplicate costs a full evaluation budget and returns no information, so
duplicate detection is one of the highest-value features in a NAS engine.

Properties the hash must have
-----------------------------
* **Deterministic across processes and machines.** Multiprocessing workers, the CLI,
  and the database must all agree, so Python's randomised :func:`hash` is unusable.
* **Insensitive to irrelevant differences.** Key order, float repr, and inactive
  conditional fields must not change the digest. Canonicalisation guarantees this.
* **Sensitive to every relevant difference.** Changing one kernel size must change the
  digest. BLAKE2b's avalanche property guarantees this.
* **Short enough to read.** 32 hex characters fit in a terminal column and in a
  filename, while retaining 128 bits of collision resistance.

The hash is *not* a security primitive here: nobody is trying to forge a colliding
architecture. It is a content-addressed identifier.
"""

from __future__ import annotations

from nas_engine.architectures.canonical import to_canonical_json
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.utilities.hashing import stable_hash

#: Digest size in bytes. 16 bytes → 32 hex characters → 128 bits.
ARCHITECTURE_HASH_BYTES: int = 16

#: Length of the hex digest produced by :func:`architecture_hash`.
ARCHITECTURE_HASH_LENGTH: int = ARCHITECTURE_HASH_BYTES * 2


def architecture_hash(spec: ArchitectureSpec) -> str:
    """Return the canonical content hash of an architecture.

    Args:
        spec: Architecture to hash.

    Returns:
        A 32-character lowercase hexadecimal digest.

    Example:
        >>> from nas_engine.architectures.spec import (
        ...     ArchitectureSpec, BlockSpec, StageSpec)
        >>> from nas_engine.architectures.types import OperationType
        >>> spec = ArchitectureSpec(
        ...     stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV),)),))
        >>> len(architecture_hash(spec))
        32
    """
    return stable_hash(to_canonical_json(spec), digest_bytes=ARCHITECTURE_HASH_BYTES)


def short_hash(architecture_hash_value: str, *, length: int = 8) -> str:
    """Return a truncated hash suitable for human-facing tables and filenames.

    Truncated hashes are display-only. They are never used as database keys, because
    8 hex characters is only 32 bits and collides at around 77 000 candidates.

    Args:
        architecture_hash_value: A full architecture hash.
        length: Number of leading characters to keep.

    Returns:
        The truncated hash.

    Raises:
        ValueError: If ``length`` is not between 4 and the full hash length.
    """
    if not 4 <= length <= len(architecture_hash_value):
        msg = f"short hash length must be in [4, {len(architecture_hash_value)}], received {length}"
        raise ValueError(msg)
    return architecture_hash_value[:length]


__all__ = [
    "ARCHITECTURE_HASH_BYTES",
    "ARCHITECTURE_HASH_LENGTH",
    "architecture_hash",
    "short_hash",
]
