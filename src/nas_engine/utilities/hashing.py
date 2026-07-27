"""Stable content hashing.

Why not :func:`hash`?
---------------------
CPython's built-in ``hash`` for :class:`str` and :class:`bytes` is randomised per
process by ``PYTHONHASHSEED``. Architecture identity must survive process restarts,
database round-trips, and machine changes, so every identity hash in this project
uses BLAKE2b over canonical UTF-8 bytes.

Why BLAKE2b?
------------
* Available in the standard library (:mod:`hashlib`), so there is no extra dependency.
* Supports a configurable digest size, which lets us produce short but collision-safe
  identifiers.
* Cryptographic-strength diffusion means a single changed field produces a completely
  different digest, which makes accidental collisions between similar architectures
  effectively impossible.

Digest length
-------------
The default is 16 bytes (32 hex characters, 128 bits). With ``N`` distinct
architectures the birthday-collision probability is approximately ``N^2 / 2^129``.
Even at ``N = 10^9`` distinct candidates that is below ``1.5e-21`` — far smaller than
the probability of silent hardware corruption, so hash equality is treated as
architecture equality throughout the codebase.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Default digest size in bytes for architecture and configuration hashes.
DEFAULT_DIGEST_BYTES: int = 16


def stable_hash_bytes(payload: bytes, *, digest_bytes: int = DEFAULT_DIGEST_BYTES) -> str:
    """Hash raw bytes with BLAKE2b and return a lowercase hex digest.

    Args:
        payload: Bytes to hash.
        digest_bytes: Digest size in bytes; must be between 1 and 64 inclusive.

    Returns:
        Lowercase hexadecimal digest of length ``2 * digest_bytes``.

    Raises:
        ValueError: If ``digest_bytes`` is outside the range accepted by BLAKE2b.
    """
    if not 1 <= digest_bytes <= 64:
        msg = f"digest_bytes must be in [1, 64], received {digest_bytes}"
        raise ValueError(msg)
    return hashlib.blake2b(payload, digest_size=digest_bytes).hexdigest()


def stable_hash(text: str, *, digest_bytes: int = DEFAULT_DIGEST_BYTES) -> str:
    """Hash a string by encoding it as UTF-8 first.

    Args:
        text: Text to hash.
        digest_bytes: Digest size in bytes.

    Returns:
        Lowercase hexadecimal digest.
    """
    return stable_hash_bytes(text.encode("utf-8"), digest_bytes=digest_bytes)


def stable_json_hash(value: Any, *, digest_bytes: int = DEFAULT_DIGEST_BYTES) -> str:
    """Hash any JSON-serialisable value using its canonical encoding.

    The value is first rendered with :func:`nas_engine.utilities.json_io.canonical_json_dumps`,
    which sorts mapping keys and removes insignificant whitespace. Two values that are
    equal as JSON documents therefore always produce the same digest, regardless of the
    insertion order of their keys.

    Args:
        value: Any JSON-serialisable Python object.
        digest_bytes: Digest size in bytes.

    Returns:
        Lowercase hexadecimal digest.
    """
    from nas_engine.utilities.json_io import canonical_json_dumps

    return stable_hash(canonical_json_dumps(value), digest_bytes=digest_bytes)


__all__ = ["DEFAULT_DIGEST_BYTES", "stable_hash", "stable_hash_bytes", "stable_json_hash"]
