"""Canonical serialisation of architecture genotypes.

"Canonical" means: two architecture specifications that describe the same network
produce byte-identical output, and different networks produce different output.
Everything downstream — hashing, duplicate detection, database identity, regression
fixtures — depends on that guarantee.

Three mechanisms combine to provide it:

1. **Field canonicalisation** (in :mod:`nas_engine.architectures.spec`): inactive
   conditional fields are forced to sentinel values at construction time, so the
   *values* being serialised are already normalised.
2. **Key ordering**: :func:`~nas_engine.utilities.json_io.canonical_json_dumps` sorts
   object keys, so declaration order in the model class cannot leak into the bytes.
3. **Numeric normalisation**: floats are quantised to a fixed precision, integers stay
   integers, and enums serialise to their string values, so no repr instability exists.

Round-tripping is total: ``from_canonical_json(to_canonical_json(spec)) == spec`` for
every valid specification. This is asserted by property tests.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.exceptions import ArchitectureValidationError
from nas_engine.utilities.json_io import canonical_json_dumps, read_json_bytes

#: Upper bound on the size of an imported architecture document. Architecture JSON is
#: at most a few kilobytes; anything larger is either corrupt or hostile.
MAX_ARCHITECTURE_JSON_BYTES: int = 1024 * 1024


def to_canonical_dict(spec: ArchitectureSpec) -> dict[str, Any]:
    """Return the canonical plain-data representation of ``spec``.

    Enums are rendered as their string values and tuples as lists, so the result is
    pure JSON data with no Python-specific types.

    Args:
        spec: Architecture to render.

    Returns:
        A nested dictionary of JSON-native values.
    """
    dumped: dict[str, Any] = spec.model_dump(mode="json")
    return dumped


def to_canonical_json(spec: ArchitectureSpec) -> str:
    """Return the canonical JSON text for ``spec``.

    Args:
        spec: Architecture to render.

    Returns:
        Canonical JSON: sorted keys, compact separators, ASCII only.
    """
    return canonical_json_dumps(to_canonical_dict(spec))


def from_canonical_dict(payload: Any) -> ArchitectureSpec:
    """Rebuild an :class:`ArchitectureSpec` from plain data.

    The payload is treated as **untrusted**: unknown fields are rejected, enum values
    outside the closed vocabulary are rejected, and numeric ranges are enforced by the
    model. No code from the payload is ever executed.

    Args:
        payload: Plain data previously produced by :func:`to_canonical_dict`, or any
            externally supplied mapping.

    Returns:
        A validated architecture.

    Raises:
        ArchitectureValidationError: If the payload is not a valid architecture.
    """
    if not isinstance(payload, dict):
        msg = f"architecture payload must be a JSON object, received {type(payload).__name__}"
        raise ArchitectureValidationError(msg, details={"received_type": type(payload).__name__})
    try:
        return ArchitectureSpec.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
                "received": repr(error.get("input")),
            }
            for error in exc.errors()[:10]
        ]
        summary = "; ".join(
            f"{item['field']}: {item['message']} (received {item['received']})" for item in errors
        )
        msg = f"architecture payload failed validation: {summary}"
        raise ArchitectureValidationError(msg, details={"errors": errors}) from exc


def from_canonical_json(text: str | bytes) -> ArchitectureSpec:
    """Parse canonical JSON text into an :class:`ArchitectureSpec`.

    Args:
        text: JSON document as text or UTF-8 bytes.

    Returns:
        A validated architecture.

    Raises:
        ArchitectureValidationError: If the document is malformed, oversized, or invalid.
    """
    payload = text.encode("utf-8") if isinstance(text, str) else text
    try:
        parsed = read_json_bytes(payload, max_bytes=MAX_ARCHITECTURE_JSON_BYTES)
    except Exception as exc:
        msg = f"architecture document could not be parsed: {exc}"
        raise ArchitectureValidationError(msg, details={"error": str(exc)}) from exc
    return from_canonical_dict(parsed)


def architectures_equal(left: ArchitectureSpec, right: ArchitectureSpec) -> bool:
    """Compare two architectures by canonical form rather than object identity.

    Pydantic's ``__eq__`` already compares field values, but comparing canonical JSON
    is the definition the rest of the system uses (hashes, database keys), so equality
    is expressed in those terms explicitly.

    Args:
        left: First architecture.
        right: Second architecture.

    Returns:
        ``True`` when the two describe the same network.
    """
    return to_canonical_json(left) == to_canonical_json(right)


__all__ = [
    "MAX_ARCHITECTURE_JSON_BYTES",
    "architectures_equal",
    "from_canonical_dict",
    "from_canonical_json",
    "to_canonical_dict",
    "to_canonical_json",
]
