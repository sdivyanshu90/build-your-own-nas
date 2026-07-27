"""JSON helpers with canonical encoding and bounded, validated reads.

Two concerns are handled here.

**Canonicalisation.** Architecture identity depends on a byte-exact encoding. The
canonical form sorts object keys, uses compact separators, forbids NaN/Infinity
(which are not valid JSON), and disables non-ASCII escaping so the output is pure
ASCII and therefore safe in any transport or database column.

**Safety.** ``read_json`` treats files as untrusted input: it refuses to read
oversized files rather than exhausting memory, and it never uses ``pickle``,
``eval``, or YAML tag resolution. See ``docs/architecture/security.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nas_engine.exceptions import NasEngineError

#: Default cap on the size of a JSON document accepted by :func:`read_json` (16 MiB).
DEFAULT_MAX_JSON_BYTES: int = 16 * 1024 * 1024


def canonical_json_dumps(value: Any) -> str:
    """Render ``value`` as canonical JSON text.

    Canonical means: keys sorted lexicographically, no insignificant whitespace,
    ASCII-only output, and no ``NaN``/``Infinity`` literals.

    Args:
        value: Any JSON-serialisable Python object.

    Returns:
        A canonical JSON string.

    Raises:
        NasEngineError: If the value contains non-finite floats or is not serialisable.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        msg = f"value is not canonically JSON-serialisable: {exc}"
        raise NasEngineError(msg, details={"error": str(exc)}) from exc


def write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    """Write ``value`` to ``path`` as UTF-8 JSON, creating parent directories.

    The write is atomic: content goes to a sibling temporary file which is then
    renamed. On POSIX filesystems ``os.replace`` is atomic, so a crash mid-write
    leaves the previous file intact rather than a truncated document.

    Args:
        path: Destination file path.
        value: JSON-serialisable object.
        indent: Indentation for human-readable output, or ``None`` for compact output.

    Raises:
        NasEngineError: If the value cannot be serialised.
    """
    try:
        text = json.dumps(value, indent=indent, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = f"cannot serialise value for {path}: {exc}"
        raise NasEngineError(msg, details={"path": str(path)}) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_bytes(payload: bytes, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES) -> Any:
    """Parse JSON from bytes with an explicit size cap.

    Args:
        payload: Raw UTF-8 encoded JSON.
        max_bytes: Maximum accepted payload size.

    Returns:
        The parsed Python object.

    Raises:
        NasEngineError: If the payload is too large or is not valid JSON.
    """
    if len(payload) > max_bytes:
        msg = (
            f"JSON payload of {len(payload)} bytes exceeds the limit of {max_bytes} bytes; "
            "raise the limit explicitly if this document is trusted"
        )
        raise NasEngineError(msg, details={"size": len(payload), "limit": max_bytes})
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"payload is not valid UTF-8 JSON: {exc}"
        raise NasEngineError(msg, details={"error": str(exc)}) from exc


def read_json(path: Path, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES) -> Any:
    """Read and parse a JSON file, refusing oversized documents.

    Args:
        path: File to read.
        max_bytes: Maximum accepted file size in bytes.

    Returns:
        The parsed Python object.

    Raises:
        NasEngineError: If the file is missing, oversized, or not valid JSON.
    """
    if not path.is_file():
        msg = f"JSON file not found: {path}"
        raise NasEngineError(msg, details={"path": str(path)})
    size = path.stat().st_size
    if size > max_bytes:
        msg = (
            f"JSON file {path} is {size} bytes which exceeds the limit of {max_bytes} bytes; "
            "this guard prevents accidental memory exhaustion from untrusted input"
        )
        raise NasEngineError(msg, details={"path": str(path), "size": size, "limit": max_bytes})
    return read_json_bytes(path.read_bytes(), max_bytes=max_bytes)


__all__ = [
    "DEFAULT_MAX_JSON_BYTES",
    "canonical_json_dumps",
    "read_json",
    "read_json_bytes",
    "write_json",
]
