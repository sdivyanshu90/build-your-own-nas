"""Filesystem path validation.

The engine writes artifacts whose names are partially derived from user-supplied
configuration and from architecture hashes. Without validation a crafted value such
as ``../../etc/cron.d/payload`` could escape the artifact root — a classic path
traversal. Every write in this project routes through :func:`resolve_under_root`.

Trust boundary: configuration files and imported architecture JSON are treated as
untrusted input. See ``docs/architecture/security.md``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nas_engine.exceptions import UnsafePathError

#: Characters permitted in generated filenames. Everything else is replaced.
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

#: Names that are reserved on Windows and therefore avoided even on POSIX hosts.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

#: Upper bound on generated filename length, comfortably under the 255-byte limit
#: enforced by ext4, APFS, and NTFS.
_MAX_FILENAME_LENGTH = 120


def safe_filename(name: str, *, fallback: str = "unnamed") -> str:
    """Reduce arbitrary text to a conservative, portable filename component.

    Path separators, control characters, and shell metacharacters are replaced with
    underscores; leading dots are stripped so the result is never a hidden file or a
    relative traversal component.

    Args:
        name: Arbitrary text, possibly hostile.
        fallback: Value returned when ``name`` reduces to the empty string.

    Returns:
        A filename component containing only ``[A-Za-z0-9._-]``.
    """
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", name.strip())
    cleaned = cleaned.strip("._")
    if len(cleaned) > _MAX_FILENAME_LENGTH:
        cleaned = cleaned[:_MAX_FILENAME_LENGTH].rstrip("._")
    if not cleaned:
        return fallback
    if cleaned.lower() in _RESERVED_NAMES:
        return f"{cleaned}_"
    return cleaned


def is_within(candidate: Path, root: Path) -> bool:
    """Report whether ``candidate`` resolves to a location inside ``root``.

    Both paths are fully resolved first so that symlinks and ``..`` components cannot
    be used to escape. ``root`` itself counts as inside.

    Args:
        candidate: Path to test.
        root: Directory that must contain the candidate.

    Returns:
        ``True`` when the resolved candidate is ``root`` or a descendant of it.
    """
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate == resolved_root:
        return True
    return resolved_root in resolved_candidate.parents


def resolve_under_root(root: Path, *parts: str | os.PathLike[str]) -> Path:
    """Join ``parts`` under ``root`` and verify the result cannot escape it.

    Absolute components are rejected outright rather than silently discarding the
    root, because ``Path("/a") / "/etc/passwd"`` evaluates to ``/etc/passwd`` in
    Python and that behaviour has produced real vulnerabilities.

    Args:
        root: Directory that must contain the result.
        *parts: Relative path components.

    Returns:
        The resolved path inside ``root``.

    Raises:
        UnsafePathError: If a component is absolute or the result escapes ``root``.
    """
    for part in parts:
        text = os.fspath(part)
        if Path(text).is_absolute():
            msg = (
                f"absolute path component {text!r} is not allowed when resolving under "
                f"{root}; supply a relative component instead"
            )
            raise UnsafePathError(msg, details={"root": str(root), "component": text})

    candidate = root.joinpath(*[os.fspath(part) for part in parts])
    if not is_within(candidate, root):
        msg = (
            f"resolved path {candidate} escapes its permitted root {root}; "
            "path traversal components such as '..' are rejected"
        )
        raise UnsafePathError(msg, details={"root": str(root), "candidate": str(candidate)})
    return candidate.resolve()


def ensure_directory(path: Path, *, writable: bool = True) -> Path:
    """Create ``path`` (including parents) and verify the process can use it.

    Args:
        path: Directory to create.
        writable: When ``True``, verify the directory is writable by this process.

    Returns:
        The resolved directory path.

    Raises:
        UnsafePathError: If the path exists as a non-directory, cannot be created,
            or is not writable when required.
    """
    if path.exists() and not path.is_dir():
        msg = f"expected {path} to be a directory but it exists as a file"
        raise UnsafePathError(msg, details={"path": str(path)})
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create directory {path}: {exc}"
        raise UnsafePathError(msg, details={"path": str(path), "error": str(exc)}) from exc
    if writable and not os.access(path, os.W_OK):
        msg = (
            f"directory {path} is not writable by the current user; "
            "adjust permissions or choose a different output directory"
        )
        raise UnsafePathError(msg, details={"path": str(path)})
    return path.resolve()


__all__ = ["ensure_directory", "is_within", "resolve_under_root", "safe_filename"]
