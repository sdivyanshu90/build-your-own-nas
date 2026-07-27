"""Regression tests protecting the structured-event vocabulary.

Event names are a public interface: dashboards, log queries, and alerts are built on them.
Two properties have to hold, and neither is enforced by the type system.

**One emitter per name.** :class:`~nas_engine.observability.events.Event` is a closed
enumeration precisely so that a name means exactly one thing. Nothing stops a module from
calling ``_LOGGER.info("evaluation.completed", ...)`` with a raw string, and once it does,
every attempt is logged twice — under one name, with two different field sets. Anything
counting evaluations then reports double, and the discrepancy is invisible until someone
compares a dashboard against the database.

That is not hypothetical: the evaluator did exactly this, which is why its inner lines now
live in an ``evaluator.*`` namespace and this test exists.

**One name per quantity.** A duration is ``duration_seconds`` everywhere. When the same
value appears as ``duration`` in one event and ``duration_seconds`` in another, every
consumer needs a special case, and the one that forgets silently reads ``null``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nas_engine.observability.events import Event

pytestmark = pytest.mark.regression

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "nas_engine"

#: Logger method names whose first positional argument is the event name.
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

#: Field names that must be spelled consistently across every event.
_CANONICAL_FIELD_NAMES = {
    "duration": "duration_seconds",
    "elapsed": "elapsed_seconds",
    "seconds": "duration_seconds",
}


def _iter_source_files() -> list[Path]:
    """Return every Python module in the package."""
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _iter_logger_event_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Yield ``(event_name, line_number)`` for each raw-string logger call.

    Args:
        tree: Parsed module.

    Returns:
        The literal event name and line of every ``<logger>.<level>("name", ...)`` call.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _LOG_METHODS or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((first.value, first.lineno))
    return found


def _iter_emit_keywords(tree: ast.AST) -> list[tuple[str, int]]:
    """Yield ``(keyword_name, line_number)`` for each ``emit(...)`` keyword argument.

    Args:
        tree: Parsed module.

    Returns:
        Every keyword passed to a call named ``emit``.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "emit"):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None:
                found.append((keyword.arg, node.lineno))
    return found


class TestEventNamesAreNotShadowed:
    def test_no_module_logs_a_raw_string_owned_by_the_event_enum(self) -> None:
        reserved = {event.value for event in Event}
        offenders: list[str] = []
        for path in _iter_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(SOURCE_ROOT)}:{line} logs {name!r}"
                for name, line in _iter_logger_event_names(tree)
                if name in reserved
            )

        assert not offenders, (
            "these call sites log a raw string that the Event enum already owns, so the "
            "event would be emitted twice per occurrence with two different field sets; "
            "use emit(Event.X, ...) or pick a module-scoped name:\n  " + "\n  ".join(offenders)
        )

    def test_module_scoped_log_names_are_namespaced(self) -> None:
        # A dotted name keeps module diagnostics ("evaluator.completed") distinguishable
        # from the Event vocabulary while staying greppable.
        undotted: list[str] = []
        for path in _iter_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            undotted.extend(
                f"{path.relative_to(SOURCE_ROOT)}:{line} logs {name!r}"
                for name, line in _iter_logger_event_names(tree)
                if "." not in name
            )
        assert not undotted, (
            "log event names must be dotted, like 'evaluator.completed':\n  "
            + "\n  ".join(undotted)
        )


class TestEventFieldNamesAreConsistent:
    def test_no_emit_uses_a_non_canonical_field_name(self) -> None:
        offenders: list[str] = []
        for path in _iter_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(SOURCE_ROOT)}:{line} passes {name!r}, "
                f"expected {_CANONICAL_FIELD_NAMES[name]!r}"
                for name, line in _iter_emit_keywords(tree)
                if name in _CANONICAL_FIELD_NAMES
            )
        assert not offenders, (
            "the same quantity must have the same field name in every event:\n  "
            + "\n  ".join(offenders)
        )


class TestTheGuardItselfWorks:
    def test_a_shadowing_call_would_be_detected(self) -> None:
        # Without this, a bug in the AST walk would make the guard silently vacuous.
        source = '_LOGGER.info("evaluation.completed", duration=1.0)\n'
        names = _iter_logger_event_names(ast.parse(source))
        assert names == [("evaluation.completed", 1)]
        assert "evaluation.completed" in {event.value for event in Event}

    def test_a_non_canonical_field_would_be_detected(self) -> None:
        source = "emit(Event.SEARCH_COMPLETED, duration=1.0)\n"
        keywords = _iter_emit_keywords(ast.parse(source))
        assert ("duration", 1) in keywords

    def test_the_scan_actually_reads_the_package(self) -> None:
        files = _iter_source_files()
        assert len(files) > 50, f"expected the whole package, found {len(files)} files"
