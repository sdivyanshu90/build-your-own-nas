"""CSV and JSON exports.

Two formats because they answer different questions. CSV is what a spreadsheet, pandas, or
R reads without ceremony, and it is the "table view" that makes every figure in the report
verifiable — every point on every chart appears as a row. JSON preserves nesting (the full
architecture specification, the failure records, the training history) that a flat table
cannot express.

Determinism
-----------
Column order is fixed and rows are sorted by rank, so regenerating an export produces
byte-identical output for unchanged data. That makes exports diffable, which turns "did the
results change?" into a ``git diff``.

Safety
------
Every field that a spreadsheet might interpret as a formula is neutralised. A value
beginning with ``=``, ``+``, ``-``, or ``@`` is prefixed with an apostrophe: opening an
untrusted CSV in Excel otherwise executes the cell, a well-known injection vector. Metric
values here are numbers, but architecture hashes and error messages flow through the same
writer, so the guard is applied uniformly rather than selectively.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from nas_engine.exceptions import ReportingError
from nas_engine.objectives.ranking import RankedCandidate
from nas_engine.utilities.json_io import write_json

#: Columns always present in a candidate export, in this order.
BASE_COLUMNS: tuple[str, ...] = (
    "rank",
    "candidate_id",
    "architecture_hash",
    "score",
    "pareto_rank",
    "on_pareto_front",
    "feasible",
    "violations",
)

#: Characters that make a spreadsheet treat a cell as a formula.
_FORMULA_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(value: Any) -> Any:
    """Neutralise a value that a spreadsheet would interpret as a formula.

    Numbers pass through untouched — a negative number is not an injection risk because
    the guard only applies to text.

    Args:
        value: Cell value.

    Returns:
        The value, with a leading apostrophe added when it is dangerous text.
    """
    if not isinstance(value, str):
        return value
    if value.startswith(_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def collect_metric_columns(candidates: Sequence[RankedCandidate]) -> list[str]:
    """Return the sorted union of metric names across candidates.

    Sorted, not insertion-ordered: two runs that measured the same metrics must produce the
    same column order regardless of which candidate finished first.

    Args:
        candidates: Ranked candidates.

    Returns:
        Metric names in a stable order.
    """
    names: set[str] = set()
    for candidate in candidates:
        names.update(candidate.metrics)
    return sorted(names)


def export_candidates_csv(candidates: Sequence[RankedCandidate], path: Path) -> Path:
    """Write one row per candidate to a CSV file.

    Args:
        candidates: Ranked candidates, written in rank order.
        path: Destination file.

    Returns:
        The resolved path.

    Raises:
        ReportingError: If the file cannot be written.
    """
    metric_columns = collect_metric_columns(candidates)
    columns = [*BASE_COLUMNS, *metric_columns]
    ordered = sorted(candidates, key=lambda candidate: candidate.rank)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # `newline=""` is required by the csv module: without it, Windows writes \r\r\n.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for candidate in ordered:
                row: dict[str, Any] = {
                    "rank": candidate.rank,
                    "candidate_id": candidate.candidate_id,
                    "architecture_hash": candidate.architecture_hash,
                    "score": candidate.score if candidate.score is not None else "",
                    "pareto_rank": candidate.pareto_rank,
                    "on_pareto_front": int(candidate.on_pareto_front),
                    "feasible": int(candidate.feasible),
                    "violations": "; ".join(candidate.violations),
                }
                for name in metric_columns:
                    row[name] = candidate.metrics.get(name, "")
                writer.writerow({key: sanitize_cell(value) for key, value in row.items()})
    except OSError as exc:
        msg = f"could not write CSV export to {path}: {exc}"
        raise ReportingError(msg, details={"path": str(path), "error": str(exc)}) from exc
    return path.resolve()


def export_rows_csv(rows: Iterable[dict[str, Any]], columns: Sequence[str], path: Path) -> Path:
    """Write arbitrary rows to a CSV file with a fixed column order.

    Args:
        rows: Row mappings.
        columns: Column order.
        path: Destination file.

    Returns:
        The resolved path.

    Raises:
        ReportingError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: sanitize_cell(value) for key, value in row.items()})
    except OSError as exc:
        msg = f"could not write CSV export to {path}: {exc}"
        raise ReportingError(msg, details={"path": str(path), "error": str(exc)}) from exc
    return path.resolve()


def export_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a JSON export.

    Args:
        payload: Data to write.
        path: Destination file.

    Returns:
        The resolved path.

    Raises:
        ReportingError: If the payload cannot be serialised or written.
    """
    try:
        write_json(path, payload)
    except Exception as exc:
        msg = f"could not write JSON export to {path}: {exc}"
        raise ReportingError(msg, details={"path": str(path), "error": str(exc)}) from exc
    return path.resolve()


__all__ = [
    "BASE_COLUMNS",
    "collect_metric_columns",
    "export_candidates_csv",
    "export_json",
    "export_rows_csv",
    "sanitize_cell",
]
