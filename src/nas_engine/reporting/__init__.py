"""Reports, exports, and figures generated from persisted search results.

This package reads only through :class:`~nas_engine.persistence.repository.SearchRepository`,
so a report can be produced from a database file long after the run that created it, on a
different machine, without the engine.
"""

from nas_engine.reporting.exporters import (
    export_candidates_csv,
    export_json,
    export_rows_csv,
    sanitize_cell,
)
from nas_engine.reporting.plots import PlotResult, generate_plots
from nas_engine.reporting.report import (
    KNOWN_LIMITATIONS,
    ReportArtifacts,
    ReportGenerator,
)

__all__ = [
    "KNOWN_LIMITATIONS",
    "PlotResult",
    "ReportArtifacts",
    "ReportGenerator",
    "export_candidates_csv",
    "export_json",
    "export_rows_csv",
    "generate_plots",
    "sanitize_cell",
]
