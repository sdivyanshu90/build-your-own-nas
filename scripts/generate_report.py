"""Generate a report for an existing search without re-running it.

Reports are built from the database, so this works long after the search finished, on a
different machine, from a copied ``nas.db``. Use it to regenerate a report after changing
the objective weighting, or to produce one for a search that was interrupted.

Usage::

    python scripts/generate_report.py --config configs/random_search.yaml
    python scripts/generate_report.py --config configs/evolution.yaml --search-id 3f2a...
    python scripts/generate_report.py --config configs/evolution.yaml --all --no-plots

The equivalent CLI command is ``nas-engine report``; this script exists for the ``--all``
case and as a worked example of using the reporting API directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nas_engine.config.loader import load_config
from nas_engine.exceptions import NasEngineError
from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import ensure_schema
from nas_engine.persistence.repository import SearchRepository
from nas_engine.reporting.report import ReportGenerator


def main() -> int:
    """Parse arguments and generate the requested reports.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a handled failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--search-id", default=None, help="search to report on")
    parser.add_argument("--all", action="store_true", help="report on every search in the database")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument("--output", type=Path, default=None, help="override the report directory")
    arguments = parser.parse_args()

    try:
        config = load_config(arguments.config)
    except NasEngineError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    database = Database(config.database_url)
    try:
        ensure_schema(database)
        repository = SearchRepository(database)

        if arguments.all:
            targets = [summary.id for summary in repository.list_searches(limit=100)]
        elif arguments.search_id:
            targets = [arguments.search_id]
        else:
            latest = repository.find_latest_search(name=config.project.name)
            if latest is None:
                latest = repository.find_latest_search()
            if latest is None:
                print(
                    f"no search found in {config.database_url}; run 'nas-engine search' first",
                    file=sys.stderr,
                )
                return 1
            targets = [latest.id]

        generator = ReportGenerator(
            repository,
            objectives=config.objectives.build_objectives(),
            constraints=config.objectives.build_constraints(),
            output_dir=arguments.output or config.report_dir,
            artifact_root=config.artifact_dir,
        )

        for search_id in targets:
            try:
                artifacts = generator.generate(search_id, include_plots=not arguments.no_plots)
            except NasEngineError as error:
                print(f"{search_id}: {error}", file=sys.stderr)
                continue
            print(f"{search_id}:")
            print(f"  markdown : {artifacts.markdown}")
            print(f"  json     : {artifacts.json}")
            print(f"  csv      : {artifacts.csv}")
            for name, path in sorted(artifacts.plots.items()):
                print(f"  plot     : {name} -> {path}")
            for name, reason in sorted(artifacts.skipped_plots.items()):
                print(f"  skipped  : {name} ({reason})")
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
