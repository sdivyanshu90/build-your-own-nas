"""The ``nas-engine`` command-line interface.

Design rules
------------
**Exit codes are meaningful.** Scripts and CI need to branch on the outcome, and "did it
print an error?" is not something a shell can test. See :class:`ExitCode`.

**Every command can emit JSON.** Human-readable tables are the default because a person is
usually reading them; ``--json`` produces machine-readable output for pipelines. The two
are generated from the same data so they cannot disagree.

**Errors are actionable, not tracebacks.** A :class:`~nas_engine.exceptions.NasEngineError`
is caught at the top level and rendered as a short message plus its structured details. An
unexpected exception still shows a traceback, because that one is a bug and the traceback
is the useful part.

**The CLI is a thin shell.** Every command loads configuration, calls into the library, and
formats the result. No domain logic lives here, which is why the Python API can do
everything the CLI can.
"""

from __future__ import annotations

import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from nas_engine.config.loader import dump_yaml, load_config
from nas_engine.config.models import SearchConfig
from nas_engine.exceptions import NasEngineError
from nas_engine.observability.logging import configure_logging

console = Console()
error_console = Console(stderr=True)


class ExitCode(IntEnum):
    """Process exit codes.

    Members:
        SUCCESS: The command completed.
        CONFIGURATION_ERROR: Configuration was missing or invalid.
        NOT_FOUND: A requested record does not exist.
        SEARCH_FAILED: A search ran but ended in a failed state.
        INTERRUPTED: The operator interrupted the command; the search can be resumed.
        RUNTIME_ERROR: Any other handled failure.
        UNEXPECTED_ERROR: An unhandled exception; this indicates a bug.
    """

    SUCCESS = 0
    CONFIGURATION_ERROR = 2
    NOT_FOUND = 3
    SEARCH_FAILED = 4
    INTERRUPTED = 130
    RUNTIME_ERROR = 1
    UNEXPECTED_ERROR = 70


app = typer.Typer(
    name="nas-engine",
    help=(
        "Neural Architecture Search from scratch.\n\n"
        "Define a search space, run a search, inspect the results, and export a report."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------------
ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to a YAML configuration file. Defaults are used when omitted.",
        show_default=False,
    ),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a configuration value, e.g. --set budget.max_evaluations=20. "
        "Repeatable. Highest precedence.",
        show_default=False,
    ),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable JSON instead of a table."),
]
SearchIdOption = Annotated[
    str | None,
    typer.Option("--search-id", help="Target a specific search. Defaults to the most recent."),
]


def _load(config: Path | None, overrides: list[str] | None) -> SearchConfig:
    """Load configuration through the full precedence chain.

    Args:
        config: YAML file path.
        overrides: ``key=value`` assignments.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If loading or validation fails.
    """
    return load_config(config, overrides=overrides or [])


def _emit(payload: Any, *, as_json: bool, renderer: Any = None) -> None:
    """Print either JSON or a human-readable rendering.

    Args:
        payload: JSON-serialisable data.
        as_json: Whether to emit JSON.
        renderer: Callable printing the human-readable form. Defaults to printing the
            payload with Rich.
    """
    if as_json:
        console.print_json(json.dumps(payload, default=str))
        return
    if renderer is not None:
        renderer()
    else:
        console.print(payload)


def _open_engine(config: SearchConfig) -> Any:
    """Construct a :class:`~nas_engine.orchestration.engine.SearchEngine`.

    Imported lazily so that ``nas-engine --help`` does not pay for importing PyTorch,
    which dominates start-up time.

    Args:
        config: Validated configuration.

    Returns:
        The engine.
    """
    from nas_engine.orchestration.engine import SearchEngine

    return SearchEngine(config)


def _open_repository(config: SearchConfig) -> Any:
    """Open the results database read-only-ish, without building a full engine.

    Inspection commands must not need a dataset or a GPU, so they bypass the engine
    entirely and talk to the repository.

    The output directory is created if absent. Running ``status`` before the first search
    should report "no search found", not a SQLite "unable to open database file" error.

    Args:
        config: Validated configuration.

    Returns:
        A ``(database, repository)`` pair; the caller disposes the database.
    """
    from nas_engine.persistence.database import Database
    from nas_engine.persistence.migrations import ensure_schema
    from nas_engine.persistence.repository import SearchRepository
    from nas_engine.utilities.paths import ensure_directory

    ensure_directory(config.output_dir)
    database = Database(config.database_url, echo=config.persistence.echo_sql)
    ensure_schema(database)
    return database, SearchRepository(database)


def _resolve_search_id(repository: Any, search_id: str | None, config: SearchConfig) -> str:
    """Resolve an explicit or implicit search id.

    Args:
        repository: Repository to query.
        search_id: Explicit identifier, or ``None``.
        config: Configuration, used to prefer searches with a matching project name.

    Returns:
        The resolved identifier.

    Raises:
        typer.Exit: If no search can be found.
    """
    if search_id:
        return search_id
    summary = repository.find_latest_search(name=config.project.name)
    if summary is None:
        summary = repository.find_latest_search()
    if summary is None:
        error_console.print(
            "[red]No search found[/red] in "
            f"{config.database_url}.\nStart one with: nas-engine search --config <file>"
        )
        raise typer.Exit(int(ExitCode.NOT_FOUND))
    return str(summary.id)


# ---------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------
@app.command()
def init(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the configuration file.")
    ] = Path("configs/search.yaml"),
    preset: Annotated[
        str, typer.Option("--preset", help="Search-space preset to start from.")
    ] = "default_cnn",
    strategy: Annotated[
        str, typer.Option("--strategy", help="Search strategy to configure.")
    ] = "random_search",
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter configuration file.

    Examples:
        nas-engine init

        nas-engine init -o configs/evolution.yaml --strategy regularized_evolution
    """
    if output.exists() and not force:
        error_console.print(f"[red]{output} already exists.[/red] Pass --force to overwrite it.")
        raise typer.Exit(int(ExitCode.CONFIGURATION_ERROR))

    config = SearchConfig.from_mapping(
        {
            "project": {"name": output.stem, "output_dir": "artifacts"},
            "search_space": {"preset": preset},
            "algorithm": {"name": strategy},
        }
    )
    dump_yaml(config, output)
    console.print(f"[green]Wrote[/green] {output}")
    console.print("\nNext steps:")
    console.print(f"  nas-engine validate-config --config {output}")
    console.print(f"  nas-engine search --config {output}")


@app.command(name="validate-config")
def validate_config(
    config: ConfigOption = None,
    overrides: SetOption = None,
    as_json: JsonOption = False,
) -> None:
    """Validate a configuration and print what it will do.

    Exits with code 2 when the configuration is invalid, listing every offending field.

    Examples:
        nas-engine validate-config --config configs/random_search.yaml

        nas-engine validate-config -c configs/evolution.yaml --set budget.max_evaluations=50
    """
    loaded = _load(config, overrides)
    space = loaded.search_space.build()
    payload = {
        "valid": True,
        "config_hash": loaded.config_hash(),
        "version": loaded.version,
        "search_space": {
            "name": space.name,
            "log10_cardinality": round(space.log10_cardinality(), 2),
        },
        "config": loaded.to_dict(),
    }

    def render() -> None:
        console.print("[green]Configuration is valid.[/green]\n")
        console.print(loaded.describe())
        console.print("")
        console.print(space.describe())

    _emit(payload, as_json=as_json, renderer=render)


@app.command()
def search(
    config: ConfigOption = None,
    overrides: SetOption = None,
    as_json: JsonOption = False,
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Generate a report when the search ends.")
    ] = False,
) -> None:
    """Run a new architecture search.

    Examples:
        nas-engine search --config configs/random_search.yaml

        nas-engine search -c configs/evolution.yaml --set budget.max_evaluations=30 --report
    """
    loaded = _load(config, overrides)
    engine = _open_engine(loaded)
    try:
        result = engine.run()
        if report:
            _generate_report(engine, loaded, result.search_id)
        _emit(result.to_dict(), as_json=as_json, renderer=lambda: console.print(result.summary()))
        if result.stop_reason.value == "interrupted":
            raise typer.Exit(int(ExitCode.INTERRUPTED))
        if result.status == "failed":
            raise typer.Exit(int(ExitCode.SEARCH_FAILED))
    finally:
        engine.close()


@app.command()
def resume(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Resume an interrupted search.

    Candidates left mid-evaluation by a crashed process are returned to the queue (or
    failed, if their retries are exhausted) before any new work is proposed.

    Examples:
        nas-engine resume --config configs/evolution.yaml

        nas-engine resume -c configs/evolution.yaml --search-id 3f2a...
    """
    loaded = _load(config, overrides)
    engine = _open_engine(loaded)
    try:
        result = engine.resume(search_id)
        _emit(result.to_dict(), as_json=as_json, renderer=lambda: console.print(result.summary()))
        if result.stop_reason.value == "interrupted":
            raise typer.Exit(int(ExitCode.INTERRUPTED))
    finally:
        engine.close()


@app.command()
def status(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show the status of a search.

    Examples:
        nas-engine status --config configs/random_search.yaml

        nas-engine status --json
    """
    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        summary = repository.get_search(target)
        counts = repository.count_candidates_by_status(target)
        checkpoints = repository.count_checkpoints(target)
        payload = {
            "search": summary.to_dict(),
            "counts": counts,
            "checkpoints": checkpoints,
        }

        def render() -> None:
            table = Table(title=f"Search {summary.id}", show_header=False, box=None)
            table.add_row("name", summary.name)
            table.add_row("strategy", summary.strategy)
            table.add_row("status", summary.status)
            table.add_row("seed", str(summary.seed))
            table.add_row("config hash", summary.config_hash)
            table.add_row("created", summary.created_at.isoformat())
            table.add_row("started", summary.started_at.isoformat() if summary.started_at else "—")
            table.add_row(
                "completed", summary.completed_at.isoformat() if summary.completed_at else "—"
            )
            duration = summary.duration_seconds
            table.add_row("duration", f"{duration:.1f}s" if duration is not None else "—")
            table.add_row("checkpoints", str(checkpoints))
            console.print(table)

            state_table = Table(title="Candidates by state")
            state_table.add_column("state")
            state_table.add_column("count", justify="right")
            for name, count in counts.items():
                state_table.add_row(name, str(count))
            console.print(state_table)

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command(name="list-candidates")
def list_candidates(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    state: Annotated[
        str | None, typer.Option("--state", help="Filter by candidate state, e.g. completed.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum rows to show.")] = 20,
    as_json: JsonOption = False,
) -> None:
    """List a search's candidates.

    Examples:
        nas-engine list-candidates --limit 50

        nas-engine list-candidates --state failed --json
    """
    from nas_engine.orchestration.lifecycle import CandidateState

    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        states = None
        if state:
            try:
                states = [CandidateState(state)]
            except ValueError as exc:
                valid = [member.value for member in CandidateState]
                error_console.print(f"[red]Unknown state '{state}'.[/red] Valid states: {valid}")
                raise typer.Exit(int(ExitCode.CONFIGURATION_ERROR)) from exc

        candidates = repository.list_candidates(
            target, statuses=states, limit=limit, order_by_objective=True
        )
        payload = {"search_id": target, "candidates": [c.to_dict() for c in candidates]}

        def render() -> None:
            table = Table(title=f"Candidates in {target}")
            table.add_column("architecture", overflow="fold")
            table.add_column("state")
            table.add_column("origin")
            table.add_column("accuracy", justify="right")
            table.add_column("params", justify="right")
            table.add_column("retries", justify="right")
            for candidate in candidates:
                accuracy = candidate.metrics.get("validation_accuracy")
                parameters = candidate.metrics.get("trainable_parameters")
                table.add_row(
                    candidate.architecture_hash[:16],
                    candidate.status,
                    candidate.origin,
                    f"{accuracy:.4f}" if accuracy is not None else "—",
                    f"{int(parameters):,}" if parameters is not None else "—",
                    str(candidate.retry_count),
                )
            console.print(table)
            if not candidates:
                console.print("[yellow]No candidates matched.[/yellow]")

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command(name="show-candidate")
def show_candidate(
    candidate: Annotated[str, typer.Argument(help="Candidate id or architecture hash prefix.")],
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show one candidate in full, including its architecture and trial history.

    Examples:
        nas-engine show-candidate a1b2c3d4

        nas-engine show-candidate a1b2c3d4 --json
    """
    from nas_engine.architectures.summary import summarise

    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        record = _find_candidate(repository, target, candidate)
        spec = repository.get_candidate_spec(record.id)
        trials = repository.list_trials(record.id)
        architecture = summarise(spec)
        payload = {
            "candidate": record.to_dict(),
            "architecture": {
                "hash": architecture.architecture_hash,
                "cost": architecture.cost.to_dict(),
                "layers": architecture.trace.to_rows(),
            },
            "trials": trials,
        }

        def render() -> None:
            console.print(architecture.to_text())
            console.print("")
            table = Table(title="Trials")
            table.add_column("attempt", justify="right")
            table.add_column("status")
            table.add_column("duration", justify="right")
            table.add_column("worker")
            table.add_column("error", overflow="fold")
            for trial in trials:
                error = trial.get("error") or {}
                table.add_row(
                    str(trial["attempt"]),
                    trial["status"],
                    f"{trial['duration_seconds']:.2f}s",
                    trial.get("worker_id") or "—",
                    str(error.get("message", ""))[:80],
                )
            console.print(table)

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command()
def best(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show the best candidate under the configured objectives.

    Examples:
        nas-engine best

        nas-engine best --json
    """
    from nas_engine.architectures.summary import summarise
    from nas_engine.objectives.ranking import rank_candidates

    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        ranking = rank_candidates(
            repository.completed_metrics(target),
            loaded.objectives.build_objectives(),
            constraints=loaded.objectives.build_constraints(),
        )
        if ranking.best is None:
            error_console.print(f"[yellow]No completed candidate in search {target}.[/yellow]")
            raise typer.Exit(int(ExitCode.NOT_FOUND))

        winner = ranking.best
        spec = repository.get_candidate_spec(winner.candidate_id)
        architecture = summarise(spec)
        payload = {
            "search_id": target,
            "best": winner.to_dict(),
            "architecture_summary": architecture.compact(),
        }

        def render() -> None:
            console.print(f"[green]Best candidate in search {target}[/green]\n")
            console.print(architecture.to_text())
            console.print("")
            score = winner.score
            console.print(f"score: {score:.4f}" if score is not None else "score: n/a")
            console.print(f"pareto rank: {winner.pareto_rank}")

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command()
def pareto(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show the Pareto front of feasible candidates.

    Examples:
        nas-engine pareto

        nas-engine pareto --json > front.json
    """
    from nas_engine.objectives.ranking import rank_candidates

    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        objectives = loaded.objectives.build_objectives()
        ranking = rank_candidates(
            repository.completed_metrics(target),
            objectives,
            constraints=loaded.objectives.build_constraints(),
        )
        payload = {
            "search_id": target,
            "objectives": [objective.describe() for objective in objectives.objectives],
            "pareto_front": [candidate.to_dict() for candidate in ranking.pareto_front],
        }

        def render() -> None:
            console.print(objectives.describe())
            console.print("")
            table = Table(title=f"Pareto front ({len(ranking.pareto_front)} candidates)")
            table.add_column("architecture", overflow="fold")
            for objective in objectives.objectives:
                table.add_column(objective.metric, justify="right")
            table.add_column("score", justify="right")
            for candidate in ranking.pareto_front:
                row = [candidate.architecture_hash[:16]]
                for objective in objectives.objectives:
                    value = candidate.metrics.get(objective.metric)
                    row.append(f"{value:,.4f}" if value is not None else "—")
                row.append(f"{candidate.score:.4f}" if candidate.score is not None else "—")
                table.add_row(*row)
            console.print(table)
            if not ranking.pareto_front:
                console.print("[yellow]No feasible candidate completed.[/yellow]")

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command()
def evaluate(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    candidate: Annotated[
        str | None,
        typer.Option("--candidate", help="Candidate to evaluate. Defaults to the best."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Evaluate a trained candidate on the held-out test split.

    The test split is used **only** here, never during a search. Running this more than
    once on the same search reintroduces the selection bias it exists to avoid.

    Examples:
        nas-engine evaluate --config configs/random_search.yaml

        nas-engine evaluate --candidate a1b2c3d4 --json
    """
    loaded = _load(config, overrides)
    engine = _open_engine(loaded)
    try:
        repository = engine.repository
        target = _resolve_search_id(repository, search_id, loaded)
        if candidate:
            record = _find_candidate(repository, target, candidate)
        else:
            ranking = engine.ranking(target)
            if ranking.best is None:
                error_console.print(f"[yellow]No completed candidate in search {target}.[/yellow]")
                raise typer.Exit(int(ExitCode.NOT_FOUND))
            record = repository.get_candidate(ranking.best.candidate_id)

        spec = repository.get_candidate_spec(record.id)
        weights = record.artifacts.get("weights")
        weights_path = engine.artifact_root / weights if weights else None
        metrics = engine.evaluator.evaluate_on_test(spec, weights_path=weights_path)
        payload = {
            "search_id": target,
            "candidate_id": record.id,
            "architecture_hash": record.architecture_hash,
            "weights": weights,
            "test_metrics": metrics,
        }

        def render() -> None:
            console.print(f"[green]Test evaluation[/green] of {record.architecture_hash}")
            table = Table(show_header=False, box=None)
            for name, value in sorted(metrics.items()):
                table.add_row(name, f"{value:.4f}")
            console.print(table)
            if weights is None:
                console.print(
                    "[yellow]Warning:[/yellow] no stored weights were found, so this "
                    "measures an untrained model."
                )

        _emit(payload, as_json=as_json, renderer=render)
    finally:
        engine.close()


@app.command()
def export(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    output_format: Annotated[
        str, typer.Option("--format", help="Export format: csv or json.")
    ] = "csv",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Destination file.")] = None,
) -> None:
    """Export search results as CSV or JSON.

    Examples:
        nas-engine export --format csv -o results.csv

        nas-engine export --format json
    """
    from nas_engine.objectives.ranking import rank_candidates
    from nas_engine.reporting.exporters import export_candidates_csv, export_json

    if output_format not in {"csv", "json"}:
        error_console.print(f"[red]Unknown format '{output_format}'.[/red] Use 'csv' or 'json'.")
        raise typer.Exit(int(ExitCode.CONFIGURATION_ERROR))

    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        ranking = rank_candidates(
            repository.completed_metrics(target),
            loaded.objectives.build_objectives(),
            constraints=loaded.objectives.build_constraints(),
        )
        destination = output or (loaded.report_dir / f"{target}_candidates.{output_format}")
        if output_format == "csv":
            written = export_candidates_csv(list(ranking.ranked), destination)
        else:
            written = export_json({"search_id": target, "ranking": ranking.to_dict()}, destination)
        console.print(f"[green]Exported[/green] {len(ranking.ranked)} candidates to {written}")
    finally:
        database.dispose()


@app.command()
def report(
    config: ConfigOption = None,
    overrides: SetOption = None,
    search_id: SearchIdOption = None,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip figure generation.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Generate a Markdown report with plots, CSV, and JSON exports.

    Examples:
        nas-engine report --config configs/random_search.yaml

        nas-engine report --no-plots --json
    """
    loaded = _load(config, overrides)
    database, repository = _open_repository(loaded)
    try:
        target = _resolve_search_id(repository, search_id, loaded)
        from nas_engine.reporting.report import ReportGenerator

        generator = ReportGenerator(
            repository,
            objectives=loaded.objectives.build_objectives(),
            constraints=loaded.objectives.build_constraints(),
            output_dir=loaded.report_dir,
            artifact_root=loaded.artifact_dir,
        )
        artifacts = generator.generate(target, include_plots=not no_plots)

        def render() -> None:
            console.print(f"[green]Report written[/green] for search {target}")
            console.print(f"  markdown : {artifacts.markdown}")
            console.print(f"  json     : {artifacts.json}")
            console.print(f"  csv      : {artifacts.csv}")
            for name, path in sorted(artifacts.plots.items()):
                console.print(f"  plot     : {name} -> {path}")
            for name, reason in sorted(artifacts.skipped_plots.items()):
                console.print(f"  [yellow]skipped[/yellow]: {name} ({reason})")

        _emit(artifacts.to_dict(), as_json=as_json, renderer=render)
    finally:
        database.dispose()


@app.command()
def doctor(
    config: ConfigOption = None,
    overrides: SetOption = None,
    as_json: JsonOption = False,
) -> None:
    """Diagnose the environment and the configuration.

    Checks Python and package versions, device availability, database access, output
    directory permissions, configuration validity, and seeding.

    Exits non-zero when any check fails.

    Examples:
        nas-engine doctor

        nas-engine doctor --config configs/random_search.yaml --json
    """
    checks = _run_doctor_checks(config, overrides)
    failures = [check for check in checks if check["status"] == "fail"]
    payload = {"checks": checks, "failures": len(failures)}

    def render() -> None:
        table = Table(title="nas-engine doctor")
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail", overflow="fold")
        styles = {"pass": "green", "warn": "yellow", "fail": "red"}
        for check in checks:
            style = styles.get(check["status"], "white")
            table.add_row(
                check["name"], f"[{style}]{check['status'].upper()}[/{style}]", check["detail"]
            )
        console.print(table)
        if failures:
            console.print(f"\n[red]{len(failures)} check(s) failed.[/red]")
        else:
            console.print("\n[green]All checks passed.[/green]")

    _emit(payload, as_json=as_json, renderer=render)
    if failures:
        raise typer.Exit(int(ExitCode.RUNTIME_ERROR))


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def _find_candidate(repository: Any, search_id: str, needle: str) -> Any:
    """Find a candidate by id or architecture-hash prefix.

    Args:
        repository: Repository to query.
        search_id: Search to look in.
        needle: Candidate id, full hash, or hash prefix.

    Returns:
        The candidate summary.

    Raises:
        typer.Exit: If nothing matches, or the prefix is ambiguous.
    """
    from nas_engine.exceptions import RecordNotFoundError

    try:
        return repository.get_candidate(needle)
    except RecordNotFoundError:
        pass

    candidates = repository.list_candidates(search_id)
    matches = [
        candidate
        for candidate in candidates
        if candidate.architecture_hash.startswith(needle) or candidate.id.startswith(needle)
    ]
    if not matches:
        error_console.print(
            f"[red]No candidate in search {search_id} matches '{needle}'.[/red]\n"
            "List candidates with: nas-engine list-candidates"
        )
        raise typer.Exit(int(ExitCode.NOT_FOUND))
    if len(matches) > 1:
        hashes = ", ".join(match.architecture_hash[:12] for match in matches[:5])
        error_console.print(
            f"[red]'{needle}' is ambiguous[/red] — it matches {len(matches)} candidates "
            f"({hashes}...). Use a longer prefix."
        )
        raise typer.Exit(int(ExitCode.CONFIGURATION_ERROR))
    return matches[0]


def _generate_report(engine: Any, config: SearchConfig, search_id: str) -> None:
    """Generate a report immediately after a search, without reopening the database.

    Args:
        engine: The engine that ran the search.
        config: Configuration in use.
        search_id: Search to report on.
    """
    from nas_engine.reporting.report import ReportGenerator

    generator = ReportGenerator(
        engine.repository,
        objectives=config.objectives.build_objectives(),
        constraints=config.objectives.build_constraints(),
        output_dir=config.report_dir,
        artifact_root=config.artifact_dir,
    )
    artifacts = generator.generate(search_id)
    console.print(f"\n[green]Report:[/green] {artifacts.markdown}")


def _run_doctor_checks(  # noqa: PLR0912, PLR0915 - a flat list of independent probes
    config: Path | None, overrides: list[str] | None
) -> list[dict[str, str]]:
    """Run every diagnostic check and return their outcomes.

    Args:
        config: Configuration file path.
        overrides: Configuration overrides.

    Returns:
        One ``{name, status, detail}`` mapping per check.
    """
    import platform

    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    version = sys.version_info
    if version >= (3, 12):
        add("python version", "pass", f"{platform.python_version()} (recommended)")
    elif version >= (3, 10):
        add(
            "python version",
            "warn",
            f"{platform.python_version()}; supported, but 3.12 is the reference version",
        )
    else:  # pragma: no cover - the package cannot be installed on older interpreters
        add("python version", "fail", f"{platform.python_version()} is below the 3.10 minimum")

    try:
        import torch

        add("pytorch", "pass", f"{torch.__version__}")
        if torch.cuda.is_available():  # pragma: no cover - requires a GPU
            names = ", ".join(
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            )
            add("cuda", "pass", f"available: {names} (CUDA {torch.version.cuda})")
        else:
            add("cuda", "warn", "not available; searches will run on CPU")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():  # pragma: no cover - Apple only
            add("mps", "pass", "Apple Silicon acceleration available")
        add("torch threads", "pass", f"{torch.get_num_threads()} intra-op threads")
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        add("pytorch", "fail", f"not importable: {exc}")

    try:
        import torchvision

        add("torchvision", "pass", f"{torchvision.__version__} (CIFAR-10 provider available)")
    except ImportError:
        add(
            "torchvision",
            "warn",
            "not installed; the cifar10 dataset provider is unavailable. "
            "Install with: pip install 'nas-engine[cifar]'",
        )

    try:
        loaded = load_config(config, overrides=overrides or [])
        add("configuration", "pass", f"valid, hash {loaded.config_hash()}")
    except NasEngineError as exc:
        add("configuration", "fail", str(exc).splitlines()[0])
        return checks

    add(
        "random seed",
        "pass",
        f"seed={loaded.reproducibility.seed}, deterministic={loaded.reproducibility.deterministic}",
    )

    try:
        device = loaded.hardware.resolve_device()
        add("device", "pass", f"resolves to {device}")
    except NasEngineError as exc:
        add("device", "fail", str(exc))

    from nas_engine.utilities.paths import ensure_directory

    for label, directory in (
        ("output directory", loaded.output_dir),
        ("artifact directory", loaded.artifact_dir),
        ("report directory", loaded.report_dir),
    ):
        try:
            resolved = ensure_directory(directory)
            add(label, "pass", f"{resolved} is writable")
        except NasEngineError as exc:
            add(label, "fail", str(exc))

    try:
        database, repository = _open_repository(loaded)
        try:
            searches = repository.list_searches(limit=5)
            add(
                "database",
                "pass",
                f"{loaded.database_url} reachable, {len(searches)} recent search(es)",
            )
        finally:
            database.dispose()
    except NasEngineError as exc:
        add("database", "fail", str(exc))

    try:
        space = loaded.search_space.build()
        add(
            "search space",
            "pass",
            f"'{space.name}', about 1e{space.log10_cardinality():.1f} architectures",
        )
    except NasEngineError as exc:
        add("search space", "fail", str(exc))

    return checks


# ---------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------
def main() -> int:
    """Run the CLI, translating domain errors into exit codes.

    Returns:
        The process exit code.
    """
    configure_logging()
    try:
        app(standalone_mode=False)
    except typer.Exit as exit_signal:
        return int(exit_signal.exit_code)
    except click_exceptions() as usage_error:
        # Typer/Click usage errors already printed their own message. `format_message`
        # is resolved eagerly because the bound name is cleared when the block exits.
        formatter = getattr(usage_error, "format_message", None)
        message = formatter() if callable(formatter) else str(usage_error)
        error_console.print(f"[red]{message}[/red]")
        return int(ExitCode.CONFIGURATION_ERROR)
    except (KeyboardInterrupt, abort_exception()):
        error_console.print("\n[yellow]Interrupted.[/yellow] Resume with: nas-engine resume")
        return int(ExitCode.INTERRUPTED)
    except NasEngineError as exc:
        error_console.print(f"[red]{type(exc).__name__}:[/red] {exc.message}")
        if exc.details:
            error_console.print_json(json.dumps(exc.details, default=str))
        return int(_exit_code_for(exc))
    except Exception as exc:
        error_console.print(f"[red]Unexpected error:[/red] {type(exc).__name__}: {exc}")
        error_console.print_exception(show_locals=False)
        return int(ExitCode.UNEXPECTED_ERROR)
    return int(ExitCode.SUCCESS)


def abort_exception() -> type[BaseException]:
    """Return Click's abort exception type.

    Returns:
        ``click.Abort``.
    """
    import click

    return click.Abort


def click_exceptions() -> tuple[type[BaseException], ...]:
    """Return the Click exception types the CLI translates.

    Imported through a function so that a Click version without one of these classes does
    not break the module import.

    Returns:
        A tuple of exception types.
    """
    import click

    return (click.UsageError, click.BadParameter)


def _exit_code_for(error: NasEngineError) -> ExitCode:
    """Map a domain error to an exit code.

    Args:
        error: The error.

    Returns:
        The exit code.
    """
    from nas_engine.exceptions import (
        ConfigurationError,
        RecordNotFoundError,
        SearchSpaceError,
    )

    if isinstance(error, (ConfigurationError, SearchSpaceError)):
        return ExitCode.CONFIGURATION_ERROR
    if isinstance(error, RecordNotFoundError):
        return ExitCode.NOT_FOUND
    return ExitCode.RUNTIME_ERROR


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
