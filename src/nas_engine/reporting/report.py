"""Markdown search reports.

A report exists so that someone who was not watching the run can understand what happened
and decide whether to trust it. That means it must contain the *caveats* as prominently as
the results: what the search cost, how noisy the estimate is, what the numbers do and do
not generalise to.

Sections
--------
1. Headline — best candidate and how the run stopped.
2. Configuration and environment — everything needed to reproduce it.
3. Search statistics — proposed, unique, invalid, pruned, failed, duration.
4. Best architecture — a full layer table and cost breakdown.
5. Pareto front — the trade-offs the search actually found.
6. Figures — accuracy against size and latency, progress over time.
7. Lineage — where the winner came from, for evolutionary searches.
8. Strategy statistics — algorithm-specific detail.
9. Known limitations — the standing caveats, always present.

Determinism
-----------
Filenames derive from the search id, never from a timestamp, so regenerating a report
overwrites in place. The content includes a generation timestamp, which is the one
deliberately non-deterministic element; everything else is a function of the stored data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nas_engine.architectures.lineage import LineageGraph
from nas_engine.architectures.summary import summarise
from nas_engine.evaluation.latency import LATENCY_WARNING
from nas_engine.exceptions import RecordNotFoundError, ReportingError
from nas_engine.objectives.constraints import ConstraintSet
from nas_engine.objectives.objective import ObjectiveSet
from nas_engine.objectives.ranking import RankedCandidate, rank_candidates
from nas_engine.observability.events import Event, emit
from nas_engine.observability.logging import get_logger
from nas_engine.persistence.repository import SearchRepository
from nas_engine.reporting.exporters import export_candidates_csv, export_json
from nas_engine.reporting.plots import PlotResult, generate_plots
from nas_engine.utilities.paths import ensure_directory, safe_filename
from nas_engine.utilities.timing import utc_now_iso

_LOGGER = get_logger(__name__)

#: Standing caveats included in every report. These are properties of NAS, not of a
#: particular run, and omitting them would make the headline number look stronger than it is.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "Neural architecture search does not find a globally optimal architecture. It finds "
    "the best architecture it happened to evaluate, within one search space, under one "
    "training recipe.",
    "Validation accuracy from a short training run is a noisy estimate. With a validation "
    "split of n examples the standard error is roughly sqrt(p(1-p)/n); differences smaller "
    "than a few standard errors are not evidence of a better architecture.",
    "Because the search selects on validation accuracy, the winner's validation number is "
    "optimistically biased. Only the held-out test split gives an unbiased estimate, and "
    "it must be used exactly once.",
    "Latency figures are specific to the machine, thread count, batch size, and library "
    "versions recorded in the environment section. They are not comparable across machines.",
    "A fair comparison between NAS methods must include the cost of the search itself, not "
    "only the quality of the architecture it returned.",
    "The discovered architecture is tuned to this dataset and this training recipe. "
    "Changing the augmentation, the optimiser, or the epoch budget can reorder the results.",
    "Search-space design frequently matters more than the choice of search algorithm. A "
    "strong result here is partly a property of the space, not only of the strategy.",
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Files a report generation produced.

    Attributes:
        markdown: The Markdown report.
        json: The JSON export.
        csv: The CSV export.
        plots: Figure name to path.
        skipped_plots: Figure name to the reason it was not drawn.
    """

    markdown: Path
    json: Path
    csv: Path
    plots: dict[str, Path] = field(default_factory=dict)
    skipped_plots: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "markdown": str(self.markdown),
            "json": str(self.json),
            "csv": str(self.csv),
            "plots": {name: str(path) for name, path in self.plots.items()},
            "skipped_plots": dict(self.skipped_plots),
        }


def _format_metric(value: float | None, *, precision: int = 4) -> str:
    """Format a metric for a Markdown table cell."""
    if value is None:
        return "—"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    return f"{value:.{precision}f}"


def _candidate_table(candidates: Sequence[RankedCandidate], *, limit: int = 20) -> list[str]:
    """Render a Markdown table of candidates.

    Args:
        candidates: Candidates in rank order.
        limit: Maximum rows to include.

    Returns:
        Markdown lines.
    """
    if not candidates:
        return ["_No candidates completed successfully._"]
    lines = [
        "| Rank | Architecture | Accuracy | Parameters | Latency (ms) | Score | Front |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates[:limit]:
        lines.append(
            f"| {candidate.rank} "
            f"| `{candidate.architecture_hash[:12]}` "
            f"| {_format_metric(candidate.metrics.get('validation_accuracy'))} "
            f"| {_format_metric(candidate.metrics.get('trainable_parameters'), precision=0)} "
            f"| {_format_metric(candidate.metrics.get('latency_median_ms'), precision=3)} "
            f"| {_format_metric(candidate.score)} "
            f"| {candidate.pareto_rank if candidate.feasible else '—'} |"
        )
    if len(candidates) > limit:
        lines.append("")
        lines.append(f"_{len(candidates) - limit} further candidates omitted; see the CSV export._")
    return lines


class ReportGenerator:
    """Builds Markdown reports and machine-readable exports from persisted results.

    The generator reads only from the repository, never from a live engine, so a report can
    be produced long after the search finished — from the CLI, from a scheduled job, or
    from a copy of the database on another machine.

    Args:
        repository: Repository to read from.
        objectives: Objectives used to rank candidates.
        constraints: Hard constraints applied when ranking.
        output_dir: Directory for reports and exports.
        artifact_root: Artifact root, used to resolve stored paths.
    """

    def __init__(
        self,
        repository: SearchRepository,
        *,
        objectives: ObjectiveSet,
        constraints: ConstraintSet | None = None,
        output_dir: Path,
        artifact_root: Path | None = None,
    ) -> None:
        self._repository = repository
        self._objectives = objectives
        self._constraints = constraints if constraints is not None else ConstraintSet()
        self._output_dir = ensure_directory(Path(output_dir))
        self._artifact_root = artifact_root

    def generate(
        self, search_id: str, *, include_plots: bool = True, table_limit: int = 20
    ) -> ReportArtifacts:
        """Generate the full report set for one search.

        Args:
            search_id: Search to report on.
            include_plots: Whether to render figures.
            table_limit: Maximum rows in the Markdown leaderboard.

        Returns:
            The generated artifacts.

        Raises:
            RecordNotFoundError: If the search does not exist.
            ReportingError: If a file cannot be written.
        """
        summary = self._repository.get_search(search_id)
        config = self._repository.get_search_config(search_id)
        environment = self._repository.get_search_environment(search_id)
        counts = self._repository.count_candidates_by_status(search_id)
        population = self._repository.completed_metrics(search_id)
        ranking = rank_candidates(population, self._objectives, constraints=self._constraints)
        candidates = list(ranking.ranked)

        prefix = safe_filename(search_id)
        plots = PlotResult(paths={}, skipped={"all": "plot generation disabled"})
        if include_plots:
            history = self._completion_history(search_id)
            plots = generate_plots(
                candidates,
                history,
                self._output_dir / "plots",
                prefix=prefix,
            )

        csv_path = export_candidates_csv(candidates, self._output_dir / f"{prefix}_candidates.csv")
        json_path = export_json(
            {
                "search": summary.to_dict(),
                "configuration": config,
                "environment": environment,
                "counts": counts,
                "ranking": ranking.to_dict(),
                "plots": plots.to_dict(),
                "limitations": list(KNOWN_LIMITATIONS),
                "generated_at": utc_now_iso(),
            },
            self._output_dir / f"{prefix}_results.json",
        )

        markdown = self._render_markdown(
            search_id=search_id,
            summary_dict=summary.to_dict(),
            config=config,
            environment=environment,
            counts=counts,
            candidates=candidates,
            pareto=list(ranking.pareto_front),
            plots=plots,
            table_limit=table_limit,
        )
        markdown_path = self._output_dir / f"{prefix}_report.md"
        try:
            markdown_path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            msg = f"could not write report to {markdown_path}: {exc}"
            raise ReportingError(
                msg, details={"path": str(markdown_path), "error": str(exc)}
            ) from exc

        emit(
            Event.REPORT_GENERATED,
            search_id=search_id,
            markdown=str(markdown_path),
            candidates=len(candidates),
            plots=len(plots.paths),
        )
        return ReportArtifacts(
            markdown=markdown_path.resolve(),
            json=json_path,
            csv=csv_path,
            plots=plots.paths,
            skipped_plots=plots.skipped,
        )

    def _completion_history(self, search_id: str) -> list[tuple[int, float]]:
        """Return validation accuracy in completion order.

        Args:
            search_id: Search to read.

        Returns:
            ``(index, accuracy)`` pairs.
        """
        from nas_engine.orchestration.lifecycle import CandidateState

        completed = self._repository.list_candidates(search_id, statuses=[CandidateState.COMPLETED])
        history: list[tuple[int, float]] = []
        for index, candidate in enumerate(completed):
            accuracy = candidate.metrics.get("validation_accuracy")
            if accuracy is not None:
                history.append((index, float(accuracy)))
        return history

    def _population_history(self, search_id: str) -> list[tuple[int, float, float, float]]:
        """Return per-generation population statistics for evolutionary searches.

        Reconstructed from candidate generations rather than from strategy state, so it
        works for any search whose candidates recorded a generation index.

        Args:
            search_id: Search to read.

        Returns:
            ``(generation, best, mean, worst)`` tuples.
        """
        from nas_engine.orchestration.lifecycle import CandidateState

        completed = self._repository.list_candidates(search_id, statuses=[CandidateState.COMPLETED])
        buckets: dict[int, list[float]] = {}
        for candidate in completed:
            if candidate.generation is None or candidate.objective_value is None:
                continue
            buckets.setdefault(candidate.generation, []).append(candidate.objective_value)
        return [
            (generation, max(values), sum(values) / len(values), min(values))
            for generation, values in sorted(buckets.items())
            if values
        ]

    def _render_markdown(
        self,
        *,
        search_id: str,
        summary_dict: dict[str, Any],
        config: dict[str, Any],
        environment: dict[str, Any],
        counts: dict[str, int],
        candidates: Sequence[RankedCandidate],
        pareto: Sequence[RankedCandidate],
        plots: PlotResult,
        table_limit: int,
    ) -> str:
        """Assemble the Markdown document.

        Args:
            search_id: Search identifier.
            summary_dict: Search summary as plain data.
            config: Stored configuration.
            environment: Stored environment snapshot.
            counts: Candidate counts per state.
            candidates: Ranked candidates.
            pareto: Pareto-front candidates.
            plots: Generated figures.
            table_limit: Maximum leaderboard rows.

        Returns:
            The Markdown text.
        """
        best = candidates[0] if candidates else None
        lines: list[str] = [
            f"# Search report: {summary_dict['name']}",
            "",
            f"- **Search id**: `{search_id}`",
            f"- **Strategy**: {summary_dict['strategy']}",
            f"- **Status**: {summary_dict['status']}",
            f"- **Seed**: {summary_dict['seed']}",
            f"- **Configuration hash**: `{summary_dict['config_hash']}`",
            f"- **Started**: {summary_dict['started_at'] or '—'}",
            f"- **Completed**: {summary_dict['completed_at'] or '—'}",
            "- **Duration**: "
            + (
                f"{summary_dict['duration_seconds']:.1f}s"
                if summary_dict.get("duration_seconds") is not None
                else "—"
            ),
            f"- **Report generated**: {utc_now_iso()}",
            "",
        ]

        if best is not None:
            accuracy = _format_metric(best.metrics.get("validation_accuracy"))
            parameters = _format_metric(best.metrics.get("trainable_parameters"), precision=0)
            lines.extend(
                [
                    "## Headline",
                    "",
                    f"The best architecture found is `{best.architecture_hash}`, reaching a "
                    f"validation accuracy of **{accuracy}** with **{parameters}** trainable "
                    "parameters.",
                    "",
                    "This number is a *selected* validation score and is therefore "
                    "optimistically biased; see Known limitations.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Headline",
                    "",
                    "No candidate completed successfully. Check the failure counts below and "
                    "the search log for the recorded error codes.",
                    "",
                ]
            )

        lines.extend(
            [
                "## Search statistics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Planned evaluations | {summary_dict['planned_evaluations']} |",
                f"| Candidates proposed | {sum(counts.values())} |",
                f"| Completed | {counts.get('completed', 0)} |",
                f"| Failed | {counts.get('failed', 0)} |",
                f"| Pruned (constraint) | {counts.get('pruned', 0)} |",
                f"| Cancelled | {counts.get('cancelled', 0)} |",
                f"| Still queued | {counts.get('queued', 0)} |",
                f"| Unique architectures ranked | {len(candidates)} |",
                f"| Pareto-front size | {len(pareto)} |",
                "",
                "## Configuration",
                "",
                "```yaml",
                _render_config_excerpt(config),
                "```",
                "",
                "## Environment",
                "",
            ]
        )
        lines.extend(_render_environment(environment))

        lines.extend(["", "## Leaderboard", ""])
        lines.extend(_candidate_table(candidates, limit=table_limit))

        lines.extend(["", "## Pareto front", ""])
        if pareto:
            lines.append(
                "These candidates are not dominated by any other: each is better than every "
                "alternative on at least one objective. Choosing between them requires a "
                "preference the search cannot supply."
            )
            lines.append("")
            lines.extend(_candidate_table(pareto, limit=table_limit))
        else:
            lines.append("_No feasible candidate completed, so no Pareto front exists._")

        if best is not None:
            lines.extend(["", "## Best architecture", ""])
            lines.extend(self._render_architecture(best))

        lines.extend(["", "## Figures", ""])
        if plots.paths:
            for name, path in sorted(plots.paths.items()):
                title = name.replace("_", " ").capitalize()
                relative = _relative_to(path, self._output_dir)
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({relative})")
                lines.append("")
            lines.append(f"> {LATENCY_WARNING}")
        else:
            lines.append("_No figures were generated._")
        for name, reason in sorted(plots.skipped.items()):
            lines.append(f"- `{name}` was not drawn: {reason}")

        lineage_lines = self._render_lineage(search_id, best)
        if lineage_lines:
            lines.extend(["", "## Lineage of the best candidate", ""])
            lines.extend(lineage_lines)

        lines.extend(["", "## Known limitations", ""])
        lines.extend(f"{index}. {text}" for index, text in enumerate(KNOWN_LIMITATIONS, start=1))
        lines.append("")
        return "\n".join(lines)

    def _render_architecture(self, best: RankedCandidate) -> list[str]:
        """Render the best candidate's architecture summary.

        Args:
            best: The top-ranked candidate.

        Returns:
            Markdown lines.
        """
        try:
            spec = self._repository.get_candidate_spec(best.candidate_id)
        except RecordNotFoundError:
            return ["_The best candidate's specification could not be loaded._"]
        summary = summarise(spec)
        return [summary.to_markdown(), "", "```", summary.compact(), "```"]

    def _render_lineage(self, search_id: str, best: RankedCandidate | None) -> list[str]:
        """Render the ancestry chain of the best candidate.

        Args:
            search_id: Search to read.
            best: The top-ranked candidate.

        Returns:
            Markdown lines; empty when there is no lineage to show.
        """
        if best is None:
            return []
        graph = LineageGraph.from_nodes(self._repository.lineage_nodes(search_id))
        chain = graph.ancestry(best.candidate_id)
        if chain.depth <= 1:
            return ["_The best candidate was sampled directly rather than derived by mutation._"]
        statistics = graph.statistics()
        return [
            "```",
            chain.to_text(),
            "```",
            "",
            f"- Ancestry depth: {chain.depth}",
            f"- Distinct lineages in this search: {statistics['roots']}",
            f"- Candidates produced by mutation: {statistics['mutated_nodes']}",
        ]


def _relative_to(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible, else the absolute path.

    Relative links keep the report portable: the whole report directory can be moved or
    committed and the images still resolve.

    Args:
        path: Target path.
        root: Directory the report lives in.

    Returns:
        A link target.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_config_excerpt(config: dict[str, Any]) -> str:
    """Render the decision-relevant part of a configuration as YAML.

    The full configuration is in the JSON export. The report shows the sections that change
    what the search *means*, so a reader can see the important settings without scrolling
    past twenty lines of logging and path configuration.

    Args:
        config: Stored configuration.

    Returns:
        YAML text.
    """
    import yaml

    excerpt = {
        section: config.get(section)
        for section in (
            "algorithm",
            "budget",
            "search_space",
            "objectives",
            "training",
            "reproducibility",
            "concurrency",
        )
        if section in config
    }
    return yaml.safe_dump(excerpt, sort_keys=True, default_flow_style=False).rstrip()


def _render_environment(environment: dict[str, Any]) -> list[str]:
    """Render the environment snapshot as a Markdown table.

    Args:
        environment: Stored environment snapshot.

    Returns:
        Markdown lines.
    """
    accelerator = environment.get("accelerator", {})
    rows = [
        ("nas-engine version", environment.get("package_version", "—")),
        ("git commit", environment.get("git_commit") or "—"),
        ("Python", environment.get("python_version", "—")),
        ("Platform", environment.get("platform", "—")),
        ("Machine", environment.get("machine", "—")),
        ("PyTorch", environment.get("torch_version", "—")),
        ("CUDA available", accelerator.get("cuda_available", False)),
        ("CUDA version", accelerator.get("cuda_version") or "—"),
        ("Devices", ", ".join(accelerator.get("device_names", [])) or "cpu"),
        ("CPU count", environment.get("cpu_count", "—")),
        ("Torch threads", environment.get("torch_threads", "—")),
    ]
    determinism = environment.get("determinism", {})
    if determinism:
        rows.append(("Deterministic mode", determinism.get("requested", False)))
        for warning in determinism.get("warnings", []):
            rows.append(("Determinism caveat", warning))
    lines = ["| Property | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


__all__ = ["KNOWN_LIMITATIONS", "ReportArtifacts", "ReportGenerator"]
